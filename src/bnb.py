#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
# Import solver library
import pybnb 

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/logs'

spec = importlib.util.spec_from_file_location("module.header", header_file+'/bnb_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

def alpha(f):
    
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

def heuristicPenalty(tmp_pi_bar, tmp_s, init_cost, DT, init_d, x_hat, des_range, auvID):

    pen_dm, pen_dM, pen_abs = 0, 0, 0
    d_max, d_min = header.config.max_distance, header.config.min_distance
    acoustic_loss = alpha(header.config.f)
    snr = []
    tmp = []
    loops = len(tmp_pi_bar)
    tmp_x_hat = x_hat

    old_tmp_x = []
    old_tmp_y = []
    for i in range(loops):
        j_pi_bar = tmp_pi_bar[i]
        tmp_x_hat[0] = tmp_x_hat[0]+DT*tmp_x_hat[2]
        tmp_x_hat[1] = tmp_x_hat[1]+DT*tmp_x_hat[3]
        tmp_d_target = np.sqrt((tmp_x_hat[1]-tmp_s[1])**2+(tmp_x_hat[0]-tmp_s[0])**2)
       
        H = int(((len(j_pi_bar) - 3)/2))
        idx2 = 3 + H
        

        if len(j_pi_bar) >= 5:
            if i != auvID-1:# j pi bar order is the same for everyone
                
                tmp_x = np.cos(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[0]
                tmp_y = np.sin(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[1]
                old_tmp_x.append(tmp_x)
                old_tmp_y.append(tmp_y)
                tmp = np.sqrt((tmp_y-tmp_s[1])**2+(tmp_x-tmp_s[0])**2)
    
                TL = 20*np.log(tmp) + (tmp*acoustic_loss*1e-3)
                
                tmp_snr = header.config.SL - TL - header.config.NL + header.config.DI
                tmp_snr = tmp_snr/(header.config.SL - header.config.NL + header.config.DI)

                if tmp_snr >= header.config.DThresh:
                
                    snr.append(tmp_snr)
                    
                else:
                    snr.append(0)



            # TODO: Generalize the formula
            if (auvID == 1 and i == 1) or (auvID == 2 and i != 1) or (auvID == 3 and i == 1):
            
                if tmp >= d_max:
                    pen_dM = -(init_cost/header.config.H)/2 - pen_dM
                if tmp <= d_min:
                    pen_dm = -(init_cost/header.config.H)/2 - pen_dm
    

        # DON T GO AWAY AFTER DETECTION
        if tmp_d_target > init_d:

            pen_abs = -(init_cost/header.config.H)
        # DON'T CONVERGE COMPLETELY TO THE TARGET 
        if tmp_d_target <= des_range:
        
            pen_abs = -(init_cost/header.config.H)
            print('-----------------------------to near!',pen_abs)



    '''tmp_ij = []
    print('old_x_state:',old_tmp_x)
    print('old_y_state',old_tmp_y)
    for i in range(len(old_tmp_x)- 1):
        tmp_ij.append(np.sqrt((old_tmp_y[i]-old_tmp_y[i+1])**2+(old_tmp_x[i]-old_tmp_x[i+1])**2))
    tmp_ij.append(np.sqrt((old_tmp_y[-1]-old_tmp_y[0])**2+(old_tmp_x[-1]-old_tmp_x[0])**2))

    for i in range(len(tmp_ij)):'''

    # SNR between the AUVs with id div from id of vehicle
    tmp_ij = np.sqrt((old_tmp_y[1]-old_tmp_y[0])**2+(old_tmp_x[1]-old_tmp_x[0])**2)
    TL = 20*np.log(tmp_ij) + (tmp_ij*acoustic_loss*1e-3)   
    tmp_snr = header.config.SL - TL - header.config.NL + header.config.DI
    tmp_snr = tmp_snr/(header.config.SL - header.config.NL + header.config.DI)
    
    if tmp_snr >= header.config.DThresh and tmp_ij>0:
        snr.append(tmp_snr)
    else:
        snr.append(0)

    laplacian = np.zeros((loops,loops))

    laplacian[0,1] = -snr[0]
    laplacian[1,0] = -snr[0]
    laplacian[0,2] = -snr[1]
    laplacian[2,0] = -snr[1]
    if len(snr) > 2:
        laplacian[0,0] = snr[0]+snr[1]+snr[2]
        laplacian[1,1] = snr[0]+snr[1]+snr[2]
        laplacian[2,2] = snr[1]+snr[1]+snr[2]
        laplacian[1,2] = -snr[2]
        laplacian[2,1] = -snr[2]
    else:
        laplacian[1,2] = 0
        laplacian[2,1] = 0
        laplacian[0,0] = snr[0]+snr[1]
    laplacian[1,1] = snr[0]+snr[1]
    laplacian[2,2] = snr[1]+snr[1]

    '''if auvID == 1 :
        laplacian[0,1] = -snr[0]
        laplacian[1,0] = -snr[0]
        laplacian[0,2] = 0
        laplacian[2,0] = 0

        laplacian[0,0] = snr[0]
        laplacian[1,1] = snr[0]+snr[2]
        laplacian[2,2] = snr[2]
        laplacian[1,2] = -snr[2]
        laplacian[2,1] = -snr[2]

    if auvID==3:
        laplacian[0,1] = -snr[2]
        laplacian[1,0] = -snr[2]
        laplacian[0,2] = 0
        laplacian[2,0] = 0

        laplacian[0,0] = snr[2]
        laplacian[1,1] = snr[1]+snr[2]
        laplacian[2,2] = snr[1]
        laplacian[1,2] = -snr[1]
        laplacian[2,1] = -snr[1]
'''



    [U, S, vh] = np.linalg.svd(laplacian)
    max_sigma = S[1]
    if auvID == 20:
        print('SECOND SINGULAR VALUE',max_sigma)
        print('LAPLACIAN',laplacian)
        print('det LAPLACIAN',np.linalg.det(laplacian))
    if max_sigma < 0:
        max_sigma = 0

        max_sigma = 0

    """ laplacian2 = np.zeros((3,3))
    laplacian2[0,:] = [3, -2, -2]
    laplacian2[1,:] = [-2, 3, -2]
    laplacian2[2,:] = [-2, -2, 3]
    [U, S, vh] = np.linalg.svd(laplacian2)
    prova = S[1]
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',prova) """    

    return max_sigma, pen_dm, pen_dM, pen_abs
    

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, x_hat, s, ctrl_cmds,
                    pi_bar, init_state, init_d, v_n = 0, cpf_control = []):
               
        # Basic parameters initialization
        self.auvNum = auvNum
        self.auvID = auvID
        sensors = []
        for i in range(auvNum):
            sensors.append(header.sensor.Sensor(str(i),1,0,0.000))
        self.sensors = sensors
        self.controller = cpf_control
        self.ctrl_cmds = ctrl_cmds

        # Belief state initialization
        self.pi_bar = pi_bar
        self._x_hat = x_hat
        self.P = np.eye(4)
        self._s = s
        self.init_d = init_d
        
        # Temporary variable for plotting pareto solution

        self.cost_c = []
        self.cost_g = []

        # Optimization Parameters
        self.value = 0 #initial value objective function 
        self.DT = header.config.DT
        self.choices = []
        self.des_range = header.config.RANGE_TO_TARGET
        self.inf = float("inf")
        self._bound = self.inf #greedy search'''
        
        # Variables for path init
        tmp = np.zeros((self.auvNum,3))
        for i in range(len(tmp)):
            for j in range(3):
                tmp[i,j] = init_state[j+(i*3)]

        self.init_s_state = tmp
        self.v_n = v_n
        self.ref_vels = [self.v_n]

    # Required methods for graph generation and searching
    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self.value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s, self.value, self._bound,
                        self.choices, self.pi_bar, self.ref_vels, self.cost_c, self.cost_g)

    def load_state(self, node):

        (self._x_hat, self._s, self.value, self._bound,
            self.choices, self.pi_bar, self.ref_vels, self.cost_c, self.cost_g) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        cost, x_hat, s, pi_bar = self.value, self._x_hat, self._s, self.pi_bar
        
        tmp_v_n = header.utils.computePursuitVel(x_hat,s,self.init_d)
        
        for i in range(header.config.U):
            
            
            x, phi, tmp_s, tmp_pi_bar = self.simulation(self.ctrl_cmds[i], 
                                                        x_hat, self.P, s, self.sensors, 
                                                        tmp_v_n, self.DT, pi_bar, self.init_s_state)
            
            # Update the sequence of control decisions
            tmp = [self.ctrl_cmds[i]]
            choices = self.choices + tmp
            tmp_ref_vels = self.ref_vels + [tmp_v_n]
            


            father_value = cost
            cost_d = self.des_range/((np.sqrt((x[0]-tmp_s[0])**2+(x[1]-tmp_s[1])**2)))  
            cost_g = 1/header.utils.compute_cost(phi)

            cost_c, pen_dm, pen_dM, pen_abs = heuristicPenalty(tmp_pi_bar, tmp_s,
                                                        cost, self.DT, 
                                                        self.init_d, x, self.des_range, self.auvID)
                                   
            if cost_c > 1.5:
                cost_c = 1.5

            weight = 1.0#0.2
            tmp_list_c = self.cost_c + [cost_c]
            tmp_list_g = self.cost_g + [cost_g]
            child_value = father_value + cost_g + weight*cost_c#+ cost_d + pen_dM + pen_dm + pen_abs
            
            if self.auvID == 20:
                '''print('--------------------HORIZON',len(choices))
                print('----------------------------------------self.value',self.value)
                print('--------------------father value',father_value)
                print('pen_dm',pen_dm)
                print('pen_dM',pen_dM)
                print('pen_abs',pen_abs)'''
                print('w_g*cost_g',cost_g)
                print('w_d*cost_d',cost_c)
                print('--------------------------------------------child_value',child_value)
            # Compute the bound according to the proposed algorithm
            if len(choices) == header.config.H:
                child_value = child_value + 1
                
                if self._bound == +self.inf:
                    self._bound = child_value

                if child_value >= self._bound:
                    self._bound = child_value
            # Branch the tree         
            child = pybnb.Node()
            child.state = (x, tmp_s, child_value, self._bound, choices, tmp_pi_bar, tmp_ref_vels, tmp_list_c, tmp_list_g)
            
            yield child

    def beliefPropagation(self, pi_bar,auvs_xy,ctrl_input,s_pose,v_n,DT):
        pi_bar_out = []
        for i in range(self.auvNum):
            j_pi_bar = pi_bar[i]
            out = []

            H = int(((len(j_pi_bar) - 3)/2))
            idx2 = 3 + H

            if np.sum(j_pi_bar) != 0 or self.auvID == i+1:
                if self.auvID == i+1: # Compute the path of the i-th AUV acoording to the choosen command
                    
                    ax = np.cos(s_pose[2]+ctrl_input)*v_n*DT+s_pose[0]
                    ay = np.sin(s_pose[2]+ctrl_input)*v_n*DT+s_pose[1]
                    out = [ax,ay,s_pose[2]+ctrl_input]
                else:
                                        
                    ax_j = np.cos(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[0]
                    ay_j = np.sin(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[1]
                    auvs_xy[i] = [ax_j,ay_j,j_pi_bar[2]+j_pi_bar[idx2]]
                    out = auvs_xy[i]

                j_pi_bar = np.delete(j_pi_bar,idx2)
                if len(j_pi_bar) > 3:#remove related ref vel
                    j_pi_bar = np.delete(j_pi_bar,3)
                
                for j in range(len(j_pi_bar[3:len(j_pi_bar)])):
                    out.append(j_pi_bar[j+3])
                
            else:
                auvs_xy[i] = [0.0, 0.0, 0.0, 0.0]
                j_pi_bar = np.delete(j_pi_bar,idx2)
                if len(j_pi_bar) > 3:#remove related ref vel
                    j_pi_bar = np.delete(j_pi_bar,3)
                for i in range(len(j_pi_bar)):
                    out.append(0.0)
            pi_bar_out.append(out)
            
        return pi_bar_out, auvs_xy, [ax, ay, s_pose[2]+ctrl_input]

    def simulation(self, ctrl_input, x_hat, P, s_pose, sensors, v_n, DT, pi_bar, init_state):
       
        # Init data structures
        meas_table = []

        # Init classes for tracker and target and propagate target estimation
        target = header.target_module.Target(x_hat, DT, P)
        estimator = header.estimator_module.Estimation()

        # Initialize data structures for agents state, policies of intent and waypoints
        auvs_xy = []

        for i in range(self.auvNum):
            auvs_xy.append([])
            for j in range((len(s_pose)+1+header.config.H)):
                auvs_xy[i].append(0.0)
        
        # Propagate the AUVs state
        pi_bar_out, auvs_xy, s_pose = self.beliefPropagation(pi_bar,auvs_xy, ctrl_input,
                                                            s_pose, v_n, DT)
        
        # Simulate measurements for cost function computation
        for i in range(self.auvNum):
            if self.auvID == i+1:
                tmp = s_pose
            else:
                tmp = auvs_xy[i]
            if np.sum(tmp[0:3]) == 0.0 and self.auvID != (i+1):
                tmp = init_state[i]

            [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.x[0],target.x[1],
                                                                           [tmp[0],tmp[1]],tmp[2])
            arr = [measure_,meas_pos[0],meas_pos[1]]
            meas_table.append(arr)

        # YOU SHOULD CONSIDER ONLY YOUR NEIGHBOURs IN OPTIMIZING THE GEOMTRY
        # TODO: Generalize the formula (brute version for 3 auv)
        if self.auvID == 1:
            meas_table.pop(2)
        elif self.auvID == 3:
            meas_table.pop(0)

        # Compute the regressor according to measurements simulated
        estimator.computeState(meas_table)

        return target.x, estimator.phi, s_pose, pi_bar_out