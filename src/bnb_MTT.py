#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
import math
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

def connectivityCost(tmp_pi_bar, tmp_s, DT, init_d,
                     x_hat, des_range, auvID, NL, AUV_failure):

    pen_dm, pen_dM, pen_abs = 0, 0, 0
    snr, tmp, old_tmp_x, old_tmp_y = [], [], [], []
    loops = len(tmp_pi_bar)
    d_max, d_min = header.config.max_distance, header.config.min_distance

    acoustic_loss = alpha(header.config.f) #f is in kHz
    TL_ideal = 20*np.log(d_min) + (d_min*acoustic_loss*1e-3)
    ideal_snr = (header.config.SL - TL_ideal - NL + header.config.DI)#SNR without transmission loss--> to bound values.
   
    for i in range(loops):
        j_pi_bar = tmp_pi_bar[i]
        x_hat[0] = x_hat[0]+DT*x_hat[2]
        x_hat[1] = x_hat[1]+DT*x_hat[3]
        tmp_d_target = np.sqrt((x_hat[1]-tmp_s[1])**2+(x_hat[0]-tmp_s[0])**2)
       
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
                tmp_snr = tmp_snr/ideal_snr

                if tmp_snr >= header.config.DThresh/ideal_snr:
                    if tmp_snr > 1.0:
                        snr.append(0.0)
                    else:
                        snr.append(tmp_snr)
                else:
                    snr.append(0)

            # TODO: Generalize the formula -- Collision Avoidance Constraint
            if (auvID == 1 and i == 1) or (auvID == 2 and i != 1) or (auvID == 3 and i == 1):
                if tmp <= d_min:
                    pen_dm = 1.0
                    #if auvID == 2:
                    #    print('NOT SAFE')

        # Constraint: stay to a certain vicinity of the target
        if tmp_d_target > init_d:
            #if auvID == 2:
                #print('TO FAR')
            pen_abs = 1.0

        elif tmp_d_target <= des_range:
            pen_abs = 1.0
            #print('TO NEAR')

    # Expected SNR between the AUVs with ID != from current ID
    d_ij = np.sqrt((old_tmp_y[1]-old_tmp_y[0])**2+(old_tmp_x[1]-old_tmp_x[0])**2)
    TL = 20*np.log(d_ij) + (d_ij*acoustic_loss*1e-3)   
    tmp_snr = header.config.SL - TL - header.config.NL + header.config.DI
    # normalize the snr according to the desired one, i.e., 1/4 of the SL (0 is not realistic)
    tmp_snr = tmp_snr/ideal_snr
    #if auvID == 1:
    #    print(tmp_snr)
    if tmp_snr >= header.config.DThresh/ideal_snr and d_ij>0:
        if tmp_snr > 1.0:
            snr.append(0.0)
        else:
            snr.append(tmp_snr)
    else:
        snr.append(0)

    laplacian = np.zeros((loops,loops))

    if AUV_failure == False:

        laplacian[0,1] = -snr[0]
        laplacian[1,0] = -snr[0]
        laplacian[0,2] = 0
        laplacian[2,0] = 0

        laplacian[0,0] = snr[0]
        laplacian[1,1] = snr[0]+snr[1]
        laplacian[2,2] = snr[1]
        laplacian[1,2] = -snr[1]
        laplacian[2,1] = -snr[1]

        if header.config.AUV2_bridge == False:
            laplacian = np.zeros((2,2))
            ''' TO DO '''
        else:
            if auvID == 1:
                laplacian[0,1] = -snr[0]
                laplacian[1,0] = -snr[0]
                laplacian[0,2] = 0
                laplacian[2,0] = 0

                laplacian[0,0] = snr[0]
                laplacian[1,1] = snr[0]+snr[2]
                laplacian[2,2] = snr[2]
                laplacian[1,2] = -snr[2]
                laplacian[2,1] = -snr[2]
            if auvID == 3:
                laplacian[0,1] = -snr[2]
                laplacian[1,0] = -snr[2]
                laplacian[0,2] = 0
                laplacian[2,0] = 0

                laplacian[0,0] = snr[2]
                laplacian[1,1] = snr[1]+snr[2]
                laplacian[2,2] = snr[1]
                laplacian[1,2] = -snr[1]
                laplacian[2,1] = -snr[1]
    else:
        laplacian = np.zeros((2,2))
        if auvID == 1:
            laplacian[0,0] = snr[1]
            laplacian[1,1] = snr[1]
            laplacian[0,1] = -snr[1]
            laplacian[1,0] = -snr[1]
        elif auvID == 3:
            laplacian[0,0] = snr[0]
            laplacian[1,1] = snr[0]
            laplacian[0,1] = -snr[0]
            laplacian[1,0] = -snr[0]

    #if auvID == 1:
        #print(laplacian)

    [U, S, vh] = np.linalg.svd(laplacian)
    max_sigma = S[1]

    if max_sigma < 0: #if the graph is disconnected
        max_sigma = 0

    return max_sigma, pen_dm, pen_abs
    

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, x_hat, s, ctrl_cmds,
                    pi_bar, init_state, init_d, NL, AUV_failure, v_n = 0, cpf_control = []):
               
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
        
        #Temporary lists for plotting pareto solution
        self.cost_c = []
        self.cost_g = []
        self.cost_d = []

        # Optimization Parameters
        self.value = 0 #initial value objective function 
        Ts = header.config.Ts
        self.DT = Ts*auvNum*2 #optimization time window
        self.choices = []
        self.des_range = header.config.RANGE_TO_TARGET
        self.inf = float("inf")
        self._bound = self.inf

        # Acoustic environment and modem parameters
        self.NL = NL

        # Parameters estimation problem
        self.gamma_w = header.config.gamma_w
        self.alpha_w = header.config.alpha_w

        # Variables for path init
        '''tmp = np.zeros((self.auvNum,3))
        for i in range(len(tmp)):
            for j in range(3):
                tmp[i,j] = init_state[j+(i*3)]'''

        self.init_s_state = init_state
    
        self.v_n = v_n
        self.ref_vels = [self.v_n]
        self.AUV_failure = AUV_failure

    # Required methods for graph generation and searching
    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self.value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s, self.value, self._bound,
                        self.choices, self.pi_bar, self.ref_vels, self.cost_c, self.cost_g, self.cost_d)

    def load_state(self, node):

        (self._x_hat, self._s, self.value, self._bound,
            self.choices, self.pi_bar, self.ref_vels, self.cost_c, self.cost_g, self.cost_d) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        cost, x_hat, s, pi_bar = self.value, self._x_hat, self._s, self.pi_bar
        
        tmp_v_n = header.utils.computePursuitVel(x_hat,s,self.init_d, self.DT)
        
        for i in range(header.config.U):
            
            x, phi, tmp_s, tmp_pi_bar = self.simulation(self.ctrl_cmds[i], 
                                                        x_hat, self.P, s, self.sensors, 
                                                        tmp_v_n, self.DT, pi_bar,
                                                        self.init_s_state, self.AUV_failure)

            father_value = cost
            cost_d = self.des_range/((np.sqrt((x[0]-tmp_s[0])**2+(x[1]-tmp_s[1])**2)))  
            cost_g = 1/header.utils.compute_cost(phi) # in [0,1]

            cost_c, pen_dm, pen_abs = connectivityCost(tmp_pi_bar, tmp_s, self.DT, 
                                                        self.init_d, x, self.des_range,
                                                        self.auvID, self.NL, self.AUV_failure)
                      
            if cost_d >= 1.0: #to bound the objective w.r.t to the desired range (parameter)
                cost_d = 1.0 # in [0,1]

            tmp_list_c = self.cost_c + [cost_c]
            tmp_list_g = self.cost_g + [cost_g]
            tmp_list_d = self.cost_d + [cost_d]

            child_value = father_value + (self.alpha_w)*cost_g + (1-self.alpha_w)*cost_d  \
                            + self.gamma_w*cost_c 
            child_value = father_value + cost_g + cost_d  \
                            +  cost_c 
            
            if pen_abs == 1.0 or pen_dm == 1.0:
                child_value = father_value

            if self.auvID == 1:
                'ADD DEBUG PRINTS HERE'
                '''print('--------------------HORIZON',len(choices))
                print('----------------------------------------self.value',self.value)
                print('--------------------father value',father_value)
                print('pen_dm',pen_dm)
                print('pen_dM',pen_dM)
                print('pen_abs',pen_abs)'''
                #print('a**cost_g',self.alpha_w*cost_g)
                print('w_c*cost_c',self.gamma_w*cost_c)
                #print('a*cost_d',self.alpha_w*cost_d)
                #print('--------------------------------------------child_value',child_value)

            # Compute the bound according to the proposed algorithm
            tmp = [self.ctrl_cmds[i]]
            
            choices = self.choices + tmp
            tmp_ref_vels = self.ref_vels + [tmp_v_n]

            if len(choices) == header.config.H:
                child_value = child_value + 1
                
                if self._bound == +self.inf:# UNIFORM COST SEARCH SETUP
                    self._bound = child_value

                if child_value >= self._bound:
                    self._bound = child_value

            # Branch the tree  
            child = pybnb.Node()
            child.state = (x, tmp_s, child_value, self._bound, choices, tmp_pi_bar, tmp_ref_vels, tmp_list_c, tmp_list_g, tmp_list_d)
            
            yield child

    def beliefPropagation(self, pi_bar,auvs_xy,ctrl_input,s_pose,v_n,DT):
        pi_bar_out = []
        for i in range(self.auvNum):
            j_pi_bar = pi_bar[i]
            out = []

            H = int(((len(j_pi_bar) - 3)/2))
            idx2 = 3 + H

            if np.sum(j_pi_bar) != 0 or j_pi_bar != [] or self.auvID == i+1:
                if self.auvID == i+1: # Compute the path of the i-th AUV acoording to the choosen command
                    
                    ax = np.cos(s_pose[2]+ctrl_input)*v_n*DT+s_pose[0]
                    ay = np.sin(s_pose[2]+ctrl_input)*v_n*DT+s_pose[1]
                    out = [ax,ay,s_pose[2]+ctrl_input]
                else:
                                        
                    ax_j = np.cos(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[0]
                    ay_j = np.sin(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[1]
                    auvs_xy[i] = [ax_j,ay_j,j_pi_bar[2]+j_pi_bar[idx2]]

                out = auvs_xy[i]
                if self.auvID != i+1:
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

    def simulation(self, ctrl_input, x_hat, P, s_pose, sensors,
                   v_n, DT, pi_bar, init_state, AUV_failure):
       
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
        if AUV_failure == False:
            if self.auvID == 1:
                meas_table.pop(2)
            elif self.auvID == 3:
                meas_table.pop(0)
        elif AUV_failure == True:
            meas_table.pop(1)
        

        # Compute the regressor according to measurements simulated
        estimator.computeState(meas_table)

        return target.x, estimator.phi, s_pose, pi_bar_out
    






'''ord_d = math.floor(math.log(cost_d, 10))
if cost_c != 0.0:
    ord_c = math.floor(math.log(cost_c, 10))
    self.gamma_w = 1**(ord_c-ord_d)
    
else:
    w_c = 0.0
'''