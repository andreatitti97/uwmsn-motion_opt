#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
# Import solver library
import pybnb 

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/uwmsn-sim'+'/logs'

spec = importlib.util.spec_from_file_location("module.header", header_file+'/bnb_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

def heuristicPenalty(choice, tmp_pi_bar, tmp_s, curren_cost, DT, ctrl_cmds, init_d, d_target):

    penalty, penalty_d, penalty_abs = 0, 0, 0
    d_max, d_min = header.config.max_distance, header.config.min_distance
    delta_fd = 2*np.abs(ctrl_cmds[0])
    tmp = []

    for i in range(len(tmp_pi_bar)):
        j_pi_bar = tmp_pi_bar[i]
        #TODO BETTER
        if len(j_pi_bar) == 3+(header.config.H+1)*2:

            idx2 = 7#len(j_pi_bar)-(header.config.H+2)
        elif len(j_pi_bar) == (3+(header.config.H+1)*2)-2:
            idx2 = 6#len(j_pi_bar)-(header.config.H+1)
        elif len(j_pi_bar) == (3+(header.config.H+1)*2)-4:
            idx2 = 5#len(j_pi_bar)-(header.config.H)
        else:
            idx2 = 4

        if len(j_pi_bar) >= 5:
            tmp_x = np.cos(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[0]
            tmp_y = np.sin(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[1]
            tmp = np.sqrt((tmp_y-tmp_s[1])**2+(tmp_x-tmp_s[0])**2)
            
            if tmp >= d_max:
                penalty_d = -curren_cost/2
            if tmp <= d_min:
                penalty_d = -curren_cost/2
 
    if np.abs(choice) >= delta_fd:  
        penalty = -curren_cost/2
    
    if d_target > init_d:

        penalty_abs = -curren_cost/2

    return penalty, penalty_d, penalty_abs

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, x_hat, s, ctrl_cmds, pi_bar, init_state, init_d, v_n = 0, cpf_control = []):
        
        inf = float("inf")
        
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
        
        # Optimization Parameters
        self.value = 0 #initial value objective function 
        self.initial_cost = -header.config.DELTA
        self.DT = header.config.DT
        self._bound = +inf #no bound for now, greedy search
        self.alpha = 0.08 #param for sigmoid activation funct
        self.choices = []

        # Variables for path init
        tmp = np.zeros((self.auvNum,3))
        for i in range(len(tmp)):
            for j in range(3):
                tmp[i,j] = init_state[j+(i*3)]

        self.init_s_state = tmp
        self.v_n = v_n
        self.ref_vels = [self.v_n]

    # Required methods for grapth generation and searching
    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self.value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s, self.value, self._bound, self.choices, self.pi_bar, self.ref_vels)

    def load_state(self, node):

        (self._x_hat, self._s, self.value, self._bound, self.choices, self.pi_bar, self.ref_vels) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        x_hat, s, pi_bar = self._x_hat, self._s, self.pi_bar
        
        tmp_v_n = header.utils.computePursuitVel(x_hat,s,self.init_d)

        for i in range(header.config.U):
            
            
            x, phi, tmp_s, tmp_pi_bar = self.simulation(self.ctrl_cmds[i], x_hat, self.P, s, self.sensors, tmp_v_n, self.DT, pi_bar, self.init_s_state)
            
            # Update the sequence of control decisions
            tmp = [self.ctrl_cmds[i]]
            choices = self.choices + tmp
            tmp_ref_vels = self.ref_vels + [tmp_v_n]
            
            # If we reached the planning horizon we subtract the DELTA to stop the algorithm and choose only terminal nodes
            if len(choices) == header.config.H:
                self.value = self.value - self.initial_cost ##THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENCE
                #self._bound = self.value #- cost1 

            # Compute the cost functions
            father_value = self.value #THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENC
            d_target = 1/(np.sqrt((x[0]-s[0])**2+(x[1]-s[1])**2))
            penalty, penalty_d, penalty_abs = heuristicPenalty(self.ctrl_cmds[i], tmp_pi_bar, tmp_s, self.value, self.DT, self.ctrl_cmds, self.init_d, d_target)
            cost_g = 1/header.utils.compute_cost(phi)
            
            w_d = header.utils.sig(d_target,self.init_d,-self.alpha)
            w_d = 1.0#2.5#2.5#20#2.5

            # IDEAL SCENARIO TOP PARAMS
            #IF w_d = 20 and w_g = 1/100 COMPLETE PURSUIT with FINAL ADJUSTMENTS for OPT GEOM
            #IF w_d = 1 and w_g = 1/10 OPT GEOM only
            # IF 2.5/3.5 and 1/10 OPTIMAL BEAVIOUR
            # TODO: TEST AGAIN COMBINED SIGOMIDS
            w_g = header.utils.sig(d_target,self.init_d,self.alpha) #geometry cost function more relevant in the proximity of the target
            w_g = 1/10#1/10#1/100#1/10
            
            #print('w_g*cost_g',w_g*cost_g)
            #print('w_d*d_target',w_d*d_target)

            child_value = father_value + w_g*cost_g + w_d*d_target + penalty_d  #+ penalty_abs #+ penalty
            #print('MAGNITUDE GEOM',w_g*cost_g)
            #print('MAGNITUDE DIST',w_d*d_target)
            # Branch the tree
            
            child = pybnb.Node()
            child.state = (x, tmp_s, child_value, self._bound, choices, tmp_pi_bar, tmp_ref_vels)

            yield child

    def beliefPropagation(self, pi_bar,auvs_xy,ctrl_input,s_pose,v_n,DT):
        pi_bar_out = []
        for i in range(self.auvNum):
            j_pi_bar = pi_bar[i]
            out = []

            #TODO BETTER
            if len(j_pi_bar) == 3+(header.config.H+1)*2:

                idx2 = 7#len(j_pi_bar)-(header.config.H+2)
            elif len(j_pi_bar) == (3+(header.config.H+1)*2)-2:
                idx2 = 6#len(j_pi_bar)-(header.config.H+1)
            elif len(j_pi_bar) == (3+(header.config.H+1)*2)-4:
                idx2 = 5#len(j_pi_bar)-(header.config.H)
            else:
                idx2 = 4

            if np.sum(j_pi_bar) != 0 or self.auvID == i+1:
                if self.auvID == i+1:
            # Compute the path of the i-th AUV acoording to the choosen command
                    
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
            
        #print('pi bar put',pi_bar_out)
        return pi_bar_out, auvs_xy, [ax, ay, s_pose[2]+ctrl_input]

    def simulation(self, ctrl_input, x_hat, P, s_pose, sensors, v_n, DT, pi_bar, init_state):
       
        # Temporal Variable
        j = 0, 0 #time and counter init
        meas_table = []

        # Init classes for tracker and target
        target = header.target_module.Target(x_hat, DT, P)
        estimator = header.estimator_module.Estimation()

        # Initialize data structures for agents state, policies of intent and waypoints
        auvs_xy = []

        for i in range(self.auvNum):
            auvs_xy.append([])
            for j in range((len(s_pose)+1+header.config.H)):
                auvs_xy[i].append(0.0)
        
        # Propagate the AUVs state
        pi_bar_out, auvs_xy, s_pose = self.beliefPropagation(pi_bar, auvs_xy, ctrl_input, s_pose, v_n, DT)
            
        # Propagate target state estimation and update i-th AUV pose
        tmp = np.zeros((4,1))
        for j in range(4):  
            tmp[j]=target.x[j]
        target.x = target.F*tmp
        
        # Simulate measurements for cost function computation
        for i in range(self.auvNum):
            if self.auvID == i+1:
                tmp = s_pose
            else:
                tmp = auvs_xy[i]
            if np.sum(tmp) == 0.0 and self.auvID != (i+1):
                tmp = init_state[i]

            [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.x[0],target.x[1],[tmp[0],tmp[1]],tmp[2])
            arr = [measure_,meas_pos[0],meas_pos[1]]
            meas_table.append(arr)

        # Compute the regressor according to measurements simulated
        estimator.computeState(meas_table)

        return target.x, estimator.phi, s_pose, pi_bar_out
