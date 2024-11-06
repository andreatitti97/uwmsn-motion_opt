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

d_max, d_min = header.config.max_distance, header.config.min_distance
Ts = header.config.Ts
# Parameters optimization problem
gamma_w = header.config.gamma_w
alpha_w = header.config.alpha_w
# Network topology
netTopology = header.config.netTopology

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, targetNum, x_hat, s, ctrl_cmds,
                    pi_bar, initState, init_d, NL=0, AUV_failure=False):
               
        # Basic parameters initialization
        self._auvNum = auvNum
        self._auvID = auvID
        self._targetNum = targetNum
        sensors = []
        for i in range(auvNum):
            sensors.append(header.sensor.Sensor(str(i),1,0,0.000))
        self.sensors = sensors
        self._u = ctrl_cmds
        self._auvFailure = AUV_failure
        self._refVels = []

        # Optimization Parameters
        self._value = 0 #initial value objective function 
        self._DT = Ts*auvNum*2 #optimization time window
        self._choices = []
        self._desRange = header.config.RANGE_TO_TARGET
        self._bound = float("inf")

        # Belief state initialization
        self._pi_bar = pi_bar
        self._x_hat = x_hat
        self.P = np.eye(4)
        self._s_i = s
        self._d0 = init_d
        self._S0 = initState
        self._netTopology = netTopology[auvID-1]
        
        #TODO Temporary lists for plotting pareto solution
        self.cost_c, self.cost_g, self.cost_d = [], [], []

        # Acoustic environment and modem parameters
        self.NL = NL

    # Required methods for graph generation and searching
    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self._value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s_i, self._value, self._bound,
                        self._choices, self._pi_bar, self._refVels, self.cost_c, self.cost_g, self.cost_d)

    def load_state(self, node):

        (self._x_hat, self._s_i, self._value, self._bound,
            self._choices, self._pi_bar, self._refVels, self.cost_c, self.cost_g, self.cost_d) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        cost, x_hat, s, pi_bar = self._value, self._x_hat, self._s_i, self._pi_bar
        
        tmp_v_n = header.utils.computePursuitVel(x_hat,s,self._d0)
        
        for i in range(header.config.U):
            #for j in range(1):#TODO : OPTIMIZE ALSO SURGE
            
            x, phi, tmp_s, tmp_pi_bar = self.simulation(self._u[i], 
                                                        x_hat, self.P, s, self.sensors, 
                                                        tmp_v_n, self._DT, pi_bar,
                                                        self._S0, self._auvFailure)

            father_value = cost
            cost_d = self._desRange/((np.sqrt((x[0]-tmp_s[0])**2+(x[1]-tmp_s[1])**2)))  
            cost_g = 1/header.utils.compute_cost(phi) # in [0,1]
            cost_c, pen_dm, pen_abs = header.connectivityCost(tmp_pi_bar, tmp_s, self._DT, 
                                                        self._d0, x, self._desRange,
                                                        self._auvID, self.NL, self._auvFailure)
                    
            if cost_d >= 1.0: #to bound the objective w.r.t to the desired range (parameter)
                cost_d = 1.0 # in [0,1]

            child_value = father_value + (alpha_w)*cost_g + (1-alpha_w)*cost_d  \
                            #+ self.gamma_w*cost_c 
            
            child_value = father_value + cost_g + cost_d  \
                            #+ self.gamma_w*cost_c

            if pen_abs == 1.0 or pen_dm == 1.0:
                child_value = 0

            tmp_list_c = self.cost_c + [cost_c]
            tmp_list_g = self.cost_g + [cost_g]
            tmp_list_d = self.cost_d + [cost_d]

            if self._auvID == 1:
                'ADD DEBUG PRINTS HERE'
                '''print('--------------------HORIZON',len(choices))
                print('----------------------------------------self._value',self._value)
                print('--------------------father value',father_value)
                print('pen_dm',pen_dm)
                print('pen_dM',pen_dM)
                print('pen_abs',pen_abs)'''
                #print('a**cost_g',self.alpha_w*cost_g)
                #print('w_c*cost_c',self.gamma_w*cost_c)
                #print('a*cost_d',self.alpha_w*cost_d)
                #print('--------------------------------------------child_value',child_value)

            # Compute the bound according to the proposed algorithm
            tmp = [self._u[i]]
            
            choices = self._choices + tmp
            tmp_ref_vels = self._refVels + [tmp_v_n]

            if len(choices) == header.config.H:
                child_value = child_value + 1

                if self._bound == +float("inf"):# UNIFORM COST SEARCH SETUP
                    self._bound = child_value

                if child_value >= self._bound:
                    self._bound = child_value

            # Branch the tree  
            child = pybnb.Node()
            child.state = (x, tmp_s, child_value, self._bound, choices,
                    tmp_pi_bar, tmp_ref_vels, tmp_list_c, tmp_list_g, tmp_list_d)
            
            yield child

    def beliefPropagation(self, pi_bar, N_i, r_i, s_i, v_n, DT):
        pi_bar_out = []

        for i in range(self._auvNum):
            j_pi_bar = pi_bar[i]
            out = []

            H = int((len(j_pi_bar) - 3) / 2)
            idx2 = 3 + H

            # Fix the condition to avoid ambiguous truth value evaluation
            if (len(j_pi_bar) > 0 and np.sum(j_pi_bar) != 0) or self._auvID == i + 1:
                if self._auvID == i + 1:  # Compute the path of the i-th AUV according to the chosen command
                    ax = np.cos(s_i[2] + r_i) * v_n * DT + s_i[0]
                    ay = np.sin(s_i[2] + r_i) * v_n * DT + s_i[1]
                    out = [ax, ay, s_i[2] + r_i]
                else:
                    ax_j = np.cos(j_pi_bar[2] + j_pi_bar[idx2]) * j_pi_bar[3] * DT + j_pi_bar[0]
                    ay_j = np.sin(j_pi_bar[2] + j_pi_bar[idx2]) * j_pi_bar[3] * DT + j_pi_bar[1]
                    N_i[i] = [ax_j, ay_j, j_pi_bar[2] + j_pi_bar[idx2]]
                    out = N_i[i]

                    if self._auvID != i + 1:
                        j_pi_bar = np.delete(j_pi_bar, idx2)
                        if len(j_pi_bar) > 3:  # Remove related ref velocity
                            j_pi_bar = np.delete(j_pi_bar, 3)

                        for j in range(len(j_pi_bar[3:])):
                            out.append(j_pi_bar[j + 3])

            else:
                N_i[i] = [0.0, 0.0, 0.0]
                j_pi_bar = np.delete(j_pi_bar, idx2)
                if len(j_pi_bar) > 3:  # Remove related ref velocity
                    j_pi_bar = np.delete(j_pi_bar, 3)

                out.extend([0.0] * len(j_pi_bar))

            pi_bar_out.append(out)

        return pi_bar_out, N_i, [ax, ay, s_i[2]+r_i]

    def simulation(self, ctrl_input, x_hat, P, s_pose, sensors,
                v_n, DT, pi_bar, initState, AUV_failure):
        # Initialize data structures and classes
        meas_table = []
        target = header.target_module.Target(x_hat, DT, P)
        estimator = header.estimator_module.Estimation()

        # Initialize AUVs state structures
        auvs_xy = [[0.0 for _ in range(len(s_pose) + 1 + header.config.H)]
                for _ in range(self._auvNum)]

        # Propagate the AUVs state
        pi_bar_out, auvs_xy, s_pose = self.beliefPropagation(pi_bar, auvs_xy, ctrl_input,
                                                            s_pose, v_n, DT)

        # Simulate measurements for cost function computation
        for i, sensor in enumerate(sensors):
            tmp = s_pose if self._auvID == i + 1 else (auvs_xy[i] if np.any(auvs_xy[i][:3]) else initState[i])
            measure_, rel_bearing_, meas_pos = sensor.measureBearing(target.x[0], target.x[1], 
                                                                    [tmp[0], tmp[1]], tmp[2])
            meas_table.append([measure_, meas_pos[0], meas_pos[1]])

        # Filter out specific measurements based on network topology and/or failure status
        if AUV_failure == False:
            for i in range(self._auvNum):
                if i+1 not in self._netTopology and i+1 != self._auvID:
                    meas_table.pop(i)
        else:
            meas_table.pop(1)

        # Compute the regressor
        estimator.computeState(meas_table)

        return target.x, estimator.phi, s_pose, pi_bar_out

    






'''ord_d = math.floor(math.log(cost_d, 10))
if cost_c != 0.0:
    ord_c = math.floor(math.log(cost_c, 10))
    self.gamma_w = 1**(ord_c-ord_d)
    
else:
    w_c = 0.0
'''