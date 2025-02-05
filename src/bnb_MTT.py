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

# Parameters optimization problem
gamma_w = header.config.gamma_w
alpha_w = header.config.alpha_w
d_max, d_min = header.config.max_distance, header.config.min_distance
Ts = header.config.Ts
# Network topology
netTopology = header.config.netTopology

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, targetNum, x_hat, s, ctrl_cmds,
                    pi_bar, initState, init_d, acousticParams, k_phi, AUV_failure=False):
               
        # Basic parameters initialization
        self._auvNum = auvNum
        self._auvID = auvID
        self._targetNum = targetNum
        sensors = []
        for i in range(auvNum):
            sensors.append(header.sensor.Sensor(str(i),1,0,0.0))
        self._sensors = sensors
        self._theta = ctrl_cmds
        self._u = [0.2,header.config.AUV_MAX_VEL/2,header.config.AUV_MAX_VEL]
        self._auvFailure = AUV_failure
        
        self._hedingChoices = []
        self._surgeChoices = []

        # Optimization Parameters
        self._value = 0 #initial value objective function 
        self._DT = Ts*auvNum*2 #optimization time window

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
        self._gamma_w = header.config.gamma_w
        self._k_phi = k_phi#is inside a list, TODO consider multi target case
        self.k_phi_goal = 6#20/len(self._netTopology)# TODO validate costant

        #TODO Temporary lists for plotting pareto solution
        self.cost_c, self.cost_g, self.cost_d = [], [], []

        # Acoustic environment and modem parameters
        self._acousticParams = acousticParams

    # Required methods for graph generation and searching
    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self._value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s_i, self._value, self._bound, self._pi_bar,
                    self._hedingChoices, self._surgeChoices)

    def load_state(self, node):

        (self._x_hat, self._s_i, self._value, self._bound, self._pi_bar,
        self._hedingChoices, self._surgeChoices) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        cost, x_hat, s, pi_bar = self._value, self._x_hat, self._s_i, self._pi_bar
        
        tmp_v_n = header.utils.computePursuitVel(x_hat,s,self._d0)
        self._u[2] = tmp_v_n

        for i in range(len(self._theta)):
            for j in range(len(self._u)):#TODO : OPTIMIZE ALSO SURGE
            
                tmp_xi, tmp_phi, tmp_s, tmp_pi_bar = self.simulation(self._theta[i],self._u[j],
                                                            x_hat, self.P, s, pi_bar)

                father_value = cost

                cost_d = self._desRange/((np.sqrt((tmp_xi[0]-tmp_s[0])**2+(tmp_xi[1]-tmp_s[1])**2)))                             
                cost_g = 1/header.utils.compute_cost(tmp_phi,cost_d,self._auvID) # in [0,1]

                cost_c, pen_dm, pen_abs = header.applyConstraints(tmp_pi_bar, tmp_xi, tmp_s, self._DT, 
                                                            self._d0, self._desRange,self._auvID, 
                                                            self._acousticParams, self._auvFailure)
                
                #[cost_g, cost_c] = header.normalizeObjFunc(cost_g,cost_c)
                if pen_abs == 1.0:# or pen_dm == 1.0: #or cost_c < 10e-4:
                    '''print('CONSTRAINED!!')
                    print('pen_abs',pen_abs)
                    print('pen_dm',pen_dm)'''
                    
                    child_value = 0
                else:
                    child_value = father_value + 0.5*cost_d + cost_g + self._gamma_w*cost_c
                    '''if self._k_phi[0] >= self.k_phi_goal:
                        child_value = father_value + cost_g + self._gamma_w*cost_c
                    else:
                        child_value = father_value + 2*cost_d + cost_g + self._gamma_w*cost_c'''
                    
                    ''' if self._auvID == 1:
                        print('OPT also DISTANCE')
                        print('DISTANCE',((np.sqrt((tmp_xi[0]-tmp_s[0])**2+(tmp_xi[1]-tmp_s[1])**2))))
                        child_value = father_value + cost_ + self._gamma_w*cost_c
                        '''
                if self._auvID == 3000:
                    'ADD DEBUG PRINTS HERE'
                    print('Distance',((np.sqrt((tmp_xi[0]-tmp_s[0])**2+(tmp_xi[1]-tmp_s[1])**2))))
                    print('cost_FINAL',cost_g)
                    print('cost_d',cost_d)
                    

                    #print('NODE VALUE',child_value)
                    
                # Compute the bound according to the proposed algorithm
                # Update data result
                #tmp_list_c = self.cost_c + [cost_c]#for plot to remove
                #tmp_list_g = self.cost_g + [cost_g]
                headingChoices = self._hedingChoices + [self._theta[i]]
                surgeChoices = self._surgeChoices + [self._u[j]]

                if len(headingChoices) == header.config.H:
                    child_value = child_value + 1

                    if self._bound == +float("inf"):# UNIFORM COST SEARCH SETUP
                        self._bound = child_value

                    if child_value >= self._bound:
                        self._bound = child_value

                # Branch the tree  
                child = pybnb.Node()
                child.state = (tmp_xi, tmp_s, child_value, self._bound, tmp_pi_bar,
                    headingChoices, surgeChoices)
                
                yield child

    def beliefPropagation(self, pi_bar, N_i, s_i, r_i, u_i):
        pi_bar_out = []

        for i in range(self._auvNum):
            j_pi_bar = pi_bar[i]
            out = []

            H = int((len(j_pi_bar) - 3) / 2)  # Number of horizon steps
            idx1 = 3 + H  # Index of the first surge delta

            # Check if j_pi_bar is valid or if the current AUV should compute its own path
            if (len(j_pi_bar) > 0 and np.sum(j_pi_bar) != 0) or self._auvID == i + 1:
                if self._auvID == i + 1:  # Compute the path for the current AUV
                    ax = np.cos(s_i[2] + r_i) * u_i * self._DT + s_i[0]
                    ay = np.sin(s_i[2] + r_i) * u_i * self._DT + s_i[1]
                    out = [ax, ay, s_i[2] + r_i, r_i, u_i]
                else:
                    ax_j, ay_j, theta_j = j_pi_bar[0], j_pi_bar[1], j_pi_bar[2]
                    for h in range(H):
                        delta_theta_h = j_pi_bar[3 + h]  # Delta theta for step h
                        delta_surge_h = j_pi_bar[idx1 + h]  # Delta surge for step h
                        theta_j += delta_theta_h
                        ax_j += np.cos(theta_j) * delta_surge_h * self._DT
                        ay_j += np.sin(theta_j) * delta_surge_h * self._DT
                        out.extend([delta_theta_h, delta_surge_h])  # Append step deltas to output

                    N_i[i] = [ax_j, ay_j, theta_j]
                    out = [ax_j, ay_j, theta_j] + out

                    if self._auvID != i + 1:
                        # Remove the delta values from j_pi_bar to keep its size consistent
                        j_pi_bar = np.delete(j_pi_bar, np.s_[3:3 + H])  # Remove delta_theta values
                        j_pi_bar = np.delete(j_pi_bar, np.s_[idx1 - H:idx1])  # Remove delta_surge values

                        for j in range(len(j_pi_bar[3:])):
                            out.append(j_pi_bar[j + 3])

            else:  # Default case for invalid j_pi_bar
                N_i[i] = [0.0, 0.0, 0.0]
                j_pi_bar = np.delete(j_pi_bar, np.s_[3:3 + H])  # Remove delta_theta values
                j_pi_bar = np.delete(j_pi_bar, np.s_[idx1 - H:idx1])  # Remove delta_surge values
                out.extend([0.0, 0.0] * H)  # Default deltas for invalid inputs
                out.extend([0.0] * len(j_pi_bar))

            pi_bar_out.append(out)

        return pi_bar_out, N_i, [ax, ay, s_i[2] + r_i, r_i, u_i]

    def simulation(self, delta_theta, delta_u, x_hat, P, s_pose, pi_bar):
        # Initialize data structures and classes
        meas_table = []
        target = header.target_module.Target(x_hat, self._DT, P)
        estimator = header.estimator_module.Estimation()

        # Initialize AUVs state structures
        auvs_xy = [[0.0 for _ in range(len(s_pose) + 1 + header.config.H)]
                for _ in range(self._auvNum)]

        # Propagate the AUVs state
        pi_bar_out, auvs_xy, s_pose = self.beliefPropagation(pi_bar, auvs_xy, s_pose, delta_theta,
                                                            delta_u)

        # Simulate measurements for cost function computation
        for i, sensor in enumerate(self._sensors):
            tmp = s_pose if self._auvID == i + 1 else (auvs_xy[i] if 
                                                       np.any(auvs_xy[i][:3]) else self._S0[i])
            
            measure_, rel_bearing_, meas_pos = sensor.measureBearing(target.x[0], target.x[1], 
                                                                    [tmp[0], tmp[1]], tmp[2])
            meas_table.append([measure_, meas_pos[0], meas_pos[1]])

        # Filter out specific measurements based on network topology and/or failure status
        if self._auvFailure == False:
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