#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
# IMPORT SOLVER LIBRARY
import pybnb #THIS MAY CHANGE ACCORDING TO THE APPLICATION
import matplotlib.pyplot as plt
import rospy
# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/uwmsn-sim'+'/logs'

spec = importlib.util.spec_from_file_location("module.header", header_file+'/bnb_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

def heuristicPenalty(choice, tmp_pi_bar, tmp_s, initial_cost, DT, ctrl_cmds):

    penalty, penalty_d = 0, 0
    d_max, d_min, delta_fd = header.config.d*2, header.config.d/2, np.pi/4
    idx = ctrl_cmds.index(min(ctrl_cmds))
    #delta_fd = 2*np.abs(ctrl_cmds[idx-1]) #chose on of the two
    delta_fd = 2*np.abs(ctrl_cmds[0])
    tmp = []

    for i in range(len(tmp_pi_bar)):
        j_pi_bar = tmp_pi_bar[i]

        if len(j_pi_bar) >= 5:
            tmp_x = np.cos(j_pi_bar[2]+j_pi_bar[4])*j_pi_bar[3]*DT+j_pi_bar[0]
            tmp_y = np.sin(j_pi_bar[2]+j_pi_bar[4])*j_pi_bar[3]*DT+j_pi_bar[1]
            tmp = np.sqrt((tmp_y-tmp_s[1])**2+(tmp_x-tmp_s[0])**2)
            
            if tmp >= d_max:
                penalty_d = initial_cost/header.config.H
            if tmp <= d_min:
                penalty_d = initial_cost/header.config.H
    # Add the cost of the node to the sequence
    #if 1 < len(choices) <= header.config.H:
    if np.abs(choice) >= delta_fd:  
        '''for i in range(len(choices)):
            tmp_ = (choices[i]) - (choices[i-1])
            if np.abs(tmp_) >= delta_fd:'''
        penalty = initial_cost/header.config.H

    return penalty, penalty_d

class Simple(pybnb.Problem):
    def __init__(self, auvNum, auvID, x_hat, s, ctrl_cmds, pi_bar, init_state, init_d, v_n = 0, cpf_control = []):
        
        inf = float("inf")
        
        # Belief state init
        self.auvNum = auvNum
        self.auvID = auvID
        sensors = []
        for i in range(auvNum):
            sensors.append(header.sensor.Sensor(str(i),1,0,0.000))
        self.sensors = sensors
        self.controller = cpf_control
        self.ctrl_cmds = ctrl_cmds

        self.pi_bar = pi_bar
        self._x_hat = x_hat
        self.P = np.eye(4)
        self._s = s
        
        self.init_d = init_d
        
        # Optimization Parameters
        self.value = header.config.DELTA
        self.initial_cost = header.config.DELTA
        self.DT = header.config.DT
        self._bound = -inf # initial_cost-100 #lower bound 
        self.choices = []
        self.alpha = 0.08 #param for sigmoid activation funct

        # Variables for path init
        tmp = np.zeros((self.auvNum,3))
        for i in range(len(tmp)):
            for j in range(3):
                tmp[i,j] = init_state[j+(i*3)]

        self.init_s_state = tmp
        
        self.v_n = v_n
        self.ax = [s[0]]
        self.ay = [s[1]]
        self.d = 0 #distance travelled on the path

    # Required methods for grapth generation and searching
    def sense(self):
        return pybnb.minimize

    def objective(self):
        return self.value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s, self.value, self._bound, self.choices, self.ax, self.ay, self.pi_bar)

    def load_state(self, node):

        (self._x_hat, self._s, self.value, self._bound, self.choices, self.ax, self.ay, self.pi_bar) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        x_hat, s, ax, ay, pi_bar = self._x_hat, self._s, self.ax, self.ay, self.pi_bar
        ax_, ay_ = [], []

        for i in range(len(ax)):
       
            ax_.append(ax[i])
            ay_.append(ay[i])
        for i in range(header.config.U):
            x, phi, y, tmp_s, tmp_ax, tmp_ay, tmp_pi_bar, ranges = self.simulation(self.ctrl_cmds[i], x_hat, self.P, s, self.sensors, ax, ay, self.v_n, self.DT, pi_bar, self.init_s_state)
            
            # Update the sequence of control decisions
            tmp = [self.ctrl_cmds[i]]
            choices = self.choices + tmp
         

            # If we reached the planning horizon we subtract the DELTA to stop the algorithm and choose only terminal nodes
            if len(choices) == header.config.H:
                self.value = self.value - self.initial_cost ##THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENCE
                self._bound = self.value #- cost1 
                #tmp_pi_bar.append([0.0])#TODO: HEURISTIC FUNCTION 
                
                

            father_value = self.value #THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENC
            

            # Compute the cost functions
            penalty, penalty_d = heuristicPenalty(self.ctrl_cmds[i], tmp_pi_bar, tmp_s, self.initial_cost, self.DT, self.ctrl_cmds)
            cost_g = header.utils.compute_cost(phi, ranges, self.init_d, self.auvID)
            d_target = (np.sqrt((x[0]-s[0])**2+(x[1]-s[1])**2))#/self.init_d
            w_d = header.utils.sig(d_target,self.init_d,self.alpha)
            #w_d = 0
            w_g = header.utils.sig(d_target,self.init_d,-self.alpha)
            w_g = 1
            
            
            # weights should be between 0 an 1 and change according to the distance.
            # other option is to make the reweighted estimation w.r.t. the range.
            # in theory you weight differently the measures in order to give more importance to near measures0
            '''if self.auvID == 1:
                print(cost_g)
                print(d_target)
                print(penalty)
                print(penalty_d)'''
            child_value = father_value + w_g*cost_g + w_d*d_target + penalty + penalty_d

            
            # Branch the tree
            child = pybnb.Node()
            waypoints_x, waypoints_y = [], []
            for i in range(len(ax_)):
                waypoints_x.append(ax_[i])
                waypoints_y.append(ay_[i])
            waypoints_x.append(tmp_ax)
            waypoints_y.append(tmp_ay)
            child.state = (x, tmp_s, child_value, self._bound, choices, waypoints_x, waypoints_y, tmp_pi_bar)
            ax.pop(-1)
            ay.pop(-1)
            yield child

    def beliefPropagation(self, pi_bar,ax,ay,auvs_xy,ctrl_input,s_pose,v_n,DT):
        pi_bar_out = []
        for i in range(self.auvNum):
            j_pi_bar = pi_bar[i]
            out = []
            if np.sum(j_pi_bar) != 0 or self.auvID == i+1:
                if self.auvID == i+1:
            # Compute the path of the i-th AUV acoording to the choosen command
                    path, ax, ay, ayaw = header.utils.update_path(ax, ay, ctrl_input, s_pose[2], v_n, DT)
                    out = [ax[-1],ay[-1],ayaw]
                    
                else:
                    ax_j,ay_j = [], []
                    auvs_xy[i] = [j_pi_bar[0],j_pi_bar[1],j_pi_bar[2]]
                    j_waypoints = j_pi_bar[4] # to change if more waypoint at this stage
                    ax_j.append(j_pi_bar[0])
                    ay_j.append(j_pi_bar[1])
                    path_j, ax_j, ay_j, ayaw_j = header.utils.update_path(ax_j, ay_j, j_waypoints, j_pi_bar[2], j_pi_bar[3], DT)
                    auvs_xy[i] = [ax_j[-1],ay_j[-1],ayaw_j]
                    out = auvs_xy[i]

                if len(j_pi_bar) > 4:
                    j_pi_bar = np.delete(j_pi_bar,4)
                
                out.append(j_pi_bar[3])
                for j in range(len(j_pi_bar[4:len(j_pi_bar)])):
                    out.append(j_pi_bar[j+4])
            else:
                auvs_xy[i] = [0.0, 0.0, 0.0, 0.0]
                if len(j_pi_bar) > 4:
                    j_pi_bar = np.delete(j_pi_bar,4)
                for i in range(len(j_pi_bar)):
                    out.append(0.0)
            pi_bar_out.append(out) 

        return pi_bar_out, auvs_xy, ax, ay, ayaw

    def simulation(self, ctrl_input, x_hat, P, s_pose, sensors, ax, ay, v_n, DT, pi_bar, init_state):
        #if self.auvID == 2:
        #    rospy.logwarn('AUV ID %s AT STAGE:%s with PI_BAR_IN: %s and SENSOR POSE: %s',self.auvID,len(self.choices), pi_bar, s_pose)
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
        pi_bar_out, auvs_xy, ax, ay, ayaw = self.beliefPropagation(pi_bar, ax, ay, auvs_xy, ctrl_input, s_pose, v_n, DT)
            
        # Propagate target state estimation and update i-th AUV pose
        tmp = np.zeros((4,1))
        for j in range(4):  
            tmp[j]=target.x[j]
        target.x = target.F*tmp
        s_pose = [ax[-1],ay[-1],ayaw]
        

        # Simulate measurements TODO: (REPRODUCE THE TDMA Sampling!!, not measure everything at the end)
        for i in range(self.auvNum):
            if self.auvID == i+1:
                tmp = s_pose
                ranges = np.sqrt((target.x[1]-tmp[1])**2+(target.x[0]-tmp[0])**2)
            else:
                tmp = auvs_xy[i]
            if np.sum(tmp) == 0.0 and self.auvID != (i+1):
                tmp = init_state[i]

            '''if self.auvID == 2:
                rospy.logwarn('AGENT POSE: %s',tmp)'''
            [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.x[0],target.x[1],[tmp[0],tmp[1]],tmp[2])
            arr = [measure_,meas_pos[0],meas_pos[1]]
            meas_table.append(arr)

        # Compute the regressor according to measurements simulated
        estimator.computeState(meas_table)

        '''if self.auvID == 2:

            plt.plot(ax,ay,'xr')
            plt.plot(s_pose[0],s_pose[1],'ob')
            plt.plot(target.x[0], target.x[1],'ok')
            
            for i in range(self.auvNum):
                pi_bar_j = pi_bar_out[i]
                if self.auvID == i+1:
                    tmp = s_pose
                else:
                    tmp = auvs_xy[i]
                print('AUVS_XY',auvs_xy[i])
                print('BOOLEAN',np.sum(tmp) == 0.0)
                if np.sum(tmp) == 0.0 and self.auvID != (i+1):
                    print('prova')
                    tmp = init_state[i]
                plt.plot(tmp[0],tmp[1],'om')
                
                plt.plot()
            
            plt.grid()
            plt.axis('equal')
            plt.show()
            rospy.logwarn('AUV ID %s with PI_BAR_OUT: %s',self.auvID, pi_bar_out)'''

        return target.x, estimator.phi, estimator.y, s_pose, ax[-1], ay[-1], pi_bar_out, ranges