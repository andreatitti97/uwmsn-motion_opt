#Import basic system modules
import os, pathlib, importlib.util
import time
# Import math modules
import numpy as np
import matplotlib.pyplot as plt
# Import ROS modules and Service
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from math import atan2
# IMPORT SOLVER LIBRARY
import pybnb #THIS MAY CHANGE ACCORDING TO THE APPLICATION

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/uwmsn-sim'+'/logs'

spec = importlib.util.spec_from_file_location("module.header", header_file+'/distr_opt_h.py')
header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(header)

# Init global variables for callbacks
t_est_x, t_est_y, s_state_x, s_state_y = [], [], [], []
t_state = [None, None, None, None, None]
s_state = [None, None, None]
policies_intent = [[], [], [], []]
init_state = None
init_d = None

for i in range(len(policies_intent)):
    tmp = []
    for j in range((len(s_state)+1+header.config.H)):
        tmp.append(0.0)
    policies_intent[i] = tmp

# Global Variables
cubicSpline = header.planner

def update_path(ax, ay, waypoint, ayaw, desired_vel, DT):
        
        a_i = [ax[-1],ay[-1]]
        
        ax.append(np.cos(ayaw+waypoint)*desired_vel*DT+a_i[0])
        ay.append(np.sin(ayaw+waypoint)*desired_vel*DT+a_i[1])
        
        path = cubicSpline.CubicSpline2D(ax, ay)
        ayaw = ayaw+waypoint
        return path, ax, ay, ayaw

def compute_cost(phi):
    length_y = len(phi)
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    
    PHI = np.dot(np.transpose(tmp_phi),tmp_phi)
    
    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
   

def compute_cost2(phi):
    # Compute Covariance of the target state
    R = np.zeros((len(phi),len(phi))) #matrice diagonale perchè errori sulle singole misure indipendenti tra loro            
    for i in range(len(phi)): 
        for j in range(len(phi)):
            if i == j:
                R[i,j] = (header.config.SIGMA_MEAS)
            else:
                R[i,j] = 0 

    a = header.config.SIGMA_MEAS
    cov = np.linalg.inv(np.dot(np.dot(np.transpose(phi),np.linalg.inv(a*np.identity(len(phi)))),phi))
        
    return np.trace(cov)

class Target():
    def __init__(self, x, dt, P=[]):
        'INPUT: estimate state of the target, eventualy associate P for uscented transform'
        self.x = x
        self.dt = dt
        self.P = P
        self.F = np.matrix([[1,0,self.dt,0], # Target State Transition Matrix - CV
                        [0,1,0,self.dt],
                        [0,0,1,0],
                        [0,0,0,1]])

class Estimation():
    def __init__(self):
        self.phi = []
        self.y = []
        self.x = np.zeros((2,1))
    
    def regressorUpdate(self, y_i, ax_i, ay_i):

        self.y.append(ax_i*np.sin(y_i) - ay_i*np.cos(y_i))
        C = [np.sin(y_i), -np.cos(y_i)]
        self.phi.append(C)

    def computeState(self,table):
        for i in range(len(table)):
            input_meas = table[i]
            self.regressorUpdate(input_meas[0],input_meas[1],input_meas[2])
        
def sig(x,d_max,alpha):

    return 1/(1 + np.exp(alpha*(-x+d_max/2)))

class Simple(pybnb.Problem):
    def __init__(self, x_hat, s, ctrl_cmds, pi_bar, init_state, init_d, v_n = 0, cpf_control = []):
        
        inf = float("inf")
        
        # Belief state init
        sensors = []
        for i in range(auvNum):
            sensors.append(header.sensor.Sensor(str(i),1,0,0.000))
        self.sensors = sensors
        self.controller = cpf_control
        self.ctrl_cmds = ctrl_cmds

        rospy.loginfo('OPTIMIZATION ID %s INITIAL CTRL_CMDS: %s',auvID,self.ctrl_cmds)
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
        self.alpha = 0.08 #parma sigmoid activation funct

        # Variables for path init
        tmp = np.zeros((auvNum,3))
        for i in range(len(tmp)):
            for j in range(3):
                tmp[i,j] = init_state[j+(i*3)]

        self.init_s_state = tmp
        
        if v_n != 0:
            self.v_n = v_n
        else:
            self.v_n = header.config.AUV_VEL
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
        ax_ = []
        ay_ = []
        for i in range(len(ax)):
       
            ax_.append(ax[i])
            ay_.append(ay[i])
        for i in range(header.config.U):
            x, phi, y, tmp_s, tmp_ax, tmp_ay, tmp_pi_bar = simulation(self.ctrl_cmds[i], x_hat, self.P, s, self.sensors, ax, ay, self.v_n, self.DT, pi_bar, self.init_s_state)
            
            # Update the sequence of control decisions
            tmp = [self.ctrl_cmds[i]]
            choices = self.choices + tmp

            # If we reached the planning horizon we subtract the DELTA to stop the algorithm and choose only terminal nodes
            if len(choices) == header.config.H:
                self.value = self.value - self.initial_cost ##THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENCE
                self._bound = self.value #- cost1 
                tmp_pi_bar.append([0.0])#TODO: HEURISTIC FUNCTION 
            father_value = self.value #THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENC
            penalty = 0
            penalty_d = 0
            d_max = header.config.d*2
            d_min = header.config.d/2
            delta_fd = np.pi/4
            tmp = []

            for i in range(len(tmp_pi_bar)):
                j_pi_bar = tmp_pi_bar[i]

                if len(j_pi_bar) >= 5:
                    tmp_x = np.cos(j_pi_bar[2]+j_pi_bar[4])*j_pi_bar[3]*self.DT+j_pi_bar[0]
                    tmp_y = np.cos(j_pi_bar[2]+j_pi_bar[4])*j_pi_bar[3]*self.DT+j_pi_bar[1]

                    tmp = np.sqrt((tmp_y-tmp_s[1])**2+(tmp_x-tmp_s[0])**2)
                    if tmp > d_max:
                        penalty_d = self.initial_cost/header.config.H
                    if tmp < d_min:
                        penalty_d = self.initial_cost/header.config.H
            # Add the cost of the node to the sequence
            if len(choices) < header.config.H:
                for i in range(len(choices)-1):
                    tmp_ = (choices[i]) - (choices[i+1])
                    
                    if np.abs(tmp_) > delta_fd:
                        penalty = self.initial_cost/header.config.H
            # Magnitude of the cost --> DECINE
            cost_g = compute_cost(phi)
            d_target = (np.sqrt((x[0]-s[0])**2+(x[1]-s[1])**2))
            
            w_d = sig(d_target,self.init_d,self.alpha)
            w_g = 2.0*sig(d_target,self.init_d,-self.alpha)
            # weights should be between 0 an 1 and change according to the distance.
            # other option is to make the reweighted estimation w.r.t. the range.
            # in theory you weight differently the measures in order to give more importance to near measures0

            #rospy.logwarn('OPTIMIZATION id %s COST1 %s COST2 %s',auvID,cost,cost2)
            child_value = father_value + w_g*cost_g + w_d*d_target + penalty +penalty_d
            
            # Branch the tree
            child = pybnb.Node()
            waypoints_x = []
            waypoints_y = []
            for i in range(len(ax_)):
                waypoints_x.append(ax_[i])
                waypoints_y.append(ay_[i])
            waypoints_x.append(tmp_ax)
            waypoints_y.append(tmp_ay)
            child.state = (x, tmp_s, child_value, self._bound, choices, waypoints_x, waypoints_y, tmp_pi_bar)
            ax.pop(-1)
            ay.pop(-1)
            yield child

def simulation(ctrl_input, x_hat, P, s_pose, sensors, ax, ay, v_n, DT, pi_bar, init_state):
    
    # Temporal Variable
    j = 0, 0 #time and counter init
    meas_table = []

    # Init classes for tracker and target
    target = Target(x_hat, DT, P)
    estimator = Estimation()

    # Initialize data structures for agents state
    auvs_xy = []

    for i in range(auvNum):
        auvs_xy.append([])
        for j in range((len(s_pose)+1+header.config.H)):
            auvs_xy[i].append(0.0)
   
    pi_bar_out = []
    # Compute the path of the j-th neighbours
    j_waypoints = []
    
    for i in range(auvNum):
        
        j_pi_bar = pi_bar[i]
        out = []
        if np.sum(j_pi_bar) != 0 or auvID == i+1:
            if auvID == i+1:
          # Compute the path of the i-th AUV acoording to the choosen command
                path, ax, ay, ayaw = update_path(ax, ay, ctrl_input, s_pose[2], v_n, DT)
                s_pose = [ax[-1],ay[-1],ayaw]
                out = s_pose
                
            else:
                ax_j,ay_j = [], []
                auvs_xy[i] = [j_pi_bar[0],j_pi_bar[1],j_pi_bar[2]]
                j_waypoints = j_pi_bar[4] # to change if more waypoint at this stage
                ax_j.append(j_pi_bar[0])
                ay_j.append(j_pi_bar[1])
                path_j, ax_j, ay_j, ayaw_j = update_path(ax_j, ay_j, j_waypoints, j_pi_bar[2], j_pi_bar[3], DT)
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

    # Propagate target state estimation
    tmp = np.zeros((4,1))
    old_t = target.x
    for j in range(4):  
        tmp[j]=target.x[j]
    target.x = target.F*tmp
    
    # Simulate measurements TODO: (REPRODUCE THE TDMA Sampling!!, not measure everything at the end)
    #auvs_xy[auvID-1] = s_pose #TODO CHECK WHY FUNDAMENTAL
    for i in range(auvNum):
        if auvID == i+1:
            tmp = s_pose
        else:
            tmp = auvs_xy[i]
        if np.sum(tmp) != 0.0 or auvID == (i+1):
            
            [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.x[0],target.x[1],[tmp[0],tmp[1]],tmp[2])
            arr = [measure_,meas_pos[0],meas_pos[1]]
            meas_table.append(arr)
            #rospy.logwarn('OPTIMIZATION ID %s has CURRENT AUV STATE: %s ---CASE 1 measure %s',auvID,tmp,measure_)
        else:
            tmp = init_state[i]
            [measure_, rel_bearing_, meas_pos] = sensors[i].measureBearing(target.x[0],target.x[1],[tmp[0],tmp[1]],tmp[2])
            arr = [measure_,meas_pos[0],meas_pos[1]]
            meas_table.append(arr)
            #rospy.logwarn('OPTIMIZATION ID %s has CURRENT AUV STATE: %s ---CASE 2 measure %s',auvID,tmp,measure_)
    estimator.computeState(meas_table)
        
    return target.x, estimator.phi, estimator.y, s_pose, ax[-1], ay[-1], pi_bar_out

def shutdown_cllbk():
    global auvID, avg_time, avg_nodes
    np.savetxt(log_path+'/wall_times'+str(auvID)+'.txt',avg_time)
    np.savetxt(log_path+'/nodes'+str(auvID)+'.txt',avg_nodes)
    
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- OPTIMIZATION '+str(auvID)+': Simulation data saved --> Shutting down ...%s',magenta,none)

def callbackTstate(data):
    global t_state
    t_state = data.data
    

def callbackSstate(data):
    global s_state
    tmp = data.data
    s_state = tmp
    
def callback1(data):
    global policies_intent
    tmp = data.data
    policies_intent[0] = tmp

def callback2(data):
    global policies_intent
    tmp = data.data
    policies_intent[1] = tmp

def callback3(data):
    global policies_intent
    tmp = data.data
    
    policies_intent[2] = tmp

def callback4(data):
    global policies_intent
    tmp = data.data
    policies_intent[3] = tmp

def callbackInit1(data):
    global init_state
    init_state = data.data

def callbackInit2(data):
    global init_d
    tmp = data.data
    init_d = tmp[0]

def listener(auvID,auvNum):
   
    rospy.Subscriber('/'+str(auvID)+'/estimation', numpy_msg(Floats), callbackTstate)
    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callbackSstate)
    rospy.Subscriber('/init_opt1', numpy_msg(Floats), callbackInit1)
    rospy.Subscriber('/init_opt2', numpy_msg(Floats), callbackInit2)
    callback_list = [callback1, callback2, callback3, callback4]
    for i in range(auvNum):
        
        if i+1 != auvID:
            rospy.Subscriber('/'+str(i+1)+'/rx_ctrl_policy', numpy_msg(Floats), callback_list[i])

def main():
    
    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace+'auv'

    # Get AUV ID and number of vehicles.
    global auvID, auvNum, avg_nodes, avg_time, policies_intent, s_state, t_state, init_state, init_d
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')

    # Node Init
    rospy.init_node('auv'+str(auvID)) #TO ADD debug prints --> log_level=rospy.DEBUG

    # ROS simulation parameters
    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)
    listener(auvID,auvNum)
    # Init time variables and counters and lists
    t, count1, count_low, count_max = 0,0, 0,0
    
    avg_time, avg_nodes, old_ctrls = [],[],[]
    old_t_state = [None, None, None, None]

    # Load simulation parames from config file
    t_scaler = header.config.TIME_SCALER
    dt = header.config.TIME_STEP*t_scaler
    u_max = header.config.u_max
    delta_u = header.config.delta_u
    
    ctrl_choices = header.config.ctrl_cmd
    limit = 0.0# Compute nodes limit according to RHC with finite memory
    for i in range(header.config.H+1):
            limit += header.config.U**i

    # Init publishers and subscribers
    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/ctrl_policy',numpy_msg(Floats),queue_size=100)

    rospy.sleep(1)
    while not rospy.is_shutdown():

        if t_state[0] != old_t_state[0] and t_state[0] != None:
            
            rospy.loginfo('OPTIMIZATION ID %s STARTING with POLICIES of INTENT: %s and V_n: %s',auvID,policies_intent,t_state[4])
            problem = Simple(t_state[0:4], s_state, ctrl_choices, policies_intent, init_state, init_d, t_state[4])
            solver = pybnb.Solver()
            results = solver.solve(problem,queue_strategy="bound" ,node_limit=limit)#tnode_limit=limi #Uniform cost search con "objective"
            best_node_states = results.best_node.state #objective_stop=90000,time_limit=5 - other queue strategies
            wall_time = results.wall_time
            nodes = results.nodes
            avg_nodes.append(nodes)
            avg_time.append(wall_time)
            output_policy = best_node_states[4]
            msg = [s_state[0],s_state[1],s_state[2],t_state[4]]

            for i in range(header.config.H+1):
                if i < header.config.H:
                    msg.append(output_policy[i])#appendi la sequenza ottimale di controllo
                else:
                    msg.append(0.0)
            pub_ctrl_policy.publish(np.array(msg,dtype=np.float32))
            rospy.loginfo('OPTIMIZATION ID %s DONE! --> Output Policy: %s',auvID,msg)

            
            old_ctrls.append(output_policy[0])
            # Update the ctrl_choices
            
            # Adapt online the heading changes: # TO DEBUG !!!!! OR TO TUNE PROPERLY -  in theory done to check
            if len(old_ctrls) == 3:
                for i in range(len(old_ctrls)):
                    if [0-(1e-3)] <=  np.abs(old_ctrls[i])-(1e-3) <= header.config.u_max *2/(header.config.U):
                        count_low += 1 
                        if count_low == 3:
                            if u_max <= header.config.MIN:
                                u_max = u_max
                                count_low = 0
                            else:
                                print('DECREASING K_MAX----------------------------')
                                u_max  = u_max  - delta_u
                                count_low = 0
                    elif np.abs(old_ctrls[i]) >= u_max :
                        count_max += 1
                        if count_max == 3:
                            if u_max >= header.config.MAX:
                                u_max = u_max
                                count_max = 0
                            else:
                                print('INCREASING K_MAX++++++++++++++++++++++++++++')
                                u_max  = u_max  + delta_u
                                count_max = 0
                count_low = 0
                count_max = 0
                old_ctrls = []

                #ctrl_choices = [-u_max, -u_max*4/(header.config.U),-u_max*2/(header.config.U),0,
                #                u_max*2/(header.config.U),u_max*4/(header.config.U),u_max]

                print('COUNT LOW-------------------------------',count_low)
                print('COUNT MAX+++++++++++++++++++++++++++++++',count_max)


        if int(t) == (header.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')
        old_t_state = t_state
        t += dt
        count1 += 1
        rate.sleep()

    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()
if __name__ == '__main__':
    
    main()

'''if auvID == 1:

        rospy.logwarn('OPTIMIZATION ID %s SENSOR POSE; %s',auvID,s_pose)
        #rospy.logwarn('OPTIMIZATION ID %s [rx, ry, ryaw]; %s',auvID,[rx, ry, ryaw])
        rospy.logwarn('AUV ID %s with AUVS_XY: %s',auvID,auvs_xy)
        rospy.logwarn('AUV ID %s with MEAS_TABLE: %s',auvID, meas_table)
        rospy.logwarn('AUV ID %s with PI_BAR_OUT: %s',auvID, pi_bar_out)
        rospy.logwarn('AUV ID %s with INIT_STATE: %s',auvID, init_state)
        rospy.logwarn('OPTIMIZATION ID %s WAYPOINTS POST updating path -> ax:(%s) ay:(%s)',auvID,ax,ay)
    
        plt.plot(s_pose[0],s_pose[1],'oy')
        plt.plot(old_t[0],old_t[1],'ok')
        plt.plot(target.x[0],target.x[1],'or')
        plt.plot(ax,ay,'xr')

        for i in range(len(auvs_xy)):
            tmp = auvs_xy[i]
            
            if np.sum(tmp) != 0.0 and auvID != i+1:
                plt.plot(tmp[0],tmp[1],'om')
            else:
                tmp = init_state[i]
            
                plt.plot(tmp[0],tmp[1],'om')
                
            tmp = init_state[i]
            plt.plot(init_state[i,0],init_state[i,1],'og')
        plt.grid()
        plt.axis('equal')
        plt.show()'''