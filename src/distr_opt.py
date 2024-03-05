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
t_state = [None, None, None, None]
s_state = [None, None, None]
policies_intent = [None, None, None, None]

# Global Variables
DELTA = 10**15
cubicSpline = header.planner
DT = 15#auvNum*header.config.Td #should be equal more or less to the expected time to perform an estimation
desired_vel = header.config.AUV_VEL

def update_path(ax, ay, waypoint, s_pose, desired_vel):
        
        t_i = s_pose[2]
    
        path = cubicSpline.CubicSpline2D(ax, ay)
       
        # Compute the distance travelled according to the new path
        d_real = path.s[-1]
        a_i = [ax[-1],ay[-1]]

        t_f = t_i+waypoint
        tmp_x = np.cos(t_f)*desired_vel*DT+a_i[0]
        tmp_y = np.sin(t_f)*desired_vel*DT+a_i[1]
        ax.append(tmp_x)
        ay.append(tmp_y)
        t_i = t_f
        a_i = [tmp_x,tmp_y]
        path = cubicSpline.CubicSpline2D(ax, ay)

        return path, d_real, ax, ay, t_f

class Target():
    def __init__(self, init_state, dt, P=[]):
        'INPUT: estimate state of the target, eventualy associate P for uscented transform'
        self.x = init_state
        self.dt = dt
        self.cov = P
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
        # Re-arrange python list into numpy array for matrix moltiplication
        tmp_y = np.zeros((len(self.y),1))
        for i in range(len(self.y)):
            tmp_y[i] = self.y[i]
        # Compute State (OPTIONAL, in this case we need only the conditioning to assigne the cost)
        self.x = np.dot(np.linalg.pinv(self.phi),tmp_y)

class Simple(pybnb.Problem):
    def __init__(self,x_hat, s,sensors, cpf_control, ctrl_cmds, ax, ay, d,cov,v_n):
        
        inf = float("inf")
        self.value = DELTA  
        self.initial_cost = DELTA
        self._bound = -inf # initial_cost-100 #lower bound 
        self.choices = []
        self._x_hat = x_hat
        self._s = s
        self.sensors = sensors
        self.controller = cpf_control
        self.ctrl_cmds = ctrl_cmds
        self.cov = cov
        # Waypoints performed until optimization
        self.ax = ax 
        self.ay = ay
        self.d = d
        self.v_n = v_n

    # required methods
    def sense(self):
        return pybnb.minimize

    def objective(self):
        return self.value

    def bound(self):
        return self._bound

    def save_state(self, node):
        
        node.state = (self._x_hat, self._s, self.value, self._bound, self.choices, self.ax, self.ay,self.d)

    def load_state(self, node):

        (self._x_hat, self._s, self.value, self._bound, self.choices, self.ax, self.ay,self.d) = node.state

    def branch(self): #durante il branch devi calcolare le varie realizzazioni quindi simuli qua

        x_hat, s, cov, ax, ay, d = self._x_hat, self._s, self.cov, self.ax, self.ay, self.d
        ax_ = []
        ay_ = []
        for i in range(len(ax)):
            ax_.append(ax[i])
            ay_.append(ay[i])
        for i in range(header.config.U):
            x, phi, y, s, tmp_ax, tmp_ay, tmp_d = simulation(self.ctrl_cmds[i], x_hat, s, self.sensors, self.controller, ax, ay, d, self.v_n ,cov)
            # Update the sequence of control decisions
            tmp = [self.ctrl_cmds[i]]
            choices = self.choices + tmp
            # If we reached the planning horizon we subtract the DELTA to stop the algorithm and choose only terminal nodes
            if len(choices) == header.config.M:
                self.value = self.value - self.initial_cost ##THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENCE
                self._bound = self.value #- cost1 
            father_value = self.value #THIS IS MANDATORY FOR ADDITIVE COST ALONG THE SEQUENC
            # Add the cost of the node to the sequence
            cost = compute_cost(phi,len(y))
            child_value = father_value + cost
            # Branch the tree
            child = pybnb.Node()
            waypoints_x = []
            waypoints_y = []
            for i in range(len(ax_)):
                waypoints_x.append(ax_[i])
                waypoints_y.append(ay_[i])
            waypoints_x.append(tmp_ax)
            waypoints_y.append(tmp_ay)
            child.state = (x, s, child_value, self._bound, choices, waypoints_x, waypoints_y, tmp_d)
            ax.pop(-1)
            ay.pop(-1)
            yield child

def simulation(control_input, s_pose, target_est, P, sensors, d, v_n):
    
    # Temporal Variable
    j = 0, 0 #time and counter init
    meas_table = []

    dt = DT
    # Init classes for tracker and target
    target = Target(target_est, dt, P)
    estimator = Estimation()

    # Compute FINAL agents pose
    auvs_xy = np.zeros((header.config.N_AUV,2))
    auvs_theta = np.zeros(header.config.N_AUV)

    # Propagate target state estimation
    tmp = np.zeros((4,1))
    for j in range(4):  
        tmp[j]=target.x[j]
    target.x = target.F*tmp

    for j in range(header.config.N_AUV):
        [measure_, rel_bearing_, meas_pos] = sensors[j].measureBearing(target.x[0],target.x[1],auvs_xy[j],auvs_theta[j])
        arr = [measure_,meas_pos[0],meas_pos[1]]
        meas_table.append(arr)
    estimator.computeState(meas_table)

    return target.x, estimator.phi, estimator.y, s_pose, d

def compute_cost(phi,length_y):
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    PHI = np.dot(np.transpose(tmp_phi),tmp_phi)
    cost2 = np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
    return cost2

def shutdown_cllbk():
    global auvID, avg_time, avg_nodes
    np.savetxt(log_path+'/wall_times'+str(auvID)+'.txt',avg_time)
    np.savetxt(log_path+'/nodes'+str(auvID)+'.txt',avg_nodes)
    
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- OPTIMIZATION '+str(auvID)+': Simulation data saved --> Shutting down ...%s',magenta,none)

def callbackTstate(data):
    global t_state
    tmp = data.data
    t_state = tmp

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

def listener(auvID,auvNum):
   
    rospy.Subscriber('/'+str(auvID)+'/estimation', numpy_msg(Floats), callbackTstate)
    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callbackSstate)
    callback_list = [callback1, callback2, callback3, callback4]
    for i in range(auvNum):
        if i+1 != auvID:
            rospy.Subscriber('/'+str(i+1)+'/rx_ctrl_policy', numpy_msg(Floats), callback_list[i])

def main():
    global policies_intent, s_state, t_state
    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace+'auv'

    # Get AUV ID and number of vehicles.
    global auvID, auvNum, avg_nodes, avg_time
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')

    # Node Init
    rospy.init_node('auv'+str(auvID)) #TO ADD debug prints --> log_level=rospy.DEBUG

    # ROS simulation parameters
    t_scaler = header.config.TIME_SCALER

    Hz = 1/(header.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)

    # Init time variables and counters and lists
    t, count1 = 0,0
    avg_time, avg_nodes = [],[]
    dt = header.config.TIME_STEP*t_scaler
    old_t_state = [None, None, None, None]
    # Init publishers and subscribers and sensor objects
    sensors = []
    for i in range(auvNum):
        sensors.append(header.sensor.Sensor(str(i),1,0,0.000))

    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/ctrl_policy',numpy_msg(Floats),queue_size=100)
    # Compute nodes limit according to RHC with finite memory
    limit = 0.0
    for i in range(header.config.H+1):
            limit += header.config.U**i

    listener(auvID,auvNum)
    rospy.sleep(1)

    
    while not rospy.is_shutdown():


        if t_state[0] != old_t_state[0] and t_state[0] != None:
            
            for i in range(len(policies_intent)):
                tmp = policies_intent[i]
                #if tmp[0] != None:
                #rospy.loginfo('OPTIMIZATION of AUV ID %s - Policy of intent of AUV %s: %s', auvID, i+1, policies_intent[i])
             
            ######## DO OPTIMIZATION HERE ! ########
            '''problem = Simple(t_state, s_state, policies_intent,sensors)
            solver = pybnb.Solver()
            results = solver.solve(problem,queue_strategy="bound" ,node_limit=limit)#tnode_limit=limi #Uniform cost search con "objective"
            best_node_states = results.best_node.state #objective_stop=90000,time_limit=5
            wall_time = results.wall_time
            nodes = results.nodes
            avg_nodes.append(nodes)
            avg_time.append(wall_time)
            output_policy = best_node_states[4]'''
            msg = [s_state[0],s_state[1],s_state[2]]

            for i in range(header.config.H):
                msg.append(0)#appendi la sequenza ottimale di controllo
            
            output_policy = np.array(msg,dtype=np.float32)
            
            pub_ctrl_policy.publish(output_policy)

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


''' TEST ON BnB problem_simplified = Simple(t_est, s_state, DELTA, sensors, cpf_control, ctrl_cmd, P)
        results_preview = solver.solve(problem,queue_strategy="objective",node_limit=limit)
        lower_bound = results_preview.objective'''