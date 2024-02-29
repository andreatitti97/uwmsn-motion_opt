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

# Global Variables
DELTA = 10**15
cubicSpline = header.planner
DT = header.config.OPTIMIZATION_TIME_STEP #should be equal more or less to the expected time to perform an estimation
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

def simulation(control_input, target_est, s_pose, sensors, controller, ax, ay, d,v_n,P):
    
    # Temporal Variable
    j = 0, 0 #time and counter init
    meas_table = []

    dt = header.config.OPTIMIZATION_TIME_STEP
    # Init classes for tracker and target
    target = Target(target_est, dt, P)
    estimator = Estimation()

     # Load Path
    path = cubicSpline.CubicSpline2D(ax, ay)#re-generate the path followed up to now
    [rx, ry, ryaw, rk, s]=header.utils.calc_spline_course(path,dt)

    # Load Path

    path, d, ax, ay, current_theta = update_path(ax,ay,control_input,s_pose, v_n)

    s_pose = [rx[-1],ry[-1],ryaw[-1]]

    # Compute FINAL agents pose
    geometry = header.config.geometry
    f = header.config.formation
    auvs_xy = np.zeros((header.config.N_AUV,2))
    auvs_theta = np.zeros(header.config.N_AUV)

    for i in range(header.config.N_AUV):

        if geometry == 'line' or geometry == 'line2':
            auvs_theta[i] = s_pose[2]
            auvs_xy[i,0] = s_pose[0] + (f[i,0]*np.cos(s_pose[2])+f[i,1]*np.sin(s_pose[2]))
            auvs_xy[i,1] = s_pose[1] - (-f[i,0]*np.sin(s_pose[2])+f[i,1]*np.cos(s_pose[2]))      
        if geometry == 'column' or geometry == 'column2':      
            
            x,y = path.calc_position(-f[i]+d)
            auvs_xy[i,0] = x
            auvs_xy[i,1] = y
            auvs_theta[i] = path.calc_yaw(-f[i]+d)

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

    return target.x, estimator.phi, estimator.y, s_pose, ax[-1], ay[-1], d

def compute_cost(phi,length_y):
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    PHI = np.dot(np.transpose(tmp_phi),tmp_phi)
    cost2 = np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
    return cost2

def main():

    # Ros Initialization
    rospy.init_node('optimization')
    pub = rospy.Publisher("ctrl_cmd",numpy_msg(Floats),queue_size=100)
    Hz = 1/(header.config.TIME_STEP)
    rate = rospy.Rate(Hz)

    # Init lists for log files.
    ctrl_opt, ctrl_plot, sensors, old_ctrls = [], [], [], []  
    avg_time, avg_nodes = [],[]

    # Init Parameters for U setup
    count_low, count_max = 0,0
    ctrl_cmd = header.config.ctrl_cmd


    u_max = header.config.u_max
    delta_u = header.config.delta_u
    limit = 0.0
    for i in range(header.config.M+1):
        limit += header.config.U**i

    # Initialize sensors class (Reproduce the AVS)
    geometry = header.config.geometry
    for i in range(header.config.N_AUV): 
        sensors.append(header.sensors.Sensor(str(i),1,0,0.000))#header.config.SIGMA_MEAS
    
    if header.config.OPTIMIZATION_ON == True:
        rospy.loginfo('STARTED OPTIMIZATION')
    
    while not rospy.is_shutdown():

        # Retrieve information from estimation module (propagate estimation) and planning module (update the path)
        t_est = rospy.wait_for_message('/estimation',numpy_msg(Floats))
        s_state = rospy.wait_for_message('/platform_state',numpy_msg(Floats))
        ax = rospy.wait_for_message('/ax',numpy_msg(Floats))
        ay = rospy.wait_for_message('/ay',numpy_msg(Floats))
        cov = rospy.wait_for_message('/cov',numpy_msg(Floats))
        t_est = t_est.data
        s_ = s_state.data
        
        # Data conversion
        s_state = [s_[0], s_[1], s_[2]]
        v_n = s_[3]
        ax_array = ax.data
        ay_array = ay.data
        cov = cov.data
        
        # Load the last section of followed path
        ax, ay = [], []
        if geometry == 'column' or geometry == 'column2': # THIS CAN BECAME A FUNCTION
            for i in range(len(ay_array)):
                #length = len(ay_array)-n
                ax.append(ax_array[i])#ax.append(ax_array[length+])
                ay.append(ay_array[i])
            path = cubicSpline.CubicSpline2D(ax, ay)
            d = path.s[-1]-1

        elif geometry == 'line' or geometry == 'line2':
            for i in range(len(ay_array)):
                
                ax.append(ax_array[i])
                ay.append(ay_array[i])
            path = cubicSpline.CubicSpline2D(ax, ay)
            d = path.s[-1]-1

        # Initialize Cooperative Path Following Class 
        cpf_control = 0
        n = len(t_est)
        P = np.zeros((n,n))
        for i in range(n):
            P[i,:] = cov[(i*n):(i*n)+n]

        ######## Compute the best solution solving the optimization with BnB or Greedy search #####
  
        problem = Simple(t_est, s_state, sensors, cpf_control, ctrl_cmd, ax, ay, d, P, v_n)
        solver = pybnb.Solver()
        ''' TEST ON BnB problem_simplified = Simple(t_est, s_state, DELTA, sensors, cpf_control, ctrl_cmd, P)
        results_preview = solver.solve(problem,queue_strategy="objective",node_limit=limit)
        lower_bound = results_preview.objective'''
        results = solver.solve(problem,queue_strategy="bound" ,node_limit=limit)#tnode_limit=limi #Uniform cost search con "objective"
        best_node_states = results.best_node.state #objective_stop=90000,time_limit=5
        wall_time = results.wall_time
        nodes = results.nodes
        avg_nodes.append(nodes)
        avg_time.append(wall_time)
        ctrl_opt = best_node_states[4]

        ###########################################################################################
        time.sleep(10/Hz)
        pub.publish(np.array(ctrl_opt,dtype=np.float32))
        ctrl_plot.append(ctrl_opt[0])
        old_ctrls.append(ctrl_opt[0])
        
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
            if header.config.U == 5: 
                ctrl_cmd = [-u_max, -u_max*4/(header.config.U),0,u_max*4/(header.config.U),u_max]
            elif header.config.U == 7:
                ctrl_cmd = [-u_max, -u_max*4/(header.config.U),-u_max*2/(header.config.U),0,u_max*2/(header.config.U),u_max*4/(header.config.U),u_max]
            else:
                ctrl_cmd = [-u_max,0,u_max]
            print('COUNT LOW-------------------------------',count_low)
            print('COUNT MAX+++++++++++++++++++++++++++++++',count_max)

        #SAVE DATA FOR PLOT
        np.savetxt(log_path+'/wall_times.txt',avg_time)
        np.savetxt(log_path+'/nodes.txt',avg_nodes)

        ctrl_opt = []
        rate.sleep()
    
if __name__ == '__main__':
    
    main()



'''np.savetxt(log_path+'/plot_cmds.txt',ctrl_plot)
np.savetxt(log_path+'/t_est_x_opt.txt',t_est_x[0])
np.savetxt(log_path+'/t_est_y_opt.txt',t_est_y[0])
np.savetxt(log_path+'/s_state_x.txt',s_state_x)
np.savetxt(log_path+'/s_state_y.txt',s_state_y)'''