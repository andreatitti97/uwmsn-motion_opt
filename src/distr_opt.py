#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
import matplotlib.pyplot as plt
# Import ROS modules and Service
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from math import atan2

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
directory = pathlib.Path(__file__).parent.resolve()

spec = importlib.util.spec_from_file_location("module.bnb", directory/'bnb.py')
bnb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnb)

spec = importlib.util.spec_from_file_location("module.utils", directory/'Classes/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

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

def shutdown_cllbk():
    global auvID, avg_time, avg_nodes
    np.savetxt(log_path+'/wall_times'+str(auvID)+'.txt',avg_time)
    np.savetxt(log_path+'/nodes'+str(auvID)+'.txt',avg_nodes)
    
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- OPTIMIZATION '+str(auvID)+': Simulation data saved --> Shutting down ...%s',magenta,none)

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
            problem = bnb.Simple(auvNum, auvID, t_state[0:4], s_state, ctrl_choices, policies_intent, init_state, init_d, t_state[4])
            solver = bnb.pybnb.Solver()
            results = solver.solve(problem,queue_strategy="bound" ,node_limit=limit)# node_limit=limi #Uniform cost search con "objective"
                                                                                    # objective_stop=90000,time_limit=5 - other queue strategies
            best_node_states, wall_time, nodes = results.best_node.state, results.wall_time, results.nodes
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