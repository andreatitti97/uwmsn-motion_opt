#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
from math import atan2
# Import ROS modules and Service
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# Load the h file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/logs'

spec = importlib.util.spec_from_file_location("module.h", header_file+'/distr_opt_h.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

# Init global variables for callbacks
AUV_XY = h.config.AUV_XY
t_est_x, t_est_y, s_state_x, policies_intent, s_state_y = [], [], [], [], []
t_state = [None, None, None, None, None]
s_state = [None, None, None]
for i in range(len(AUV_XY)):
    policies_intent.append([])
init_state, init_d = None, None

def frbdDcsMtd(output_policy):
    # Forbidden Decision Method
    if -0.1 <= np.sum(output_policy) <= +0.1: #check if zig-zag trajectorys
        waypoints = []
        for i in range(len(output_policy)):
            waypoints.append(0)
    else:
        waypoints = output_policy
    return waypoints

def adptCtrlSet(old_ctrls,ctrl_set,count_low,count_max,u_max, delta_u):

    U, loops = len(ctrl_set), len(old_ctrls)
    for i in range(loops):
        
        idx = ctrl_set.index(min(ctrl_set))
        if [0-(1e-3)] <=  np.abs(old_ctrls[i]) <= ctrl_set[idx+1]+(1e-3):
            count_low += 1 
            if count_low == 3:
                if u_max <= h.config.MIN:
                    u_max = u_max
                    count_low = 0
                else:
                    u_max  = u_max  - delta_u
                    count_low = 0
                    rospy.logwarn('|---- Decreasing u_max (deg) --> %s',u_max)
                    
        elif np.abs(old_ctrls[i]) >= u_max-(1e-3):
            count_max += 1
            if count_max == 3:
                if u_max >= h.config.MAX:
                    u_max = u_max
                    count_max = 0
                else:
                    u_max  = u_max  + delta_u
                    count_max = 0
                    rospy.logwarn('|---- Increasing u_max (deg) --> %s',u_max)
    ctrl_set = []
    u_i = u_max/((U-1)/2)
    for i in range(U):
        if i <= U/2:
            ctrl_set.append(u_max-i*u_i)
        if i > U/2:
            ctrl_set.append((i-((U-1)/2))*u_i)
    
    return ctrl_set, count_max, count_low, u_max

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
    init_d = data.data

def shutdown_cllbk():
    global auvID, avg_time, avg_nodes
    np.savetxt(log_path+'/wall_times'+str(auvID)+'.txt',avg_time)
    np.savetxt(log_path+'/nodes'+str(auvID)+'.txt',avg_nodes)
    
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- OPTIMIZATION '+str(auvID)+': Simulation data saved --> Shutting down ...%s',
                magenta,none)

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
    cyan = "\033[0;36m"
    none = "\033[0m"
    # ROS simulation parameters
    Hz = 1/(h.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)
    listener(auvID,auvNum)

    # Init time variables and counters and lists
    t, count1, count_low, count_max = 0,0,0,0
    sim_time = h.config.TIME_DURATION
    avg_time, avg_nodes, old_ctrls = [],[],[]

    pareto_data_c = []
    pareto_data_g = []
    pareto_data_d = []

    old_t_state = [None, None, None, None]

    # Load simulation parames from config file
    t_scaler = h.config.TIME_SCALER
    dt = h.config.TIME_STEP*t_scaler
    u_max = h.config.u_max
    delta_u = h.config.delta_u
    DT = h.config.DT
    
    ctrl_set = h.config.ctrl_cmd
    limit = 0.0# Compute nodes limit according to RHC with finite memory
    for i in range(h.config.H+1):
        limit += h.config.U**i

    # Init publishers and subscribers
    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/ctrl_policy',numpy_msg(Floats),queue_size=100)

    AUV_failure = h.config.AUV_failure

    rospy.sleep(1)
    while not rospy.is_shutdown():

        if t > sim_time/2:# for changing noise level during the sim.
            NL = h.config.NL
        else:
            NL = h.config.NL

        if t > sim_time/2:#to simulate AUV failure during the sim.
            AUV_failure = True

        if t_state[0] != old_t_state[0] and t_state[0] != None:
            pi_bar_in, s, x_hat = [], [], []

            for i in range(len(policies_intent)):
                pi_bar_in.append(policies_intent[i])
            for i in range(len(t_state)):
                x_hat.append(t_state[i])
            for i in range(len(s_state)):
                s.append(s_state[i])

            if t_state[4] != -10**3:# check that the target state has been updated.

                if t < DT:
                    pi_bar_in = []
                    for i in range(auvNum):
                        pi_i = []
                        for j in range(len(s_state)):
                            pi_i.append(AUV_XY[i+1,j])
                        for k in range((((h.config.H+1)*2))):
                            pi_i.append(0.0)
                        pi_bar_in.append(pi_i)
                        

                # Initialize the problem
                rospy.loginfo('%s OPTIMIZATION ID %s STARTING! --> Policy of intent: %s %s',cyan,auvID,pi_bar_in,none)
                
                problem = h.bnb.Simple(auvNum, auvID, x_hat[0:4], s_state, ctrl_set,
                                            pi_bar_in, init_state, init_d[auvID-1],
                                            NL, AUV_failure, x_hat[4])                                        
                # Solve the optimization problem
                solver = h.bnb.pybnb.Solver()
                # Store the results
                res = solver.solve(problem,queue_strategy="breadth",
                                       node_limit=limit,relative_gap=0.001)
                        
                bns, wall_time, nodes = res.best_node.state, res.wall_time, res.nodes

                avg_nodes.append(nodes), avg_time.append(wall_time)

                output_policy, ref_vels, list_c, list_g, list_d  = bns[4],bns[6], bns[7], bns[8], bns[9]
   
                if auvID == 2:#DEBUG
                    pareto_data_c.append(list_c[0])
                    pareto_data_g.append(list_g[0])
                    pareto_data_d.append(list_d[0])

                    np.savetxt(log_path+'/list_c',pareto_data_c)
                    np.savetxt(log_path+'/list_d',pareto_data_g)
                    np.savetxt(log_path+'/list_d',pareto_data_d)
                #Formatting the results according to the communication protocol
                msg = [s_state[0],s_state[1],s_state[2]]
                for i in range(h.config.H+1):
                    msg.append(ref_vels[i]) # append the vels for complete policy of intent
                # Apply the Forbidden Decision Method
                waypoints = frbdDcsMtd(output_policy)
                for i in range(h.config.H):
                    msg.append(waypoints[i])               
                msg.append(0.0) #HEURISTIC FUNCTION to complete the POLICY OF INTENT

                # Ros pub
                pub_ctrl_policy.publish(np.array(msg,dtype=np.float32))
                rospy.loginfo('%s OPTIMIZATION ID %s DONE! --> Output Policy: %s %s',cyan,auvID,msg,none)
                
                # Adapt Online ctrl set
                old_ctrls.append(output_policy[0])
                # Adapt online the heading changes
                if len(old_ctrls) == 3:
                    ctrl_set, count_max, count_low, u_max = adptCtrlSet(old_ctrls,ctrl_set,
                                                                        count_low,count_max,u_max, delta_u)
                    old_ctrls = []

            elif t_state[4] == -10**3: #in case of a static target only
                msg = [s_state[0],s_state[1],s_state[2],0.0]

                for i in range(h.config.H*2+1):
                    msg.append(0.0)
                pub_ctrl_policy.publish(np.array(msg,dtype=np.float32))

        if int(t) == (h.config.TIME_DURATION-1):
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
