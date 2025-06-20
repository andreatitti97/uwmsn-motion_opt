#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
from math import atan2
# Import ROS modules and Service
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from uwmsn_msgs.msg import Matrix
# Module std:out
from contextlib import redirect_stdout
import io
import time
from pybnb.solver import Solver

# Load the h file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/logs'

spec = importlib.util.spec_from_file_location("module.header", header_file+'/distr_opt_h_MTT.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

# Init global variables for callbacks
AUV_XY = h.config.AUV_XY
t_est_x, t_est_y, s_state_x, plcyInt, s_state_y = [], [], [], [], []

targetsState = [[] for _ in range(h.config.targetNum)] # _ python convention for unused vars
senState = [None, None, None]
plcyInt = [[] for _ in range(len(AUV_XY))]
tsMsg, acquiredTargets = 0.0, 0

def callbackTstate(data):
    global targetsState, tsMsg, acquiredTargets

    # Reshape incoming data
    rxMsg = np.array(data.data).reshape(data.rows, data.cols)
    acquiredTargets = len(rxMsg)

    # Initialize targetsState and timestamp from the first target's data
    targetState1 = rxMsg[0]
    tsMsg = targetState1[0]  # Shared timestamp for each target

    # Extract target states without the timestamp
    targetsState = {0: targetState1[1:].tolist()}  # Ensures at least one target is present

    # Append data for the additional targets if available
    if acquiredTargets > 1:
        targetState2 = rxMsg[1]
        targetsState[1] = targetState2[1:].tolist()

    if acquiredTargets > 2:
        targetState3 = rxMsg[2]
        targetsState[2] = targetState3[1:].tolist()

def callbackSstate(data):
    global senState
    tmp = data.data
    senState = tmp

def shutdown_cllbk():
    global auvID, avg_time, avg_nodes
    np.savetxt(log_path+'/wall_times'+str(auvID)+'.txt',avg_time)
    np.savetxt(log_path+'/nodes'+str(auvID)+'.txt',avg_nodes)
    
    magenta = "\033[0;35m"
    none = "\033[0m"
    rospy.loginfo('%s|---- OPTIMIZATION '+str(auvID)+': Simulation data saved --> Shutting down ...%s',
                magenta,none)

def listener(auvID: int,auvNum: int) -> None:
   
    rospy.Subscriber('/'+str(auvID)+'/estimation', Matrix, callbackTstate)
    rospy.Subscriber('vehicle_state_'+str(auvID), numpy_msg(Floats), callbackSstate)

    '''for i in range(auvNum):
        #if i + 1 != auvID:
            # Define a wrapper function to capture the index
        def create_callback(index):
            def callback(data):
                global plcyInt
                plcyInt[index] = list(data.data)
            return callback
        rospy.Subscriber('/'+str(i+1)+'/rx_ctrl_policy', numpy_msg(Floats), create_callback(i))'''



def compute_neighbor_policy(h,auv_id_neighbor, init_state_neighbor, x_hat_shared, pi_bar_placeholder,
                            ctrl_cmds, init_d, acousticParams, k_phi, auvNum, targetNum, netTopology):
    """
    Compute a plausible policy of intent for a neighbor AUV using local optimization.

    Parameters
    ----------
    auv_id_neighbor : int
        The ID of the neighbor AUV (1-based).
    init_state_neighbor : list
        Initial state of the neighbor [x, y, theta].
    x_hat_shared : list
        Shared estimate of the target state.
    pi_bar_placeholder : list of lists
        Placeholder pi_bar to initialize, e.g., all zeros.
    ctrl_cmds : list
        Heading command discretization.
    init_d : float
        Initial guess for distance to target.
    acousticParams : dict
        Acoustic constraints.
    k_phi : list
        Target-specific observability metric.
    auvNum : int
        Number of AUVs in the team.
    targetNum : int
        Number of targets (usually 1).
    netTopology : list
        Topology as per config.

    Returns
    -------
    policy_pi_bar_neighbor : list
        The locally computed policy of intent (pi_bar) for this neighbor.
    """

    # Instantiate the local MPC for the neighbor
    neighbor_mpc = h.bnb.Simple(
        auvNum=auvNum,
        auvID=auv_id_neighbor,
        targetNum=targetNum,
        x_hat=x_hat_shared,
        s=init_state_neighbor,
        ctrl_cmds=ctrl_cmds,
        pi_bar=pi_bar_placeholder,
        initState=[init_state_neighbor for _ in range(auvNum)],
        init_d=init_d,
        acousticParams=acousticParams,
        k_phi=k_phi,
        AUV_failure=False,
        localAUV=False,  # Local optimization for neighbor
    )

    # Run pybnb solver (limit time or tree depth)
    solver = Solver()
    solver.max_depth = h.config.H  # Limit search horizon
    with io.StringIO() as buf, redirect_stdout(buf):
        solver.solve(neighbor_mpc)

    # After solving, extract final control sequence
    heading_seq = neighbor_mpc._hedingChoices
    surge_seq = neighbor_mpc._surgeChoices

    # Build pi_bar format
    # Start from initial state
    x, y, theta = init_state_neighbor
    policy_pi_bar = [x, y, theta]

    for h, s in zip(heading_seq, surge_seq):
        policy_pi_bar.append(h)  # delta_theta
    for h, s in zip(heading_seq, surge_seq):
        policy_pi_bar.append(s)  # delta_surge

    return policy_pi_bar



def main():
    
    # ROS INIT   
    namespace = rospy.get_namespace()
    params_path = namespace+'auv'

    # Get AUV ID and number of vehicles.
    global auvID, auvNum, avg_nodes, avg_time, plcyInt, senState, tsMsg, tsMsg_old, acquiredTargets, targetsState, init_state, init_d
    auvID = rospy.get_param(params_path+'/auvID')
    auvNum = rospy.get_param(params_path+'/auvNum')
    targetNum = rospy.get_param(params_path+'/targetNum')
    # Node Init
    rospy.init_node('auv'+str(auvID)) #TO ADD debug prints --> log_level=rospy.DEBUG
    cyan = "\033[0;36m"
    none = "\033[0m"
    # ROS simulation parameters
    Hz = 1/(h.config.TIME_STEP) #NB: different from sampling rate for move things, this is ros rate   
    rate = rospy.Rate(Hz)
    listener(auvID,auvNum)

    # Init time variables and counters and lists
    t, count1, count_low, count_max, tsMsg_old = 0,0,0,0,0.0
    avg_time, avg_nodes, old_ctrls = [],[],[]
    pareto_data_c, pareto_data_g, pareto_data_d = [], [], []

    # Load simulation parames from config file
    sim_time, dt = h.config.TIME_DURATION, h.config.TIME_STEP
    u_max, delta_u, Ts, H = h.config.u_max, h.config.delta_u, h.config.Ts, h.config.H
    AUV_failure, ctrl_set = h.config.AUV_failure, h.config.ctrl_cmd
    acousticParams = [h.config.SL,h.config.NL,h.config.DI]
    DT = Ts*auvNum+60 #optimization time window
    netTopology = h.config.netTopology


    limit = 0.0 #Compute nodes limit according to RHC with finite memory
    U = h.config.U*3
    for i in range(H+1):
        limit += U**i

    # Initialize polices of intent and inital team state
    init_state = [[] for _ in range(auvNum)]
    for i in range(auvNum):
        tmp = [AUV_XY[i,j] for j in range(3)]+[0.0]*((H + 1) * 2)
        plcyInt[i] = tmp
        init_state[i] = tmp[0:3]
        
    initialized = [False for _ in range(targetNum)]
    init_d = [[] for _ in range(targetNum)]

    # Init publishers and subscribers
    pub_ctrl_policy = rospy.Publisher('/'+str(auvID)+'/ctrl_policy',numpy_msg(Floats),queue_size=100)

    rospy.sleep(1)
    while not rospy.is_shutdown():

        if t > sim_time/4:# for changing noise level during the sim.
            acousticParams[1] = 10#70

        if t > sim_time/2:#to simulate AUV failure during the sim.
            AUV_failure = True

        if tsMsg != tsMsg_old:#CHECK IF A NEW MSGS IS ARRIVED

            # Compute the relevant data structures from the callbacks variable
            #inputPlcy = [sublist[:] for sublist in plcyInt]

            xi_hat = [state[2:6] for state in targetsState.values()]
            k_phi = [state[0] for state in targetsState.values()]
            cov_list = [state[6:] for state in targetsState.values()]
            # TODO REMEMBER THE LABEL!!
            
            # Compute the current cost function:    
            estimator = h.estimator_module.Estimation()
            sensors, meas_table = [], []
            for i in range(auvNum):
                sensors.append(h.sensor.Sensor(str(i),1,0,0.000))

            for i in range(acquiredTargets):
                xi_i = xi_hat[i]
                if initialized[i] == False:
                    init_d[i] = np.sqrt((xi_i[1]-senState[1])**2+(xi_i[0]-senState[0])**2)
                
                # Simulate measurements for cost function computation
                for j, sensor in enumerate(sensors):
                    if j == auvID + 1:
                        tmp = [senState[0],senState[1],senState[2]]
                    else:
                        tmp = plcyInt[j]
                    measure_, rel_bearing_, meas_pos = sensor.measureBearing(xi_i[0], xi_i[1], 
                                                                        [tmp[0], tmp[1]], tmp[2])
                    meas_table.append([measure_, meas_pos[0], meas_pos[1]])              

                # Compute the regressor
                estimator.computeState(meas_table)
                # Filter out specific measurements based on network topology and/or failure status

                k_phi = h.utils.compute_cost(estimator.phi,1,auvID)#TODO CHECK THIS COMPUTATION
                meas_table = []

            # Dec.MPC (locally compute the policy of intent by means of optimization)

            start = time.time()
            for i in range(auvNum):
                if i + 1 != auvID:

                    j_pi_bar = plcyInt[i]
                    d0 = (np.sqrt((xi_i[1]-j_pi_bar[1])**2+(xi_i[0]-j_pi_bar[0])**2))

                    plcyInt[i] = compute_neighbor_policy(
                        h,
                        auv_id_neighbor=2,
                        init_state_neighbor=j_pi_bar[0:3],  # e.g., [x, y, theta]
                        x_hat_shared=xi_hat[0],
                        pi_bar_placeholder=[[0.0]*h.config.H*2 + [0.0]]*auvNum,
                        ctrl_cmds=ctrl_set,
                        init_d=d0,
                        acousticParams=acousticParams,
                        k_phi=k_phi,
                        auvNum=auvNum,
                        targetNum=targetNum,
                        netTopology=netTopology
                    )




            problem = h.bnb.Simple(auvNum, auvID, acquiredTargets, xi_hat[0], senState, ctrl_set,
                                        plcyInt, init_state, init_d[0], acousticParams, [k_phi], AUV_failure, True)                                        
            # Solve the optimization problem
            solver = h.bnb.pybnb.Solver()
            # Store the results

            
            with io.StringIO() as buf, redirect_stdout(buf):
                res = solver.solve(problem,queue_strategy="breadth",
                                    node_limit=limit,relative_gap=0.001)
            stop = time.time()
            rospy.loginfo('%s Optimization AUV%s done, elapsed time (s): %s. %s',cyan,auvID,stop-start,none)
            bns, wall_time, nodes = res.best_node.state, res.wall_time, res.nodes

            avg_nodes.append(nodes), avg_time.append(stop-start)

            headingChoices, surgeChoices = bns[5],bns[6]

            # Apply the Forbidden Decision Method
            headingChoices = h.frbdDcsMtd(headingChoices)
        
            #Formatting the results according to the communication protocol
            msg = [senState[0],senState[1],senState[2]]
            if len(headingChoices) == H:
                for i in range(H):
                    msg.append(headingChoices[i])               
                msg.append(0.0) #HEURISTIC FUNCTION to complete the POLICY OF INTENT
                for i in range(H):
                    msg.append(surgeChoices[i]) # append the vels for complete policy of intent
                msg.append(surgeChoices[-1]) 
            else:
                rospy.logwarn('UNFEASIBLE OPTIMIZATION - idle state')
                for i in range((H+1)*2):
                    msg.append(0.0)
            #HEURISTIC FUNCTION to complete the POLICY OF INTENT

            # Ros pub
            pub_ctrl_policy.publish(np.array(msg,dtype=np.float32))
            
            # Adapt Online ctrl set
            if len(headingChoices) > 1:
                old_ctrls.append(headingChoices[0])
            # Adapt online the heading changes
            if len(old_ctrls) == 3:
                ctrl_set, count_max, count_low, u_max = h.adptCtrlSet(old_ctrls,ctrl_set,
                                                                    count_low,count_max,u_max, delta_u)
                old_ctrls = []

        if int(t) == (h.config.TIME_DURATION-1):
            rospy.on_shutdown(shutdown_cllbk)
            rospy.signal_shutdown('Simulation time limit reached')
        tsMsg_old = tsMsg
        t += dt
        count1 += 1
        rate.sleep()

    rospy.on_shutdown(shutdown_cllbk)
    rospy.spin()
    
if __name__ == '__main__':
    
    main()
