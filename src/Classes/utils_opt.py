#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve())))

config_file_dir = pkg_directory+'/uwmsn-sim/src/Classes'
log_path = pkg_directory+'/uwmsn-sim'+'/logs'

spec = importlib.util.spec_from_file_location("module.config", config_file_dir+'/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def compute_cost(phi):

    length_y = len(phi)
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        W[i,i] = 1.0

    PHI = np.dot(np.transpose(tmp_phi),np.dot(np.linalg.inv(W),tmp_phi))

    return np.linalg.norm(np.linalg.inv(PHI),ord=2)*np.linalg.norm(PHI,ord=2)
   
def compute_cost2(phi):
    # Compute Covariance of the target state
    R = np.zeros((len(phi),len(phi))) #matrice diagonale perchè errori sulle singole misure indipendenti tra loro            
    for i in range(len(phi)): 
        for j in range(len(phi)):
            if i == j:
                R[i,j] = (config.SIGMA_MEAS)
            else:
                R[i,j] = 0 

    a = config.SIGMA_MEAS
    cov = np.linalg.inv(np.dot(np.dot(np.transpose(phi),np.linalg.inv(a*np.identity(len(phi)))),phi))

    return np.trace(cov)

def sig(x,d_max,alpha):

    return 1/(1 + np.exp(alpha*(-x+(d_max-config.RANGE_TO_TARGET)/2)))

def computePursuitVel(curr_est,s_pose,d_max):

    predicted_pose = np.array(np.zeros(2))
    predicted_pose[0] = curr_est[0] + config.DT*curr_est[2]
    predicted_pose[1] = curr_est[1] + config.DT*curr_est[3]

    eucl_dist = np.sqrt((predicted_pose[0]-s_pose[0])**2+(predicted_pose[1]-s_pose[1])**2)
    epsi = config.RANGE_TO_TARGET #DISTANCA VOLUTA DAL TARGET
    
    alpha = 0.09
    x = (eucl_dist-epsi)
    
    weigth = 1/(1 + np.exp(alpha*(-x+d_max/2))) #sigmoidal behaviour
    #weigth = -alpha*x #linear behaviour
    v_n = weigth*config.AUV_MAX_VEL
    
    if x < 1:
        v_n = -10**3

    return v_n