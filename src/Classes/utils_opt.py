#Import basic system modules
import os, pathlib, importlib.util
# Import math modules
import numpy as np
from math import sqrt

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

    predicted_pose = [0,0]
    
    tmp_s_pose = [s_pose[0],s_pose[1],s_pose[2]]
    tmp_curr_est = [curr_est[0],curr_est[1],curr_est[2],curr_est[3]]
    
    predicted_pose[0] = tmp_curr_est[0] + config.DT*tmp_curr_est[2]
    predicted_pose[1] = tmp_curr_est[1] + config.DT*tmp_curr_est[3]

    eucl_dist = np.sqrt((predicted_pose[0]-tmp_s_pose[0])**2+(predicted_pose[1]-tmp_s_pose[1])**2)
    epsi = config.RANGE_TO_TARGET #DISTANCA VOLUTA DAL TARGET
    
    beta = 1/d_max #coeficente angolare retta per due punti m = y2-y1/x1-x2 
    x = (eucl_dist-epsi)
    v_n = beta*x

    if config.TARGET_INIT[3] == 0:
        if eucl_dist <= epsi: #ONLY IF THE TARGET IS STATIC
            v_n = -10**3
    if 1 < x < epsi:
        v_n = np.sqrt((tmp_curr_est[2])**2+(tmp_curr_est[3])**2)
    elif v_n > config.AUV_MAX_VEL:
        v_n = config.AUV_MAX_VEL

    return v_n