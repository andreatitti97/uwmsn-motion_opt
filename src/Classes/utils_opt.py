#Import basic system modules
import os, pathlib, importlib.util
import time
# Import math modules
import numpy as np
import matplotlib.pyplot as plt
# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve())))


config_file_dir = pkg_directory+'/uwmsn-sim/src/Classes'
log_path = pkg_directory+'/uwmsn-sim'+'/logs'


spec = importlib.util.spec_from_file_location("module.config", config_file_dir+'/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location("module.planner", config_file_dir+'/spline_planner.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)

# Global Variables
cubicSpline = planner

def update_path(ax, ay, waypoint, ayaw, desired_vel, DT):
        
    a_i = [ax[-1],ay[-1]]
    
    ax.append(np.cos(ayaw+waypoint)*desired_vel*DT+a_i[0])
    ay.append(np.sin(ayaw+waypoint)*desired_vel*DT+a_i[1])
    
    path = cubicSpline.CubicSpline2D(ax, ay)
    ayaw = ayaw+waypoint
    return path, ax, ay, ayaw

def compute_cost(phi, ranges, d_max, auvID):
    length_y = len(phi)
    tmp_phi = np.zeros((length_y,2))
    for i in range(length_y):
        a = phi[i]
        tmp_phi[i,:] = [a[0],a[1]]
    
    W = np.zeros((length_y,length_y))
    for i in range(length_y):
        if i == auvID+1:
            #W[i,i] = sig(ranges,d_max,-0.8)# RE-WEIGHTED LEAST SQUARE over the RANGE
            W[i,i] = 1.0
        else:
            W[i,i] = 1.0
    PHI = np.dot(np.transpose(tmp_phi),np.dot(np.linalg.inv(W),tmp_phi))
    x = np.linspace(0,d_max)
    '''plt.plot(x,sig(x,d_max,-0.08))
    plt.grid()
    plt.show()
    '''
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

    return 1/(1 + np.exp(alpha*(-x+d_max/2)))

