#!/usr/bin/env python
import os
import importlib.util, pathlib
import numpy as np

# Import Costum classes
local_path = pathlib.Path(__file__).parent.resolve()
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(local_path)))

pkg_directory = pkg_directory+'/uwmsn-sim'+'/src'+'/Classes'
local_directory = os.path.dirname(os.path.dirname(local_path))+'/src'

spec = importlib.util.spec_from_file_location("module.config", pkg_directory+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location("module.bnb_MTT", local_directory+'/bnb_MTT.py')
bnb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnb)

spec = importlib.util.spec_from_file_location("module.planner", local_directory+'/PSO_planner.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)

spec = importlib.util.spec_from_file_location("module.estimator", local_directory+'/Classes/estimator.py')
estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator_module)

spec = importlib.util.spec_from_file_location("module.utils", local_directory+'/Classes/utils_opt.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("module.sensor", pkg_directory+"/sensor.py")
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)



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
                if u_max <= config.MIN:
                    u_max = u_max
                    count_low = 0
                else:
                    u_max  = u_max  - delta_u
                    count_low = 0
                    
                    
        elif np.abs(old_ctrls[i]) >= u_max-(1e-3):
            count_max += 1
            if count_max == 3:
                if u_max >= config.MAX:
                    u_max = u_max
                    count_max = 0
                else:
                    u_max  = u_max  + delta_u
                    count_max = 0
                    
    ctrl_set = []
    u_i = u_max/((U-1)/2)
    for i in range(U):
        if i <= U/2:
            ctrl_set.append(u_max-i*u_i)
        if i > U/2:
            ctrl_set.append((i-((U-1)/2))*u_i)
    
    return ctrl_set, count_max, count_low, u_max