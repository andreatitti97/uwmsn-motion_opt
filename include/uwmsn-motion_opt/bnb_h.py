import os, pathlib, importlib
import numpy as np
import math

ws_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
pkg_directory = os.path.dirname(os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve())))
local_directory = ws_directory+"/src/Classes"
external_directory = pkg_directory+"/uwmsn-sim/src/Classes"

spec = importlib.util.spec_from_file_location("module.utils", local_directory+'/utils_opt.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("module.estimator", local_directory+'/estimator.py')
estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator_module)

spec = importlib.util.spec_from_file_location("module.target", local_directory+'/target.py')
target_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target_module)

spec = importlib.util.spec_from_file_location("module.config", external_directory+"/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location("module.sensor", external_directory+"/sensor.py")
sensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensor)

# Parameters optimization problem
Ts = config.Ts
d_max, d_min = config.max_distance, config.min_distance
gamma_w = config.gamma_w
f = config.f

def alphaFunc(f):
    
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

def applyConstraints(tmp_pi_bar, xi_hat, tmp_s, DT, init_d, desRange, auvID, acousticParams, AUV_failure):
    
    pen_dm, pen_abs = 0, 0
    snr = []
    old_tmp_positions = []
    loops = len(tmp_pi_bar)
    SL,NL,DI = acousticParams[0],acousticParams[1],acousticParams[2]
    
    # Update xi_hat position incrementally for each loop iteration
    xi_hat[:2] += DT * xi_hat[2:4]
    tmp_d_target = np.linalg.norm([xi_hat[1] - tmp_s[1], xi_hat[0] - tmp_s[0]])

    # Penalties for target distance constraints
    if tmp_d_target > (2*init_d) or tmp_d_target <= desRange/2:
        pen_abs = 1.0

    # Compute expected signal to noise ratio between local AUV and his neighbours
    for i in range(loops):
        j_pi_bar = tmp_pi_bar[i]
        if len(j_pi_bar) >= 5 and i != auvID - 1:
            H = (len(j_pi_bar) - 3) // 2
            # Calculate AUV's relative position
            tmp_x = np.cos(j_pi_bar[2] + j_pi_bar[3]) * j_pi_bar[4] * DT + j_pi_bar[0]
            tmp_y = np.sin(j_pi_bar[2] + j_pi_bar[3]) * j_pi_bar[4] * DT + j_pi_bar[1]
            d_ij = np.linalg.norm([tmp_y - tmp_s[1], tmp_x - tmp_s[0]])
            old_tmp_positions.append((tmp_x, tmp_y))

            # Calculate transmission loss and SNR
            TL = 20 * np.log10(d_ij) + (d_ij * alphaFunc(f) * 1e-3)
            tmp_SNR = SL - TL - NL + DI
            snr.append(tmp_SNR if config.SNR_lb < tmp_SNR < config.SNR_ub and d_ij > 0 else 0)

            # Collision avoidance constraint
            if (auvID == 1 and i == 1) or (auvID == 2 and i != 1) or (auvID == 3 and i == 1):
                if d_ij <= d_min:
                    pen_dm = 1.0

    # Calculate pairwise SNR between AUVs that are not the local one using the policies of intent
    if len(old_tmp_positions) > 1:
        d_jj = np.linalg.norm(np.subtract(*old_tmp_positions[:2]))
        TL = 20 * np.log10(d_jj) + (d_jj * alphaFunc(f) * 1e-3)
        tmp_SNR = (SL - TL - NL + DI)
        snr.append(tmp_SNR if config.SNR_lb < tmp_SNR < config.SNR_ub and d_ij > 0 else 0)

    # Construct Laplacian matrix with calculated SNR values
    laplacian = np.zeros((loops, loops))
    snr = [snr[i]/config.SNR_ub for i in range(len(snr))]
    laplacian = np.zeros((loops,loops))
    
    laplacian[0,1] = -snr[0]
    laplacian[1,0] = -snr[0]
    laplacian[0,2] = 0
    laplacian[2,0] = 0

    laplacian[0,0] = snr[0]
    laplacian[1,1] = snr[0]+snr[1]
    laplacian[2,2] = snr[1]
    laplacian[1,2] = -snr[1]
    laplacian[2,1] = -snr[1]

    # Compute SVD to get max singular value
    _, S, _ = np.linalg.svd(laplacian)
    max_sigma = max(S[1], 0)  # Ensure non-negative sigma for graph connectivity

    return max_sigma, pen_dm, pen_abs

def normalizeObjFunc(cost2,cost1):#g and c
    if cost1 > 10e-6:
        ord_1 = math.floor(math.log(cost1, 10))
        ord_2 = math.floor(math.log(cost2, 10))
        if cost1>cost2:  
            alpha = 10**(ord_2-ord_1)
            beta = 1.0

        else:
            beta = 10**(ord_1-ord_2)
            alpha = 1.0
    else:
        beta, alpha = 1.0, 1.0

    return alpha*cost2, beta*cost1
