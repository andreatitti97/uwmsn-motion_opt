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
    SL, NL, DI = acousticParams[0], acousticParams[1], acousticParams[2]
    
    # Update xi_hat position incrementally for each loop iteration
    xi_hat[:2] += DT * xi_hat[2:4]
    tmp_d_target = np.linalg.norm([xi_hat[1] - tmp_s[1], xi_hat[0] - tmp_s[0]])

    # Penalty for exceeding target distance constraints
    if tmp_d_target > (2 * init_d) or tmp_d_target <= desRange / 2:
        print('auvID Penalty Distance', auvID)
        pen_abs = 1.0

    # Compute expected SNR between the local AUV and its neighbors
    for i in range(loops):
        j_pi_bar = tmp_pi_bar[i]
        if len(j_pi_bar) >= 5 and i != auvID - 1:
            
            # Compute predicted position of neighbor
            tmp_x = np.cos(j_pi_bar[2] + j_pi_bar[3]) * j_pi_bar[4] * DT + j_pi_bar[0]
            tmp_y = np.sin(j_pi_bar[2] + j_pi_bar[3]) * j_pi_bar[4] * DT + j_pi_bar[1]
            d_ij = np.linalg.norm([tmp_y - tmp_s[1], tmp_x - tmp_s[0]])
            old_tmp_positions.append((tmp_x, tmp_y))

            # Compute transmission loss and SNR
            TL = 20 * np.log10(d_ij) + (d_ij * alphaFunc(f) * 1e-3)
            tmp_SNR = SL - TL - NL + DI
            snr.append(tmp_SNR if config.SNR_lb < tmp_SNR < config.SNR_ub and d_ij > 0 else 0)

            # Collision avoidance constraint
            if d_ij <= d_min:
                print('auvID Penalty Safety', auvID)
                pen_dm = 1.0

    # Compute pairwise SNR between AUVs using the propagated positions
    adj_matrix = np.zeros((loops, loops))  # Initialize adjacency matrix
    for i in range(len(old_tmp_positions)):
        for j in range(i + 1, len(old_tmp_positions)):  # Avoid redundant calculations
            d_ij = np.linalg.norm(np.subtract(old_tmp_positions[i], old_tmp_positions[j]))
            TL = 20 * np.log10(d_ij) + (d_ij * alphaFunc(f) * 1e-3)
            tmp_SNR = SL - TL - NL + DI
            if config.SNR_lb < tmp_SNR < config.SNR_ub and d_ij > 0:
                adj_matrix[i, j] = -tmp_SNR / config.SNR_ub
                adj_matrix[j, i] = adj_matrix[i, j]  # Symmetric matrix

    # Compute the degree matrix
    degree_matrix = np.diag(np.abs(adj_matrix).sum(axis=1))

    # Compute the Laplacian matrix
    laplacian = degree_matrix - adj_matrix

    # Compute singular values
    singular_values = np.linalg.svd(laplacian, compute_uv=False)

    # Extract the second smallest singular value
    if len(singular_values) > 1:
        sigma_2 = np.sort(singular_values)[1]  # Second smallest
    else:
        sigma_2 = 0  # Default if only one singular value exists

    # Compute SVD to get max singular value
    #_, S, _ = np.linalg.svd(laplacian)
    #sigma2 = max(S[1], 0)  # Ensure non-negative sigma for graph connectivity

    return sigma_2, pen_dm, pen_abs

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
