import os, pathlib, importlib
import numpy as np

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

d_max, d_min = config.max_distance, config.min_distance
Ts = config.Ts
# Parameters optimization problem
gamma_w = config.gamma_w
alpha_w = config.alpha_w

def alpha(f):
    
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

def connectivityCost(tmp_pi_bar, tmp_s, DT, init_d,
                     x_hat, _desRange, auvID, NL, AUV_failure):

    pen_dm, pen_dM, pen_abs = 0, 0, 0
    snr, tmp, old_tmp_x, old_tmp_y = [], [], [], []
    loops = len(tmp_pi_bar)
    
    acoustic_loss = alpha(config.f) #f is in kHz
    TL_ideal = 20*np.log(d_min) + (d_min*acoustic_loss*1e-3)
    ideal_snr = (config.SL - TL_ideal - NL + config.DI)#SNR without transmission loss--> to bound values.
   
    for i in range(loops):
        j_pi_bar = tmp_pi_bar[i]
        x_hat[0] = x_hat[0]+DT*x_hat[2]
        x_hat[1] = x_hat[1]+DT*x_hat[3]
        tmp_d_target = np.sqrt((x_hat[1]-tmp_s[1])**2+(x_hat[0]-tmp_s[0])**2)
       
        H = int(((len(j_pi_bar) - 3)/2))
        idx2 = 3 + H
        
        if len(j_pi_bar) >= 5:
            if i != auvID-1:# j pi bar order is the same for everyone
                
                tmp_x = np.cos(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[0]
                tmp_y = np.sin(j_pi_bar[2]+j_pi_bar[idx2])*j_pi_bar[3]*DT+j_pi_bar[1]
                old_tmp_x.append(tmp_x)
                old_tmp_y.append(tmp_y)
                tmp = np.sqrt((tmp_y-tmp_s[1])**2+(tmp_x-tmp_s[0])**2)
    
                TL = 20*np.log(tmp) + (tmp*acoustic_loss*1e-3)
                
                tmp_snr = config.SL - TL - config.NL + config.DI
                tmp_snr = tmp_snr/ideal_snr

                if tmp_snr >= config.DThresh/ideal_snr:
                    if tmp_snr > 1.0:
                        snr.append(0.0)
                    else:
                        snr.append(tmp_snr)
                else:
                    snr.append(0)

            # TODO: Generalize the formula -- Collision Avoidance Constraint
            if (auvID == 1 and i == 1) or (auvID == 2 and i != 1) or (auvID == 3 and i == 1):
                if tmp <= d_min:
                    pen_dm = 1.0
                    #if auvID == 2:
                    #    print('NOT SAFE')

        # Constraint: stay to a certain vicinity of the target
        if tmp_d_target > init_d:

            pen_abs = 0.0#TODO better because like this does not make to much sense
            # first of all because can happen that even performing the best strategy
            # in no possible to full fill this requirement (while for the one below is different
            #if you can stop if optimizing surge)
        elif tmp_d_target <= _desRange:
            pen_abs = 1.0
            print('TO NEAR')

    # Expected SNR between the AUVs with ID != from current ID
    d_ij = np.sqrt((old_tmp_y[1]-old_tmp_y[0])**2+(old_tmp_x[1]-old_tmp_x[0])**2)
    TL = 20*np.log(d_ij) + (d_ij*acoustic_loss*1e-3)   
    tmp_snr = config.SL - TL - config.NL + config.DI
    # normalize the snr according to the desired one, i.e., 1/4 of the SL (0 is not realistic)
    tmp_snr = tmp_snr/ideal_snr

    if tmp_snr >= config.DThresh/ideal_snr and d_ij>0:
        if tmp_snr > 1.0:
            snr.append(0.0)
        else:
            snr.append(tmp_snr)
    else:
        snr.append(0)

    laplacian = np.zeros((loops,loops))

    if AUV_failure == False:

        laplacian[0,1] = -snr[0]
        laplacian[1,0] = -snr[0]
        laplacian[0,2] = 0
        laplacian[2,0] = 0

        laplacian[0,0] = snr[0]
        laplacian[1,1] = snr[0]+snr[1]
        laplacian[2,2] = snr[1]
        laplacian[1,2] = -snr[1]
        laplacian[2,1] = -snr[1]

        if config.AUV2_bridge == False:
            laplacian = np.zeros((2,2))
            ''' TO DO '''
        else:
            if auvID == 1:
                laplacian[0,1] = -snr[0]
                laplacian[1,0] = -snr[0]
                laplacian[0,2] = 0
                laplacian[2,0] = 0

                laplacian[0,0] = snr[0]
                laplacian[1,1] = snr[0]+snr[2]
                laplacian[2,2] = snr[2]
                laplacian[1,2] = -snr[2]
                laplacian[2,1] = -snr[2]
            if auvID == 3:
                laplacian[0,1] = -snr[2]
                laplacian[1,0] = -snr[2]
                laplacian[0,2] = 0
                laplacian[2,0] = 0

                laplacian[0,0] = snr[2]
                laplacian[1,1] = snr[1]+snr[2]
                laplacian[2,2] = snr[1]
                laplacian[1,2] = -snr[1]
                laplacian[2,1] = -snr[1]
    else:
        laplacian = np.zeros((2,2))
        if auvID == 1:
            laplacian[0,0] = snr[1]
            laplacian[1,1] = snr[1]
            laplacian[0,1] = -snr[1]
            laplacian[1,0] = -snr[1]
        elif auvID == 3:
            laplacian[0,0] = snr[0]
            laplacian[1,1] = snr[0]
            laplacian[0,1] = -snr[0]
            laplacian[1,0] = -snr[0]

    #if auvID == 1:
        #print(laplacian)

    [U, S, vh] = np.linalg.svd(laplacian)
    max_sigma = S[1]

    if max_sigma < 0: #if the graph is disconnected
        max_sigma = 0

    return max_sigma, pen_dm, pen_abs