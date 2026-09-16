"""Particle-swarm optimization planner for the UWMSN control problem.

The planner explores candidate heading and surge sequences and evaluates them
using the project-specific cost and constraint model.
"""

import numpy as np
import random
import os, pathlib, importlib.util

# Load the header file as a Python module 
pkg_directory = os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

header_file = pkg_directory+'/uwmsn-motion_opt'+'/include'+'/uwmsn-motion_opt'
log_path = pkg_directory+'/logs'

spec = importlib.util.spec_from_file_location("module.h", header_file+'/bnb_h.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

# Parameters optimization problem
Ts = h.config.Ts
d_max, d_min = h.config.max_distance, h.config.min_distance
gamma_w = h.config.gamma_w
f = h.config.f

def alphaFunc(f):
    
    return 0.11*(f**2/(1+f**2))+44*(f**2/(4100+f**2))+(2.75*(1e-4)*(f**2))+0.003

class PSOPlanner:
    """Search for a feasible control policy using a particle swarm optimizer."""

    def __init__(self, auvID, auvNum, H, ctrl_range, senState, target_state, 
                 plcyInt, netTopology, DT, init_d, max_iter=50, n_particles=30):
        self.auvID = auvID
        self.auvNum = auvNum
        self.H = H
        self.ctrl_range = ctrl_range  # assumed to be a tuple: (max_heading_delta, max_surge)
        self.senState = senState      # current state s = [x, y, theta]
        self.target_state = target_state  # e.g., x_hat
        self.plcyInt = plcyInt        # current pi_bar
        #self.simulate = simulation_fn # pointer to Simple.simulation method
        self.netTopology = netTopology
        self.auvFailure = h.config.AUV_failure  # Placeholder for AUV failure status
        self.DT = DT  # time step for simulation

        self.init_d = init_d 
        self.acousticParams = [h.config.SL,h.config.NL,h.config.DI]

        sensors = []
        for i in range(auvNum):
            sensors.append(h.sensor.Sensor(str(i),1,0,0.0))
        self._sensors = sensors

        self.max_iter = max_iter
        self.n_particles = n_particles

        self.heading_range = [-h.config.u_max,h.config.u_max]  # heading delta
        self.surge_range = [0.0, ctrl_range[1]]               # surge velocity

        # always one target 
        self._targetNum = 1

    def initialize_particles(self):
        particles = []
        
        # Create discrete sets of admissible values
        heading_choices = np.linspace(-h.config.u_max, h.config.u_max, num=7)  # example: [-u, ..., 0, ..., +u]
        surge_choices = np.linspace(0.0, self.ctrl_range[1], num=5)            # example: [0, ..., u_max]

        for _ in range(self.n_particles):
            heading_seq = np.random.choice(heading_choices, self.H)
            surge_seq = np.random.choice(surge_choices, self.H)
            particles.append((heading_seq, surge_seq))
    
        return particles

    def evaluate_particle(self, heading_seq, surge_seq):
        total_cost = 0.0
        s_pose = self.senState
        pi_bar = self.plcyInt
        x_hat = self.target_state

        for k in range(self.H):
            delta_theta = heading_seq[k]
            delta_u = surge_seq[k]

            # Run simulation at this step
            targetsEstState, regressors, s_pose, pi_bar = self.simulation(
                delta_theta, delta_u, x_hat, np.eye(4), s_pose, pi_bar
            )

            for idx in range(len(targetsEstState)):
                tmp_phi = regressors[idx].phi
                tmp_xi = targetsEstState[idx].x

                # Apply same cost breakdown as in BnB
                cost_d = h.config.RANGE_TO_TARGET / np.linalg.norm(np.array(tmp_xi[:2]) - np.array(s_pose[:2]))
                cost_g = 1 / h.utils.compute_cost(tmp_phi, cost_d, self.auvID)
                cost_c, pen_dm, pen_abs = self.applyConstraints(
                    pi_bar, tmp_xi, s_pose, self.DT,  self.init_d,
                    h.config.RANGE_TO_TARGET, self.auvID, 
                    self.acousticParams, False
                )

                if pen_dm == 1.0 or pen_abs == 1.0:
                    return np.inf  # discard infeasible particles

                cost = (1 - h.config.alpha_w) * cost_d + \
                       h.config.alpha_w * cost_g 
                total_cost += 1/cost

        return total_cost

    def optimize(self):
        # Discrete control options
        heading_choices = np.linspace(-h.config.u_max, h.config.u_max, num=7)
        surge_choices = np.linspace(0.0, self.ctrl_range[1], num=5)

        # Initialize particles
        particles = self.initialize_particles()
        personal_best = particles[:]
        personal_best_costs = [self.evaluate_particle(x, y) for x, y in particles]
        global_best_idx = np.argmin(personal_best_costs)
        global_best = personal_best[global_best_idx]

        for iter_ in range(self.max_iter):
            for i in range(self.n_particles):
                # Mutation: randomly tweak the current particle by changing a few values
                h_seq = particles[i][0].copy()
                s_seq = particles[i][1].copy()
                for k in range(self.H):
                    if np.random.rand() < 0.3:  # mutation probability
                        h_seq[k] = np.random.choice(heading_choices)
                    if np.random.rand() < 0.3:
                        s_seq[k] = np.random.choice(surge_choices)

                # Evaluate mutated particle
                cost = self.evaluate_particle(h_seq, s_seq)

                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best[i] = (h_seq, s_seq)
                    personal_best_costs[i] = cost

            # Update global best
            global_best_idx = np.argmin(personal_best_costs)
            global_best = personal_best[global_best_idx]

            # Rebuild particle swarm for next iteration centered around best
            for i in range(self.n_particles):
                h_seq = np.array([
                    np.random.choice([personal_best[i][0][k], global_best[0][k]]) if np.random.rand() < 0.5 else personal_best[i][0][k]
                    for k in range(self.H)
                ])
                s_seq = np.array([
                    np.random.choice([personal_best[i][1][k], global_best[1][k]]) if np.random.rand() < 0.5 else personal_best[i][1][k]
                    for k in range(self.H)
                ])
                particles[i] = (h_seq, s_seq)

        return global_best, personal_best_costs[global_best_idx]


    def beliefPropagation(self, pi_bar, N_i, s_i, r_i, u_i):
        pi_bar_out = []

        for i in range(self.auvNum):


            j_pi_bar = pi_bar[i]
            out = []

            H = int((len(j_pi_bar) - 3) / 2)  # Number of horizon steps
            idx1 = 3 + H  # Index of the first surge delta

            # Check if j_pi_bar is valid or if the current AUV should compute its own path
            if (len(j_pi_bar) > 0 and np.sum(j_pi_bar) != 0) or self.auvID == i + 1:
                if self.auvID == i + 1:  # Compute the path for the current AUV
                    ax = np.cos(s_i[2] + r_i) * u_i * self.DT + s_i[0]
                    ay = np.sin(s_i[2] + r_i) * u_i * self.DT + s_i[1]
                    out = [ax, ay, s_i[2] + r_i, r_i, u_i]
                else:
                    ax_j, ay_j, theta_j = j_pi_bar[0], j_pi_bar[1], j_pi_bar[2]
                    for h in range(H):
                        delta_theta_h = j_pi_bar[3 + h]  # Delta theta for step h
                        delta_surge_h = j_pi_bar[idx1 + h]  # Delta surge for step h
                        theta_j += delta_theta_h
                        ax_j += np.cos(theta_j) * delta_surge_h * self.DT
                        ay_j += np.sin(theta_j) * delta_surge_h * self.DT
                        out.extend([delta_theta_h, delta_surge_h])  # Append step deltas to output

                    N_i[i] = [ax_j, ay_j, theta_j]
                    out = [ax_j, ay_j, theta_j] + out

                    if self.auvID != i + 1:
                        # Remove the delta values from j_pi_bar to keep its size consistent
                        j_pi_bar = np.delete(j_pi_bar, np.s_[3:3 + H])  # Remove delta_theta values
                        j_pi_bar = np.delete(j_pi_bar, np.s_[idx1 - H:idx1])  # Remove delta_surge values

                        for j in range(len(j_pi_bar[3:])):
                            out.append(j_pi_bar[j + 3])

            else:  # Default case for invalid j_pi_bar
                N_i[i] = [0.0, 0.0, 0.0]
                j_pi_bar = np.delete(j_pi_bar, np.s_[3:3 + H])  # Remove delta_theta values
                j_pi_bar = np.delete(j_pi_bar, np.s_[idx1 - H:idx1])  # Remove delta_surge values
                out.extend([0.0, 0.0] * H)  # Default deltas for invalid inputs
                out.extend([0.0] * len(j_pi_bar))

            pi_bar_out.append(out)

        return pi_bar_out, N_i, [ax, ay, s_i[2] + r_i, r_i, u_i]

    def simulation(self, delta_theta, delta_u, x_hat, P, s_pose, pi_bar):
        # Initialize data structures and classes
        meas_table = []
        targets, estimators = [], []
        for i in range(self._targetNum):
            targets.append(h.target_module.Target(x_hat, self.DT, P))
            estimators.append(h.estimator_module.Estimation())

        # Initialize AUVs state structures
        auvs_xy = [[0.0 for _ in range(len(s_pose) + 1 + h.config.H)]
                for _ in range(self.auvNum)]

        # Propagate the AUVs state
        pi_bar_out, auvs_xy, s_pose = self.beliefPropagation(pi_bar, auvs_xy, s_pose, delta_theta,
                                                            delta_u)

        for j in range(len(targets)):
            target = targets[j]
            # Simulate measurements for cost function computation
            for i, sensor in enumerate(self._sensors):
                
                tmp = s_pose if self.auvID == i + 1 else (auvs_xy[i] if 
                                                        np.any(auvs_xy[i][:3]) else self.senState[i])
                
                measure_, rel_bearing_, meas_pos = sensor.measureBearing(target.x[0], target.x[1], 
                                                                        [tmp[0], tmp[1]], tmp[2])
                meas_table.append([measure_, meas_pos[0], meas_pos[1]])

            # Filter out specific measurements based on network topology and/or failure status
            '''if self.auvFailure == False:
                for i in range(self.auvNum):
                    if i+1 not in self.netTopology and i+1 != self.auvID:
                        meas_table.pop(i)
            else:
                meas_table.pop(2)'''

            # Compute the regressor
            estimators[j].computeState(meas_table)

        return targets, estimators, s_pose, pi_bar_out
    
    def applyConstraints(self, tmp_pi_bar, xi_hat, tmp_s, DT, init_d, desRange, auvID, acousticParams, AUV_failure):
    
        pen_dm, pen_abs = 0, 0
        snr = []
        old_tmp_positions = []
        loops = len(tmp_pi_bar)
        SL, NL, DI = acousticParams[0], acousticParams[1], acousticParams[2]
        
        # Update xi_hat position incrementally for each loop iteration
        xi_hat[:2] += DT * xi_hat[2:4]
        tmp_d_target = np.linalg.norm([xi_hat[1] - tmp_s[1], xi_hat[0] - tmp_s[0]])

        # Penalty for exceeding target distance constraints
        '''if tmp_d_target > (2 * init_d) or tmp_d_target <= desRange / 2:
            print('auvID Penalty Distance', auvID)
            print('tmp_d_target', tmp_d_target, 'init_d', init_d, 'desRange', desRange)
            pen_abs = 1.0'''

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
                snr.append(tmp_SNR if h.config.SNR_lb < tmp_SNR < h.config.SNR_ub and d_ij > 0 else 0)

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
                if h.config.SNR_lb < tmp_SNR < h.config.SNR_ub and d_ij > 0:
                    adj_matrix[i, j] = -tmp_SNR / h.config.SNR_ub
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
