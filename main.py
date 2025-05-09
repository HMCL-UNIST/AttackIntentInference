import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from utils import *
from model import *
from model_jax import *
from ckf import *
from plot import *
from inference import *

import jax
import jax.numpy as jnp

import warnings
warnings.filterwarnings('ignore')


''' Run a single simulation of HGV attack intent inference '''
def test(foldername, iteration):
    

    ''' Target Initialization'''
    # Define ground targets (altitude, latitude [deg], longitude [deg])
    targets = [
        {"altitude": 0, "latitude": 9.8, "longitude": -5},     # Target 1
        {"altitude": 0, "latitude": 12, "longitude": -3},      # Target 2
        {"altitude": 0, "latitude": 13.5, "longitude": 0.0},   # Target 3
        {"altitude": 0, "latitude": 12, "longitude": 3},      # Target 4
        {"altitude": 0, "latitude": 10.5, "longitude": 5},    #  Target 5
        {"altitude": 0, "latitude": 8.5, "longitude": 6},     #  Target 6
    ]
    num_target = len(targets)
    target_region = 0.5
    
    #Assign random threat levels and select the highest threat value target
    threat_value = np.random.random(num_target)
    current_target_index = np.argmax(threat_value)
    current_target = targets[current_target_index]

    ''' Vehicle Model Initialization'''
    v_init = np.random.randint(3100,3150)
    h_init = earth_radius + np.random.randint(54500,55000)
    theta_int = -np.random.random()*np.radians(0.1)
    psi_int = 0
    lat_init = 0
    lon_init = 0
    current_true_state = [h_init, lon_init, lat_init, v_init, theta_int, psi_int] 
    state_dim = 6
    u_dim = 3
    hgv = HGV()
    velocity_command_set=np.array([[1000,500],[700,500],[1200,500],[700,500], [1500,1000],[2000,1000] ])
    hgv.set_v1v2(velocity_command_set[current_target_index,0],velocity_command_set[current_target_index,1])
    true_states =[current_true_state]

    ''' Simulation Parameters Initialization'''
    time_step = 2
    simul_time = 400
    measurement_noise = np.array([5, np.radians(0.0005), np.radians(0.0005), 2, np.radians(0.05), np.radians(0.05)])
    PNcontrol = False
    save_result = True # Ture: If you want to save the results
    Ns = 2000

    ''' CKF and Bayesian Inference Initialization'''
    estimated_vehicle_state = current_true_state.copy() + [0.0002, 0.0005, 0.0]
    estimated_states = [estimated_vehicle_state]
    Q = np.diag([1, 1e-3, 1e-3, 1, 1e-3, 1e-3, .00005, .00005, .0001])
    R = np.diag([5, np.radians(0.0005), np.radians(0.0005), 2, np.radians(0.05), np.radians(0.05)])
    P = np.diag([1, 1e-3, 1e-3, 1, 1e-3, 1e-3, .00005, .00005, .0001])
    ckf = CKF(state_dim+u_dim, state_dim, Q, R, P)
    ckf.X = estimated_vehicle_state
    inference = Inference(targets, num_target, target_region, threat_value, state_dim)
    attack_intent = []

    ''' visualization '''
    color = ['darkgreen', 'deeppink', 'lightseagreen', 'navy', 'yellowgreen', 'tomato', 'saddlebrown', 'darkblue', 'violet', 'darkorange']
    plot_propagation = False  # Ture: If you want to plot the sampled trajectories with current estimated state
    

    print(f" Simulation {iteration}: HGV attack Target {current_target_index}")
    print("Threat", np.round(threat_value,3))
    for t in range(simul_time):
        
        # Terminate if HGV hits ground
        Terminal = stop_at_ground(t, current_true_state, current_target)
        if not Terminal: 
            break
        
        # Enable PN control if close to target
        Distance = np.sqrt((current_true_state[1]-np.radians(current_target["longitude"]))**2 + (current_true_state[2]-np.radians(current_target["latitude"]))**2)
        if Distance <= np.radians(9):
            PNcontrol= True

        if current_target_index != 2 and PNcontrol:
            sigma_c = hgv.PN_control(current_true_state, current_target)
        else:
            sigma_c = 0.0

        # Propagate HGV state using dynamics model
        sol = solve_ivp(
            hgv.HGV_dynamics,
            (0, time_step),
            current_true_state,
            args=(sigma_c,),
            t_eval=[time_step],
            events=stop_at_ground,
            method='RK45'
        )
        hgv.prev_sigma = sigma_c
        current_true_state = sol.y.reshape(6,)
        
        # Add Gaussian noise to obtain measurement
        measurement_state = current_true_state + np.random.normal(0, measurement_noise)

        # CKF predict and update step
        pred_vehicle_state, P_pred = ckf.predict(time_step)
        estimated_vehicle_state, P_est = ckf.update(pred_vehicle_state, P_pred, measurement_state)

        # Run inference if probability hasn't converged
        if t >= 10 and (any(np.round(attack_intent[-30:-1],3)[:,current_target_index] < 1)): 
            key = jax.random.PRNGKey(t + 999)
            key_state, key_u_k = jax.random.split(key, 2)

            # Sample initial states and controls
            mean_state = jnp.array(ckf.X[:8])
            std_state = jnp.diag(jnp.array(ckf.P[:8, :8]))
            eps = jax.random.normal(key_state, shape=(Ns, 8))
            samples_all = mean_state + eps * std_state
            sampled_states, u_i_samples, u_j_samples = samples_all[:, :6], samples_all[:, 6], samples_all[:, 7]


            key_normal, key_uniform, key_mask = jax.random.split(key_u_k, 3)
            uk_uniform = jax.random.uniform(key_uniform, shape=(Ns,), minval=-jnp.pi/4, maxval=jnp.pi/4)
            # uk_normal = jax.random.normal(key_normal, shape=(Ns,)) * jnp.sqrt(ckf.P[8,8]) + ckf.P[8,8]
            # mask = jax.random.bernoulli(key_mask, p=0.1, shape=(Ns,))
            # u_k_samples = jnp.where(mask, uk_normal, uk_uniform)


            # Rollout sampled trajectories
            traj, impact_index = rollout_until_ground_hit(
                sampled_states,
                u_i_samples,
                u_j_samples,
                uk_uniform,
                time_step
            )

            valid_mask = impact_index != -1
            traj = traj[valid_mask]
            impact_index = impact_index[valid_mask]

            impact_lon_lat = jnp.degrees(jnp.stack([
                traj[jnp.arange(traj.shape[0]), impact_index, 1],
                traj[jnp.arange(traj.shape[0]), impact_index, 2]
            ], axis=1)) 

            # Perform reachability analysis + Bayesian inference
            inference.compute_reachability(targets, traj, impact_lon_lat)
            inference.bayesian(measurement_state)

            if plot_propagation: # Optional: plot trajectories in 3D
                plt.ion() 
                fig , ax = plot_trajectory_pro(np.array(true_states), estimated_vehicle_state, targets, current_target_index, target_region, figure_num=0, return_fig = True)
                for j in range(len(inference.attack_hits_ind)):
                    x_vals = []
                    for _, index in enumerate(inference.attack_hits_ind[j]):
                        alt = (traj[index,:impact_index[index],0] - earth_radius)/1000
                        lon = np.degrees(traj[index,:impact_index[index],1])
                        lat = np.degrees(traj[index,:impact_index[index],2])
                        if j >= num_target:
                            if alt[-1] >= 1e2:
                                continue
                        else:
                            ax.plot(lon, lat, alt, color = color[j])

                ax.view_init(30,60)
                plt.draw()
                plt.pause(0.001)
        print_progress_bar(t + 1, simul_time, prefix=f"Sim {iteration}")

        # Update CKF with new estimates
        ckf.X = estimated_vehicle_state
        ckf.P = P_est

        # Record states and intent inference
        estimated_states.append(estimated_vehicle_state)
        true_states.append(current_true_state)
        attack_intent.append(inference.prob_delta)
    
    print(f"Simulation {iteration} is Done")
    print(f"  Sce{iteration}//Attack Probability of Target {current_target_index} at Time {t}: {100*inference.prob_delta[current_target_index]:.2f}% ")
    print("Beta: ", np.round(inference.prob_beta_truncated_mu[current_target_index], 4))

    # Save results
    if save_result:
        true_states = np.array(true_states)
        estimated_states = np.array(estimated_states)
        np.savez(f'./{foldername}/result_{iteration}.npz' ,
            threat_value = threat_value,
            current_target_index = current_target_index,
            prob_delta = attack_intent,
            estimated_states = estimated_states,
            true_states = true_states)
        
if __name__ == "__main__":
    iteration = 20
    foldername = 'Result'
    plot_result = True

    for itr in range(iteration):
        test(foldername = foldername, iteration=itr)

    if plot_result:
        prob_delta_arr = []
        ind_min = 400
        for i in range(iteration):
            result = np.load(f'./{foldername}/result_{i}.npz')
            current_idx = result['current_target_index']
            prob_delta_arr.append(result['prob_delta'][:ind_min][:,current_idx])

        mean_delta = np.mean(prob_delta_arr, axis =0)
        std_delta = np.var(prob_delta_arr, axis =0)
        plt.figure(7)
        t = np.linspace(0,len(mean_delta),len(mean_delta))
        plt.plot(t, mean_delta, color = 'b')
        plt.fill_between(t, mean_delta-std_delta, mean_delta+std_delta, alpha=0.2, facecolor='b')
        plt.xlim([10, 400])
        plt.grid(True)
        plt.ylim([0,1.01])
        plt.show()
