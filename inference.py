import numpy as np
import jax
import jax.numpy as jnp

from scipy.stats import multivariate_normal
from utils import *
from scipy.stats import truncnorm
from scipy.integrate import nquad
from scipy.stats import norm
from scipy.spatial import ConvexHull

class Inference():

    def __init__(self, target, num_target, target_region, threat_val, state_dim):
        
        self.targets = target 
        self.num_target = num_target
        self.target_region = target_region
        self.threat_val = threat_val
        self.state_dim = state_dim
        self.reachable_region = [1]*num_target

        # Store posterior statistics for each target
        self.prob_state_mean = np.zeros((self.num_target, state_dim))
        self.prob_state_cov = np.zeros((self.num_target, state_dim ))
        self.prob_beta_mu = 1*np.ones((num_target))
        self.prob_beta_truncated_mu = 1*np.ones((num_target))
        self.prob_beta_sig = 1*np.ones((num_target))
        self.prob_delta = 1/num_target*np.ones((num_target))

        # Truncation bounds for beta prior
        self.a = 0.0
        self.b = np.inf 
        self.threshold = 1e-10 # To prevent numerical issues

    def compute_reachability(self, targets, traj, impact_lon_lat):
        attack_hits_ind = []
        for j in range(self.num_target+1):
            attack_hits_ind.append([])        

        target_coords = jnp.array([[t["longitude"], t["latitude"]] for t in targets])  # shape: (T, 2)

        #Check reachability and calculate probabilities
        reachable_region = [1] * len(targets)
        for target_idx, target in enumerate(targets):
            target_point = [target["longitude"], target["latitude"]]
            attack, polygon = self.ray_casting_algorithm(target_point, impact_lon_lat)
            if not attack:
                reachable_region[target_idx] = 0
            else:
                reachable_region[target_idx] = 1
        reachable_region = np.array(reachable_region)
        self.reachable_region = reachable_region

        dists = jnp.linalg.norm(
            impact_lon_lat[:, None, :] - target_coords[None, :, :], axis=-1
        ) 

        # Identify which samples are within target region
        target_hit_mask = dists <= self.target_region
        traj_indices = jnp.arange(traj.shape[0])[:, None]
        target_hit_traj_indices = [traj_indices[target_hit_mask[:, j]].flatten() for j in range(target_hit_mask.shape[1])]
        
        for j in range(len(target_hit_traj_indices)):
            selected_vals = traj[target_hit_traj_indices[j],0]

            if selected_vals.size > 1:
                self.prob_state_mean[j] = jnp.mean(selected_vals, axis = 0)
                self.prob_state_cov[j] = jnp.var(selected_vals, axis = 0)

        self.attack_hits_ind = target_hit_traj_indices
    
    def ray_casting_algorithm(self, target, reachable_states):
        lon_target, lat_target = target
        ray_count = 0

        reachable_states = np.array(reachable_states)

        hull = ConvexHull(reachable_states)
        polygon = reachable_states[hull.vertices].tolist()
        polygon.append(polygon[0]) 

        for i in range(len(polygon) - 1):
            p1 = polygon[i]
            p2 = polygon[i + 1]

            if ((p1[1] > lat_target) != (p2[1] > lat_target)) and \
            (lon_target < (p2[0] - p1[0]) * (lat_target - p1[1]) / (p2[1] - p1[1]) + p1[0]):
                ray_count += 1

        is_inside = (ray_count % 2 == 1)
        return is_inside, polygon

    def bayesian(self, measurement):
        prob_delta_arr = []
        prob_delta = []
        for i in range(self.num_target):
            current_beta_mu = self.prob_beta_mu[i]
            current_beta_sig = self.prob_beta_sig[i]
            
            current_state= self.prob_state_mean[i]
            current_state_cov = self.prob_state_cov[i]
            current_measurement = measurement

            if len(self.attack_hits_ind[i]) <= 1: # Skip targets with too few samples
                prob_delta_arr.append(0.0)
                continue
            
            state_sig = np.diag(current_state_cov)
            try:
                inv_state_sig = np.linalg.inv(state_sig)
            except np.linalg.LinAlgError:
                inv_state_sig = np.linalg.pinv(state_sig)
            state_ = (current_measurement-current_state)
            diff_state = np.matmul(state_.T,np.matmul(inv_state_sig,state_))

            A = -1/current_beta_sig
            B = current_beta_mu/current_beta_sig - self.threat_val[i]*diff_state
            C = 1/2

            next_beta_mu_ = (-B - np.sqrt(B**2 -4*A*C))/(2*A)
            next_beta_sig_ = 1/(1/current_beta_sig + 1/(2*next_beta_mu_**2))
            if next_beta_sig_ <= self.threshold:
                next_beta_sig_ = self.threshold

            # Truncated mean of updated beta
            a_ = (self.a - next_beta_mu_) / next_beta_sig_
            next_beta_mu = truncnorm.mean(a_ , self.b, loc=next_beta_mu_, scale=np.sqrt(next_beta_sig_))
            
            # Store updated beta values
            self.prob_beta_mu[i] = next_beta_mu_
            self.prob_beta_sig[i] = next_beta_sig_
            self.prob_beta_truncated_mu[i]= next_beta_mu

            # Calculate likelihood of measurement
            prob_ =  1
            for p in range(len(current_measurement)):
                prob_ = prob_*self.pdf(current_measurement[p], current_state[p], np.sqrt(current_state_cov[p]/(self.threat_val[i]*next_beta_mu)))

            prob_delta_arr.append(float(prob_))
            prob_delta.append(float(prob_))

        if len(prob_delta) >= 1:
            prob_delta_arr = np.array(prob_delta_arr)
            if sum(self.reachable_region == 1) > 2:
                for i in range(len(prob_delta_arr)):
                    if self.reachable_region[i] == 1 and prob_delta_arr[i] == 0.0:
                        prob_delta_arr[i] = min(prob_delta)
            
            prob_delta_arr_ = prob_delta_arr.copy()
            for i in range(len(prob_delta_arr)):       
                prob_delta_arr_[i] = prob_delta_arr_[i]*self.prob_delta[i]

            if sum(prob_delta_arr_) > 0:
                self.prob_delta =  0.5*self.prob_delta + 0.5*prob_delta_arr_/sum(prob_delta_arr_)

    def pdf(self, state, mean, std):
        z = abs(state - mean) / std
        normal_prob = 2 * norm.cdf(-z) 

        return normal_prob 