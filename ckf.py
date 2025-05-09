import numpy as np
from utils import *

class CKF():
    def __init__(self, state_dim, meas_dim, process_noise, meas_noise, P):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self.Q = process_noise
        self.R = meas_noise
        self.X = np.zeros(state_dim)
        self.P = P   

    def cubature_points(self, mean, covariance):
        """Generate cubature points and weights for CKF."""
        n = len(mean)
        points = []
        sqrt_cov = np.linalg.cholesky(covariance)
        for i in range(n):
            points.append(mean + np.sqrt(n) * sqrt_cov[:, i])
            points.append(mean - np.sqrt(n) * sqrt_cov[:, i])
        weights = [1.0 / (2 * n)] * (2 * n)
        return np.array(points), weights

    def predict(self, dt):
        """ CKF prediction step. """
        points, weights = self.cubature_points(self.X, self.P)
        propagated_points = np.array([self.HGV_dynamics_ckf(pt, dt) for pt in points])
        mean = np.sum([w * pt for w, pt in zip(weights, propagated_points)], axis=0)
        covariance = self.Q + np.sum([
            w * np.outer(pt - mean, pt - mean)
            for w, pt in zip(weights, propagated_points)
        ], axis=0)
        return mean, covariance

    def update(self, pred_mean, pred_cov, measurement):
        """ CKF update step. """
        points, weights = self.cubature_points(pred_mean, pred_cov)
        transformed_points = np.array([self.measurement_model(pt) for pt in points])
        meas_mean = np.sum([w * pt for w, pt in zip(weights, transformed_points)], axis=0)
        innovation_cov = self.R + np.sum([
            w * np.outer(pt - meas_mean, pt - meas_mean)
            for w, pt in zip(weights, transformed_points)
        ], axis=0)
        cross_cov = np.sum([
            w * np.outer(points[i] - pred_mean, transformed_points[i] - meas_mean)
            for i, w in enumerate(weights)
        ], axis=0)
        kalman_gain = cross_cov @ np.linalg.inv(innovation_cov)
        updated_state = pred_mean + kalman_gain @ (measurement - meas_mean)
        updated_covariance = pred_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        return updated_state, updated_covariance    

    def measurement_model(self, x):
        r, lambda_, phi, v, theta, psi, _, _, _ = x
        return np.array([r, lambda_, phi, v, theta, psi])

    def HGV_dynamics_ckf(self, x, dt):
        r, lambda_, phi, v, theta, psi, u_i, u_j, u_k = x
        altitude = r - earth_radius
        rho = get_density(altitude)

        dr = v * np.sin(theta)
        dlambda = v * np.cos(theta) * np.sin(psi) / (r * np.cos(phi))
        dphi = v * np.cos(theta) * np.cos(psi) / r

        dv = -0.5*rho*v**2*u_i - g * np.sin(theta)
        dtheta = (0.5*rho*v*u_j * np.cos(u_k)) - (g * np.cos(theta)) / v + (v * np.cos(theta)) / r
        dpsi = (0.5*rho*v*u_j * np.sin(u_k)) / (np.cos(theta)) + v * np.cos(theta) * np.sin(psi) * np.tan(phi) / r

        # unknown paramter dynamics 
        du_i = 0 #np.random.normal(0, 1e-4)
        du_j = 0 #np.random.normal(0, 1e-4)
        du_k = 0 #np.random.normal(0, 0.1)

        dx = np.array([dr, dlambda, dphi, dv, dtheta, dpsi, du_i, du_j, du_k])

        return x + dx*dt
    
