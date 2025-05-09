import numpy as np
from utils import *

class HGV():
    def __init__(self):
        ''' Parameter for HGV model'''
        self.mass = 907  # Vehicle mass in kg
        self.S = 0.48  # Reference area in m^2

        ''' Parameter for HGV aerodynamics'''
        self.mach_numbers = np.array([3.5, 5, 8, 10, 15, 20, 23]) 
        self.aoa_values = np.array([10, 15, 20])
        self.cl_data = np.array([
            [0.4500, 0.4250, 0.4000, 0.3800, 0.3700, 0.3600, 0.3500],
            [0.7400, 0.7000, 0.6700, 0.6300, 0.6000, 0.5700, 0.5570],
            [1.0500, 1.0000, 0.9500, 0.9000, 0.8500, 0.8000, 0.7800]
        ]) 
        self.prev_sigma = 0.0

    def set_v1v2(self,v1,v2):
        self.V1 = v1
        self.V2 = v2

    def HGV_dynamics(self, t, x, sigma_c):
        r, lambda_, phi, v, theta, psi = x 
        
        altitude = r - earth_radius
        rho = get_density(altitude)
        mach = v / 343  

        alpha = self.calculate_aoa(v)
        cl, cd = self.get_cl_from_data(mach, alpha) 
        L = 0.5 * rho * v**2 * self.S * cl
        D = 0.5 * rho * v**2 * self.S * cd

        dr = v * np.sin(theta)
        dlambda = v * np.cos(theta) * np.sin(psi) / (r * np.cos(phi))
        dphi = v * np.cos(theta) * np.cos(psi) / r

        dv = -D / self.mass - g * np.sin(theta)
        dtheta = (L * np.cos(sigma_c)) / (v * self.mass) - (g * np.cos(theta)) / v + (v * np.cos(theta)) / r
        dpsi = (L * np.sin(sigma_c)) / (self.mass * v * np.cos(theta)) + v * np.cos(theta) * np.sin(psi) * np.tan(phi) / r

        return [dr, dlambda, dphi, dv, dtheta, dpsi]


    def get_cl_from_data(self, mach, aoa):
        interpolated_cl_values = np.array([
            np.interp(aoa, self.aoa_values, self.cl_data[:, i])
            for i in range(self.cl_data.shape[1])
        ])

        cl = np.interp(mach, self.mach_numbers, interpolated_cl_values)
        cd = self.aerodynamics_curve(cl)
        return cl, cd

    def aerodynamics_curve(self, C_L): 
        # Curve fitting function following the form C_D = C_D0 + K * C_L^2
        c_D0 = 0.08207728274395293
        k = 0.33301362495957676
        return c_D0 + k * C_L**2
    
    def calculate_aoa(self, v):
        alpha_max, alpha_K = 20, 10  # Maximum AOA and optimal L/D AOA
        if v >= self.V1:
            return alpha_max
        elif v <= self.V2:
            return alpha_K
        else:
            return ((alpha_K - alpha_max) / (self.V2 - self.V1)) * (v - self.V1) + alpha_max
    
    def PN_control(self, state, target):
        # Proportional Navigation control for bank angle adjustment
        r, lambda_, phi, v, theta, psi = state

        target_alt = target["altitude"] + earth_radius
        target_lon = np.radians(target["longitude"])
        target_lat = np.radians(target["latitude"])

        r_hgv = r*np.array([
            np.cos(lambda_) * np.cos(phi),
            np.sin(lambda_) *np.cos(phi),
            np.sin(phi),
        ])

        r_target = target_alt*np.array([
            np.cos(target_lon) * np.cos(target_lat),
            np.sin(target_lon) *np.cos(target_lat),
            np.sin(target_lat),
        ])

        los_vector = r_target-r_hgv
        los_distance =  np.linalg.norm(r_target-r_hgv)
        los_unit = los_vector / los_distance

        rel_vel = np.array([
            v * np.sin(theta),
            v * np.cos(theta) * np.sin(psi),
            v * np.cos(theta) * np.cos(psi)
        ])
        
        N = 0.1
        los_rate = np.cross(los_unit, rel_vel)/los_distance
        los_rate_magnitude = np.linalg.norm(los_rate[:2])
        commanded_acc = N*v*los_rate

        sigma = np.arctan2(commanded_acc[0], commanded_acc[1])

        alpha = 0.2 
        sigma = alpha * sigma + (1 - alpha) * self.prev_sigma
        sigma = np.clip(sigma, self.prev_sigma - np.radians(2), self.prev_sigma + np.radians(2))
        sigma = np.clip(sigma, -np.radians(45), np.radians(45))

        if np.sqrt((target_lon-lambda_)**2 +(target_lat-phi)**2 ) <= np.radians(1):
            sigma = 0.0
        elif los_rate_magnitude <= 1e-5:
            sigma = 0.0

        return sigma


