import numpy as np

earth_radius = 6371e3  # Earth's radius in m
g = 9.81  # Gravitational acceleration in m/s^2

def get_density(altitude):
    return 1.225 * np.exp(-altitude / 10000)


def stop_at_ground(t, x, current_target):
    r, _, _, _, _, _ = x  
    altitude = r - earth_radius

    # Stop when either altitude is zero or position error is below a threshold
    if altitude <= 0: 
        return 0
    return 1  # Continue otherwise
