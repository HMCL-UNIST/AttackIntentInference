import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_target_2d(targets, region, current_target_idx, ax):
    for ind, target in enumerate(targets):
        u = np.linspace(0, 2 * np.pi, 100)
        x = region * np.cos(u) + target["longitude"]
        y = region * np.sin(u) + target["latitude"]

        if ind == current_target_idx:
            ax.plot(x, y, color='r', alpha=0.5) 
            ax.fill(x, y, color='r', alpha=0.2)
        else:
            ax.plot(x, y, color='k', alpha=0.5)  
            ax.fill(x, y, color='k', alpha=0.2)
            
def plot_target_3d(targets, region, current_target_idx, ax):    
    for ind, target in enumerate(targets):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = region * np.outer(np.cos(u), np.sin(v)) + target["longitude"]
        y = region * np.outer(np.sin(u), np.sin(v)) + target["latitude"]
        z = region * np.outer(np.ones(np.size(u)), np.cos(v)) + target["altitude"]
        if ind == current_target_idx:
            ax.plot_surface(x, y, z, color='r', alpha=0.5, edgecolor='none')
        else:
            ax.plot_surface(x, y, z, color='k', alpha=0.2, edgecolor='none')


def plot_trajectory_pro(true_states, estimated_state, targets, current_target_idx, region, figure_num= 0, return_fig = False):    
    fig = plt.figure(figsize=(5, 5), num = figure_num)
    plt.clf()

    lon = np.degrees( true_states[:,1] )
    lat = np.degrees( true_states[:,2] )
    alt = (true_states[:,0] - earth_radius)/1000

    lon_est = np.degrees( estimated_state[1] )
    lat_est = np.degrees( estimated_state[2] )
    alt_est = (estimated_state[0] - earth_radius)/1000

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(lon, lat, alt, 'k--')
    ax.plot(lon_est, lat_est, alt_est, 'bo', linewidth = 10)

    for ind, target in enumerate(targets):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = region * np.outer(np.cos(u), np.sin(v)) + target["longitude"]
        y = region * np.outer(np.sin(u), np.sin(v)) + target["latitude"]
        z = region * np.outer(np.ones(np.size(u)), np.cos(v)) + target["altitude"]

        if ind == current_target_idx:
            ax.plot_surface(x, y, z, color='r', alpha=0.5, edgecolor='none')
        else:
            ax.plot_surface(x, y, z, color='k', alpha=0.2, edgecolor='none')
    
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('Altitude [km]')
    ax.relim()
    ax.autoscale_view()

    if return_fig:
        return fig, ax 



