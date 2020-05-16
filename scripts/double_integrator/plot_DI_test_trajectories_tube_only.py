import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvas
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)

track_radius_outer = 2 + .125
track_radius_inner = 2 - .125

# Draw the bounds for the track
theta = np.linspace(0,2*np.pi, 1000)
x_track_inner = track_radius_inner*np.cos(theta)
y_track_inner = track_radius_inner*np.sin(theta)
x_track_outer = track_radius_outer*np.cos(theta)
y_track_outer = track_radius_outer*np.sin(theta)

cur_timestep = 0
nominal = None
actual = None
axs = None

def callback(event):
    ''' this function gets called if we hit the left button'''
    #print('you pressed', event.key)
    plt.cla()
    plot_boundaries(axs)
    global cur_timestep
    if event.key == "right":
        if not (actual[cur_timestep,0,:] == 0).all():  # Means that the system stopped saving trajectories because of task failure
            cur_timestep += 1
    elif event.key == "left":
        if cur_timestep > 0:
            cur_timestep -= 1

    global actual, acillary, nominal
    global axs
    # plot actual trajectory
    axs.plot(actual[cur_timestep,:,0], actual[cur_timestep,:,1], 'b', linewidth=2.0, label='final trajectory')
    axs.plot(actual[:cur_timestep,0,0], actual[:cur_timestep,0,1], 'bx', label='actual state')

    axs.plot(ancillary[cur_timestep,:,0], ancillary[cur_timestep,:,1], 'k', linewidth=2.0, label='ancillary trajectory')

    # TODO nominal stae
    axs.plot(nominal[cur_timestep,:,0], nominal[cur_timestep,:,1], 'g', linewidth=2.0, label='nominal trajectory')
    axs.plot(nominal[:cur_timestep,0,0], nominal[:cur_timestep,0,1], 'g+', label='nominal state')
    # fix legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def plot_trajectories(axis, axis_name, trajectory_array):
    axis.set_title(axis_name)
    time_horizon, trajectory_length, state_dim = trajectory_array.shape
    i = 0
    for i in range(time_horizon):
        if (trajectory_array[i,0,:] == 0).all():  # Means that the system stopped saving trajectories because of task failure
            break
        axis.plot(trajectory_array[i,:,0], trajectory_array[i,:,1], 'b', linewidth=0.5)
    axis.plot(trajectory_array[:i,0,0], trajectory_array[:i,0,1], 'y', linewidth=2)

def plot_boundaries(axis):
    axis.plot(x_track_inner, y_track_inner, 'r', linewidth=2, label='boundary')
    axis.plot(x_track_outer, y_track_outer, 'r', linewidth=2)


def main(args):
    build_dir = args['build_dir']
    data_dir = build_dir + 'tests/controllers/'
    # Create the figures for each subplot
    global axs
    fig, axs = plt.subplots(1,1)

    # set up callback for arrow keys
    fig.canvas.mpl_connect('key_press_event', callback)


    axs.set(xlabel='X Position (m)', ylabel='Y Position (m)')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # Tube MPPI with large noise
    global actual, ancillary, nominal
    actual = np.load(data_dir + 'tube_large_actual.npy')
    nominal = np.load(data_dir + 'tube_large_nominal.npy')
    ancillary = np.load(data_dir + 'tube_ancillary.npy')
    plot_boundaries(axs)
    # TODO plot the initial location

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('--build_dir', help='Location of MPPI-Generic build folder', required=True)
    args = vars(parser.parse_args())

    main(args)
