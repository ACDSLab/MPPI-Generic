import numpy as np
import matplotlib.pyplot as plt
import argparse

track_radius_outer = 2 + .125
track_radius_inner = 2 - .125

# Draw the bounds for the track
theta = np.linspace(0,2*np.pi, 1000)
x_track_inner = track_radius_inner*np.cos(theta)
y_track_inner = track_radius_inner*np.sin(theta)
x_track_outer = track_radius_outer*np.cos(theta)
y_track_outer = track_radius_outer*np.sin(theta)


def plot_trajectories(axis, axis_name, trajectory_array):
    axis.set_title(axis_name)
    time_horizon, trajectory_length, state_dim = trajectory_array.shape
    i = 0
    for i in range(time_horizon):
        if (trajectory_array[i,0,:] == 0).all():  # Means that the system stopped saving trajectories because of task failure
            break
        axis.plot(trajectory_array[i,:,0], trajectory_array[i,:,1], 'b', linewidth=0.5)
    axis.plot(trajectory_array[:i,0,0], trajectory_array[:i,0,1], 'y', linewidth=2)
    axis.plot(x_track_inner, y_track_inner, 'r', linewidth=2)
    axis.plot(x_track_outer, y_track_outer, 'r', linewidth=2)


def main(args):
    build_dir = args['build_dir']
    data_dir = build_dir + 'tests/controllers/'
    # Create the figures for each subplot
    fig, axs = plt.subplots(2,2)

    for ax in axs.flat:
        ax.set(xlabel='X Position (m)', ylabel='Y Position (m)')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # Plot the vanilla trajectory
    vanilla_nominal = np.load(data_dir + 'vanilla_nominal.npy')
    plot_trajectories(axs[0,0], 'Vanilla MPPI Nominal Disturbance', vanilla_nominal)

    # Vanilla MPPI with large noise
    vanilla_large = np.load(data_dir + 'vanilla_large.npy')
    plot_trajectories(axs[0,1], 'Vanilla MPPI Large Disturbance', vanilla_large)

    # Vanilla MPPI with large noise + a tracking controller
    vanilla_large_track = np.load(data_dir + 'vanilla_large_track.npy')
    plot_trajectories(axs[1,0], 'Vanilla MPPI Large Disturbance + Tracking', vanilla_large_track)

    # Tube MPPI with large noise
    tube_large = np.load(data_dir + 'tube_large_actual.npy')
    plot_trajectories(axs[1,1], 'Tube MPPI Large Disturbance', tube_large)

    # # Tube MPPI ancillary trajectories
    # tube_ancillary = np.load(data_dir + 'tube_ancillary.npy')
    # plot_trajectories(axs[1,1], 'Tube MPPI Ancillary Solution', tube_ancillary)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('--build_dir', help='Location of MPPI-Generic build folder', required=True)
    args = vars(parser.parse_args())

    main(args)
