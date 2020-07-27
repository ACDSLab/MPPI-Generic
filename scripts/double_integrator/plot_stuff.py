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

def plot_fe_vs_time(fe_array, crash_cost=1000, title=""):
	plt.figure()
	axis = plt.gca()
	axis.set_title(title)
	time = np.linspace(0,fe_array.shape[0]*0.02, fe_array.shape[0])
	plt.plot(time, crash_cost*np.ones_like(time), 'r', label='Crash Threshold')
	plt.plot(time, fe_array, 'b', label='Free Energy')


def plot_fe_vs_space(fe_array, state_array):
	pass

def plot_trajectory(trajectory, title=""):
	plt.figure()
	axis = plt.gca()
	axis.set_title(title)
	axis.plot(trajectory[:,0], trajectory[:,1], 'b', linewidth=2)
	axis.plot(x_track_inner, y_track_inner, 'r', linewidth=2)
	axis.plot(x_track_outer, y_track_outer, 'r', linewidth=2)

def plot_nominal_state_used_vs_time(ns_array, title=""):
	plt.figure()
	axis = plt.gca()
	axis.set_title(title)
	time = np.linspace(0,ns_array.shape[0]*0.02, ns_array.shape[0])
	plt.scatter(time, ns_array)


def main(args):
	build_dir = args['build_dir']
	data_dir = build_dir + 'examples/'

	vanilla_fe = np.load(data_dir + "vanilla_free_energy.npy")
	vanilla_state_trajectory = np.load(data_dir + "vanilla_state_trajectory.npy")

	vanilla_large_fe = np.load(data_dir + "vanilla_large_free_energy.npy")
	vanilla_large_state_trajectory = np.load(data_dir + "vanilla_large_state_trajectory.npy")

	tube_large_rfe = np.load(data_dir + "tube_real_free_energy.npy")
	tube_large_nfe = np.load(data_dir + "tube_nominal_free_energy.npy")
	tube_large_state_trajectory = np.load(data_dir + "tube_state_trajectory.npy")
	tube_large_nominal_state_used = np.load(data_dir + "tube_nominal_state_used.npy");

	robust_sc_rfe = np.load(data_dir + "robust_sc_real_free_energy.npy")
	robust_sc_nfe = np.load(data_dir + "robust_sc_nominal_free_energy.npy")
	robust_sc_state_trajectory = np.load(data_dir + "robust_sc_state_trajectory.npy")
	robust_sc_nominal_state_used = np.load(data_dir + "robust_sc_nominal_state_used.npy");


	robust_rc_rfe = np.load(data_dir + "robust_rc_real_free_energy.npy")
	robust_rc_nfe = np.load(data_dir + "robust_rc_nominal_free_energy.npy")
	robust_rc_state_trajectory = np.load(data_dir + "robust_rc_state_trajectory.npy")
	robust_rc_nominal_state_used = np.load(data_dir + "robust_rc_nominal_state_used.npy");


	plot_fe_vs_time(vanilla_fe, 1000,"Vanilla Real Free Energy")
	plot_trajectory(vanilla_state_trajectory, "Vanilla State Trajectory")

	plot_fe_vs_time(vanilla_large_fe, 1000,"Vanilla Large Real Free Energy")
	plot_trajectory(vanilla_large_state_trajectory, "Vanilla Large State Trajectory")

	plot_fe_vs_time(tube_large_rfe, 1000,"Tube Real Free Energy")
	plot_fe_vs_time(tube_large_nfe, 1000,"Tube Nominal Free Energy")
	plot_trajectory(tube_large_state_trajectory, "Tube State Trajectory")
	plot_nominal_state_used_vs_time(tube_large_nominal_state_used)

	plot_fe_vs_time(robust_sc_rfe, 1000, "Robust Standard Real Free Energy")
	plot_fe_vs_time(robust_sc_nfe, 1000, "Robust Standard Nominal Free Energy")
	plot_trajectory(robust_sc_state_trajectory, "Robust Standard State Trajectory")
	plot_nominal_state_used_vs_time(robust_sc_nominal_state_used)

	plot_fe_vs_time(robust_rc_rfe, 100, "Robust Robust Real Free Energy")
	plot_fe_vs_time(robust_rc_nfe, 100, "Robust Robust Nominal Free Energy")
	plot_trajectory(robust_rc_state_trajectory, "Robust Robust State Trajectory")
	plot_nominal_state_used_vs_time(robust_rc_nominal_state_used)

	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Say hello')
	parser.add_argument('--build_dir', help='Location of MPPI-Generic build folder', required=True)
	args = vars(parser.parse_args())

	main(args)
