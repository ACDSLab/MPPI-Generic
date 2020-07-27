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

def main(args):
	build_dir = args['build_dir']
	data_dir = build_dir + 'examples/'
	fe = np.load(data_dir + "vanilla_free_energy.npy")
	plt.plot(fe)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Say hello')
	parser.add_argument('--build_dir', help='Location of MPPI-Generic build folder', required=True)
	args = vars(parser.parse_args())

	main(args)
