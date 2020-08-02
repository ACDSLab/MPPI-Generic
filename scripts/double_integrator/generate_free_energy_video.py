import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from matplotlib import rc
import argparse

controller_dict = {'v': 'vanilla_large_',
                   't': 'tube_',
                   'rs': 'robust_sc_',
                   'rr': 'robust_rc_'}

title_dict = {'v': 'MPPI Standard Cost',
                   't': 'Tube-MPPI Standard Cost',
                   'rs': 'RMPPI Standard Cost',
                   'rr': 'RMPPI Robust Cost'}

rc('font', **{'size': 16})
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Manan Gandhi'), bitrate=-1)

fig, ax = plt.subplots()
fig.set_dpi(100)
fig.set_size_inches(10, 10)
xdata, ydata = [], []

ax.set_ylabel('Free Energy')
ax.set_xlabel('Time (sec)')
title = None
ln2, = ax.plot([], [], 'b', alpha=0.7, label='Real')
ln1, = ax.plot(np.linspace(0,100,100), 1*np.ones(100), 'r', alpha=0.7, label='Crash')


def init():
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.1)
    return ln1, ln2

def update(frame):
    xdata.append(frame*0.02)
    ydata.append(real_fe[frame])
    ln2.set_data(xdata, ydata)
    if len(xdata) > 5/0.02:
        xdata.pop(0)
        ydata.pop(0)
        ax.set_xlim(xdata[0], xdata[-1])

    return ln1, ln2

def main(args):
    build_dir = args['build_dir']
    data_dir = build_dir + 'examples/'
    controller_name = controller_dict[args['controller']]

    # Let us load the data first
    global real_fe, nominal_fe

    # Load the free energy data
    if not args['controller'] == 'rr':
        real_fe = np.load(data_dir + controller_name + 'real_free_energy.npy')/1000
    else:
        real_fe = np.load(data_dir + controller_name + 'real_free_energy.npy')/100
    ax.set_title(title_dict[args['controller']])

    ani = FuncAnimation(fig, update, frames=np.arange(0,5000,1),
                    init_func=init, blit=False, interval=1, repeat=False)
    # ani.save(title_dict[args['controller']] + '.mp4', writer=writer)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('--build_dir', help='Location of MPPI-Generic build folder', required=True)
    parser.add_argument('--controller', help="Which controller we are plotting", required=True)
    args = vars(parser.parse_args())

    main(args)
