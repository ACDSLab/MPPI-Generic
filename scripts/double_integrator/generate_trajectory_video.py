import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=7000000)

track_radius_outer = 2 + .125
track_radius_inner = 2 - .125

# Draw the bounds for the track
theta = np.linspace(0,2*np.pi, 1000)
x_track_inner = track_radius_inner*np.cos(theta)
y_track_inner = track_radius_inner*np.sin(theta)
x_track_outer = track_radius_outer*np.cos(theta)
y_track_outer = track_radius_outer*np.sin(theta)

fig, ax = plt.subplots(2,1)
fig.set_dpi(100)
fig.set_size_inches(10, 10)
xdata, ydata = [], []
xndata, yndata = [], []

ax[0].plot(x_track_inner, y_track_inner, 'r', linewidth=2)
ax[0].plot(x_track_outer, y_track_outer, 'r', linewidth=2)
ax[1].plot(x_track_inner, y_track_inner, 'r', linewidth=2)
ax[1].plot(x_track_outer, y_track_outer, 'r', linewidth=2)
ax[0].axis('equal')
ax[1].axis('equal')
ln1, = ax[0].plot([], [], 'go')
ln2, = ax[1].plot([], [], 'b*')

# Let us load the data first
actual_state = np.load('/home/mgandhi3/git/MPPI-Generic/build/release/examples/robust_rc_state_trajectory.npy')
nominal_trajectory = np.load('/home/mgandhi3/git/MPPI-Generic/build/release/examples/robust_sc_state_trajectory.npy')
nominal_state = nominal_trajectory[:,:]

def init():
    ax[0].set_xlim(-2.25, 2.25)
    ax[0].set_ylim(-2.25, 2.25)
    ax[1].set_xlim(-2.25, 2.25)
    ax[1].set_ylim(-2.25, 2.25)
    return ln1, ln2,

def update(frame):
    xdata.append(actual_state[frame, 0])
    ydata.append(actual_state[frame, 1])
    xndata.append(nominal_state[frame, 0])
    yndata.append(nominal_state[frame, 1])
    if (len(xdata) > 100):
        xdata.pop(0)
        ydata.pop(0)
        xndata.pop(0)
        yndata.pop(0)
    ln1.set_data(xdata, ydata)
    ln2.set_data(xndata, yndata)
    return ln1, ln2

ani = FuncAnimation(fig, update, frames=np.arange(0,500,1),
                    init_func=init, blit=True, interval=1, repeat=False)
ani.save('lines.mp4', writer=writer)
plt.show()

# We want to have a scatterplot that shows the current real state, the current nominal state. Then a line plot that shows the
# nominal trajectory