import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from . import fitting
from .geometry import rotation_matrix_from_axis_and_angle


def show_G_distribution(data):
    """Show the distribution of the G function."""
    Xs, t = fitting.preprocess_data(data)

    Theta, Phi = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50))
    G = []

    for i in range(len(Theta)):
        G.append([])
        for j in range(len(Theta[i])):
            w = fitting.direction(Theta[i][j], Phi[i][j])
            G[-1].append(fitting.G(w, Xs))

    plt.imshow(G, extent=[0, np.pi, 0, 2 * np.pi], origin="lower")
    plt.show()


# https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def show_fit(w_fit, C_fit, r_fit, Xs):
    """Plot the fitting given the fitted axis direction, the fitted
    center, the fitted radius and the data points.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    # Plot the data points

    ax.scatter([X[0] for X in Xs], [X[1] for X in Xs], [X[2] for X in Xs])

    # Get the transformation matrix

    theta = np.arccos(np.dot(w_fit, np.array([0, 0, 1])))
    phi = np.arctan2(w_fit[1], w_fit[0])

    M = np.dot(
        rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
        rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta),
    )

    # Plot the cylinder surface

    delta = np.linspace(-np.pi, np.pi, 20)
    z = np.linspace(-10, 10, 20)

    Delta, Z = np.meshgrid(delta, z)
    X = r_fit * np.cos(Delta)
    Y = r_fit * np.sin(Delta)

    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + C_fit

            X[i][j] = p[0]
            Y[i][j] = p[1]
            Z[i][j] = p[2]

    ax.plot_surface(X, Y, Z, alpha=0.2)

    # Plot the center and direction

    ax.quiver(
        C_fit[0],
        C_fit[1],
        C_fit[2],
        r_fit * w_fit[0],
        r_fit * w_fit[1],
        r_fit * w_fit[2],
        color="red",
    )

    # Set the axes equal
    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])

    plt.show()
