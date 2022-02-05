import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def q_11_14(x_y_z):
    scaled = np.dot(np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]]), x_y_z)
    orto = get_orthogonal_matrix(3)
    cov2 = np.cov(scaled)
    mul = np.dot(orto, scaled)
    cov = np.cov(mul)
    print(cov2)
    print("\n")
    print(cov)
    plot_3d(x_y_z)
    plot_3d(scaled)
    plot_3d(mul)
    plot_2d(x_y_z[0:2][:])
    con = ((x_y_z[2, :] > - 0.4) & (x_y_z[2, :] < 0.1))
    plot_2d(x_y_z[:, con])


def q_16_a():
    data = np.random.binomial(1,0.25, (100000,1000))
    x_m = np.zeros((5,1000))
    m = np.arange(1, 1001)
    plt.figure()
    plt.title("X_m vs m = Number of tosses")
    plt.ylabel("X_m")
    plt.xlabel("m = # Tosses")
    for i in range(0,5):
        x_m[i][:] = np.cumsum(data[i][:]) / m
        plt.plot(x_m[i][:], label='X_m ' + str(i))
        plt.legend()


def q_16_b_c(e, data, x_m):

    m = np.arange(1, 1001)
    for i in range(0, 100000):
        x_m[i][:] = np.abs(np.cumsum(data[i]) * (1 / m) - 0.25)
    percentage = np.zeros((1000,))
    for i in range(0, 1000):
        percentage[i] = np.sum(x_m[:, i] >= e).astype(np.int32) / 100000

    toses = np.arange(1, 1001)
    print(toses.shape)
    chebyshev = np.zeros((1000,))
    for i in range(0, 1000):
        chebyshev[i] = 1 / (4 * (i + 1) * e ** 2)
    hoeffding = 2 * np.exp(-2 * toses * e ** 2)
    chebyshev = chebyshev.clip(0, 1)
    hoeffding = hoeffding.clip(0, 1)
    fig = plt.figure()
    plt.title("Upper Bounds vs Number of tosses " + "e = " + str(e))
    plt.ylabel("Upper Bound")
    plt.xlabel("# Tosses")
    plt.plot(toses, chebyshev, label='Chebyshev')
    plt.legend()
    plt.plot(toses, hoeffding, label='Hoeffding')
    plt.legend()
    plt.plot(toses, percentage, label='percentage sequences')
    plt.legend()
    plt.savefig("abc.png")
    plt.show()


if __name__ == '__main__':
    q_11_14(x_y_z)
    q_16_a()
    data = np.random.binomial(1, 0.25, (100000, 1000))
    x_m = np.zeros((100000, 1000))
    for e in [0.5, 0.25, 0.1, 0.01, 0.001]:
        q_16_b_c(e, data, x_m)