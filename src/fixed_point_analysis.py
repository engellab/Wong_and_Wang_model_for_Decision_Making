import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from tqdm.auto import tqdm
from src.Model import DM_model
import scipy
from scipy.optimize import fsolve, minimize
from scipy.interpolate import interp1d

# defining the parameters
tau_s = 0.1
tau_AMPA = 0.002
a = 270
b = 108
d = 0.154
gamma = 0.641
sigma_noise = 0.02
J_E = 0.2609
J_I = -0.0497
J_ext = 0.0156
J = np.array([[J_E, J_I], [J_I, J_E]])
Ib = 0.3255 * np.ones((2, 1))

def in_the_list(x, x_list, tol=1e-3):
    for i in range(len(x_list)):
        diff = np.linalg.norm(x-x_list[i],2)
        if diff < tol:
            return True
    return False

def sort_eigs(E, R):
    # sort eigenvectors
    data = np.hstack([E.reshape(-1, 1), R.T])
    data = np.array(sorted(data, key=lambda l: np.real(l[0])))[::-1, :]
    E = data[:, 0]
    R = data[:, 1:].T
    return E, R

def sort_by_distance(points):
    aligned_point_inds = []
    not_aligned_point_inds = list(range(points.shape[0]))

    ind = np.where(points[:, 0] == np.min(points[:, 0]))[0][0]
    aligned_point_inds.append(ind)
    not_aligned_point_inds.remove(ind)
    # first find the leftmost point
    while len(not_aligned_point_inds) !=0:
        point = points[aligned_point_inds[-1], :]
        dists = []
        for k in not_aligned_point_inds:
            dist = np.linalg.norm(point - points[k, :])
            dists.append(dist)
        min_ind = not_aligned_point_inds[np.where(dists == np.min(dists))[0][0]]
        not_aligned_point_inds.remove(min_ind)
        aligned_point_inds.append(min_ind)
    return points[aligned_point_inds, :]


def fI(x):
    tmp = (a * x - b)
    return tmp / (1 - np.exp(-d * tmp))

def der_fI(x):  # correct
    tmp = (a * x - b)
    term1 = a / (1 - np.exp(-d * tmp))
    term2 = - a * tmp * d * np.exp(-d * tmp) / (1 - np.exp(-d * tmp)) ** 2
    return term1 + term2

def rhs(s):
    x = J @ s.reshape(-1, 1) + Ib + J_ext * np.ones((2, 1))
    rhs_vect = - s.reshape(-1, 1) / tau_s + gamma * (1 - s.reshape(-1, 1)) * fI(x)
    return rhs_vect.flatten()

def rhs_jac(s):
    x = J @ s.reshape(-1, 1) + Ib + J_ext * np.ones((2, 1))
    return -np.eye(s.shape[0]) / tau_s \
           - gamma * np.diag(fI(x).flatten()) \
           + gamma * (J * (1 - s) * der_fI(x).flatten()).T

def objective(s): # needed to finding the minimal points
    return (1.0 / 2) * np.sum(rhs(s) ** 2)

def objecive_grad(s):
    return (rhs(s).reshape(1, -1) @ rhs_jac(s)).flatten()

def find_unstable_fp(fun_tol=1e-6):
    unstable_fps = []
    while (len(unstable_fps) != 1):
        x0 = np.random.randn(2)
        x_root = fsolve(rhs, x0, args=())
        fun = objective(x_root)
        if fun < fun_tol and (not (-1 in np.sign(x_root))):
            J = rhs_jac(x_root)
            L = np.linalg.eigvals(J)
            if (np.max(np.real(L)) >= 0) and (not in_the_list(x_root, unstable_fps)):
                unstable_fps.append(x_root)
    unstable_fps = np.array(unstable_fps)
    return unstable_fps

def find_stable_fp(fun_tol=1e-6):
    stable_fps = []
    while (len(stable_fps) != 2):
        x0 = np.random.randn(2)
        x_root = fsolve(rhs, x0, args=())
        fun = objective(x_root)
        if fun < fun_tol and (not (-1 in np.sign(x_root))):
            J = rhs_jac(x_root)
            L = np.linalg.eigvals(J)
            if (np.max(np.real(L)) < 0) and (not in_the_list(x_root, stable_fps)):
                stable_fps.append(x_root)
    stable_fps = np.array(stable_fps)
    return stable_fps

def find_minimal_points(N_points, fun_tol=1e-6):
    minimal_points = []
    stable_fps = find_stable_fp(fun_tol)
    x0 = stable_fps[0]
    increment = (1 / (N_points - 1)) * (np.array(stable_fps[1]) - np.array(stable_fps[0]))
    # find minimal points, such that each point is the minimal point of the objective at a specific section along the
    # shortest path connecting the two stable fixed points corresponding to a decision
    for i in tqdm(range(N_points)):
        res = minimize(objective, x0, method='SLSQP',
                       jac=objecive_grad, options={'disp': False, 'maxiter': 1000},
                       constraints={'type': 'eq', 'fun': lambda x: np.dot(x - x0, increment)})
        x_root = res.x
        x0 = x_root + increment
        minimal_points.append(x_root)
    minimal_points = np.array(minimal_points)
    return minimal_points

def find_unstable_manifold(dt, num_steps, N, fun_tol=1e-6):
    unstable_point = find_unstable_fp(fun_tol)[0]
    J = rhs_jac(unstable_point)
    E, R = np.linalg.eig(J)
    E, R = sort_eigs(E, R)
    v = R[:, 0] # top eigenvect
    x = copy(unstable_point)
    unstable_manifold = [copy(x)]
    for i in range(num_steps):
        if i == 0:
            x += dt * v
        else:
            x += dt * rhs(x)
        unstable_manifold.append(copy(x))
    unstable_manifold = unstable_manifold[::-1]
    x = copy(unstable_point)
    for i in range(num_steps):
        if i == 0:
            x += dt * (-v)
        else:
            x += dt * rhs(x)
        unstable_manifold.append(copy(x))
    unstable_manifold = np.array(unstable_manifold)
    # interpolate the manifold
    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(unstable_manifold, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, N)
    interpolator = scipy.interpolate.interp1d(distance, unstable_manifold, kind="quadratic", axis=0)
    unstable_manifold = interpolator(alpha)
    return unstable_manifold

def get_selection_vectors(points):
    drive_vects = []
    selection_vects = []
    max_eigs = []
    for i in range(points.shape[0]):
        point = points[i, :]
        J = rhs_jac(point)
        E, R = np.linalg.eig(J)
        E, R = sort_eigs(E, R)
        L = np.linalg.inv(R)
        max_eigs.append((np.max(np.real(E))))
        drive_vects.append(copy(R[:, 0]))
        selection_vects.append(copy(L[0, :]))
    drive_vects = np.array(drive_vects)
    selection_vects = np.array(selection_vects)
    max_eigs = np.array(max_eigs)
    return drive_vects, selection_vects, max_eigs

def nullcline_s1(s2, Num_trials=500, tol=1e-13, N_points = 100):

    def explicit_function(s1, s2):
        s1 = s1[0]
        x1 = (J[0, 0] * s1 + J[0, 1] * s2 + Ib[0, 0] + J_ext)
        return (-s1 / tau_s + gamma * (1 - s1) * fI(x1))**2

    nullcline = []
    for z in tqdm(s2):
        for i in range(Num_trials):
            y0 = np.random.randn()
            y_root = fsolve(explicit_function, y0, args=(z,))
            fun = explicit_function(y_root, z)
            el = np.array([y_root[0], z])
            if (not in_the_list(el, nullcline)) and (fun < tol):
                nullcline.append(el)
    nullcline = np.array(nullcline)
    nullcline = sort_by_distance(nullcline)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(nullcline, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, N_points)
    interpolator = scipy.interpolate.interp1d(distance, nullcline, kind="quadratic", axis=0)
    nullcline = interpolator(alpha)
    return nullcline

def nullcline_s2(s1, Num_trials=250, tol=1e-23, N_points = 100):

    def explicit_function(s2, s1):
        s2 = s2[0]
        x2 = J[1, 0] * s1 + J[1, 1] * s2 + Ib[1, 0] + J_ext
        return (-s2 / tau_s + gamma * (1 - s2) * fI(x2))**2

    nullcline = []
    for z in tqdm(s1):
        for i in range(Num_trials):
            y0 = np.random.randn()
            y_root = fsolve(explicit_function, y0, args=(z,))
            fun = explicit_function(y_root, z)
            el = np.array([z, y_root[0]])
            if (not in_the_list(el, nullcline)) and (fun < tol):
                nullcline.append(el)
    nullcline = np.array(nullcline)
    nullcline = sort_by_distance(nullcline)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(nullcline, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, N_points)
    interpolator = scipy.interpolate.interp1d(distance, nullcline, kind="quadratic", axis=0)
    nullcline = interpolator(alpha)
    return nullcline


if __name__ == '__main__':
    N_points = 31
    N_points_nullclines = 30
    N_points_unstable_manifold = 40
    N_points_meshgrid = 16

    stable_points = find_stable_fp(fun_tol=1e-6)
    unstable_points = find_unstable_fp(fun_tol=1e-6)
    minimal_points = find_minimal_points(N_points)

    unstable_manifold = find_unstable_manifold(dt=0.02, num_steps=100, N = N_points_unstable_manifold)

    z = np.linspace(0, 0.7, N_points_nullclines)
    s1_nullcline = nullcline_s1(z)
    s2_nullcline = nullcline_s2(z)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(s1_nullcline[:, 0], s1_nullcline[:, 1], color='m', label=r'$s_1$ nullcline')
    plt.plot(s2_nullcline[:, 0], s2_nullcline[:, 1], color='teal', label=r'$s_2$ nullcline')

    #stable and unstable manifolds of a saddle
    plt.plot(unstable_manifold[:, 0], unstable_manifold[:, 1], color = 'k', alpha = 0.5)
    drive_vects, selection_vects, max_eigs = get_selection_vectors(unstable_manifold)

    # checking how the drive vectos overlap with the directions
    diff = unstable_manifold[1:, :] - unstable_manifold[:-1, :]
    overlaps = []
    for i in range(diff.shape[0]):
        l1 = diff[i, :]
        l2 = drive_vects[i, :]
        overlaps.append(np.dot(l1, l2)/(np.linalg.norm(l1) * np.linalg.norm(l2)))

    plt.quiver(unstable_manifold[:, 0], unstable_manifold[:, 1], drive_vects[:, 0], drive_vects[:, 1], color='k', width = 0.003)
    plt.quiver(unstable_manifold[:, 0], unstable_manifold[:, 1], selection_vects[:, 0], selection_vects[:, 1], color='r', width = 0.003)
    plt.plot(np.linspace(0,0.7,10), np.linspace(0,0.7,10), color='k', alpha = 0.5)

    plt.plot(minimal_points[:, 0], minimal_points[:, 1], color='g', label='minimal points manifold')
    # drive_vects, selection_vects, max_eigs = get_selection_vectors(minimal_points)
    # ax.quiver(*(minimal_points[:, t] for t in range(2)),
    #           *(drive_vects[:, t] for t in range(2)),
    #           color='k', width=0.003)

    plt.scatter(stable_points[:, 0], stable_points[:, 1], color='b', marker='o', s=50, label='stable fp')
    plt.scatter(unstable_points[:, 0], unstable_points[:, 1], color='r', marker='o', s=50, label='unstable fp')
    plt.grid(True)
    plt.legend(fontsize=14)

    X, Y = np.meshgrid(np.linspace(0, 0.7, N_points_meshgrid), np.linspace(0, 0.7, N_points_meshgrid))
    U = []
    V = []
    for x, y in zip(X.flatten(), Y.flatten()):
        u, v = rhs(np.array([x, y]).reshape(-1, 1))
        U.append(u)
        V.append(v)
    U = np.array(U).reshape(*X.shape)
    V = np.array(V).reshape(*Y.shape)
    ax.quiver(X, Y, U, V, width=0.001)

    m = DM_model()
    m.s_init = unstable_points[0]
    m.s = unstable_points[0]
    T = 20
    m.sigma_noise = 0
    m.run(T, -0.5)
    s_array = m.get_history()
    plt.plot(s_array[0, :], s_array[1, :], color = 'orange')
    m.clear_history()
    m.run(T, 0.5)
    s_array = m.get_history()
    plt.plot(s_array[0, :], s_array[1, :], color = 'salmon')
    plt.show()

    fig2 = plt.figure(figsize = (12,6))
    plt.plot(np.arange(max_eigs.shape[0]), max_eigs, color = 'r', label = r'$\lambda_0$')
    plt.legend(fontsize = 14)
    plt.xlabel("Num of point", fontsize = 14)
    plt.grid(True)
    plt.show()

    fig3 = plt.figure(figsize=(12,6))
    plt.plot(overlaps, color = 'b')
    plt.legend(fontsize = 14)
    plt.xlabel("Num of point", fontsize = 14)
    plt.grid(True)
    plt.show()
