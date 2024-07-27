import numpy as np
from math import exp

# Bayesian optimization, implemented as explained in
# Wang, J. (2023). An intuitive tutorial to Gaussian processes regression. Computing in Science & Engineering.

def next_point(X1, Y, bounds=None, prec=50, acq="UCB", sigma=0.2, tau=0.15, kappa=30, processes=1, scale_var=True):
    if isinstance(X1[0], float):
        d = 1
    else:
        d = len(X1[0])
    if bounds is None:
        bounds = np.hstack((np.zeros([d, 1]), np.ones([d, 1])))
    if len(np.shape(bounds)) == 1:
        if scale_var:
            scale = bounds[1] - bounds[0]
        else:
            scale = 1.
        delta = bounds[0]
    else:
        if scale_var:
            scale = bounds[:, 1] - bounds[:, 0]
        else:
            scale_var = np.ones(len(bounds))
        delta = bounds[:, 0]

    if isinstance(X1[0], float):
        delta = delta[0]

    X1 = (X1 - delta) / scale

    if d == 1:
        X2 = np.linspace(0, 1, prec)
    elif d < 4: # relatively low-dimensional
        X2 = np.meshgrid(*np.vstack([np.linspace(0, 1, prec)]*d))
        X2 = np.transpose([np.reshape(Xi, [product(np.shape(Xi)), 1]) for Xi in X2])[0]
    else:
        prec2 = min(20, int(8 * (1 + (processes-1)/10)**(1/d)))  # goes to 10 times faster (but max prec=20)
        # first do a rough search
        X2 = np.meshgrid(*np.vstack([np.linspace(0.5/prec2, 1-0.5/prec2, prec2)] * d))
        X2 = np.transpose([np.reshape(Xi, [product(np.shape(Xi)), 1]) for Xi in X2])[0]
        if processes > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        else:
            rank = 0

        X2 = X2[(rank * len(X2)) // processes:((rank + 1) * len(X2)) // processes]
        mu, var = GP_pred(X1, X2, Y, sigma, tau)
        nr_test = int(3 * (1 + (processes-1)/20)**(1/d))  # minimally 3
        best_is = np.argpartition(mu + np.sqrt(var) * kappa, -nr_test)[-nr_test:]  # get the nr_test best points
        best_X = list(X2[best_is])
        best_vals = list(mu[best_is] + np.sqrt(var[best_is]) * kappa)

        if processes > 1:
            if rank == 0:
                for i in range(1, processes):
                    best_X.extend(comm.recv(source=i))
                    best_vals.extend(comm.recv(source=i))
            else:
                comm.send(best_X, dest=0)
                comm.send(best_vals, dest=0)

            best_X = np.array(comm.bcast(best_X, root=0))
            best_vals = np.array(comm.bcast(best_vals, root=0))

            best_X = best_X[np.argpartition(best_vals, -nr_test)[-nr_test:]]  # get the nr_test best points

        best_value = -np.inf
        best_i = None

        for x in best_X:
            # detailed search around best 3 points
            X2 = np.meshgrid(*np.vstack([np.linspace(-0.5/prec2, 0.5/prec2, prec2)] * d))
            X2 = np.transpose([np.reshape(Xi, [product(np.shape(Xi)), 1]) for Xi in X2])[0] + x
            X2 = X2[(rank * len(X2)) // processes:((rank + 1) * len(X2)) // processes]  # split up on processes
            mu, var = GP_pred(X1, X2, Y, sigma, tau)
            i = np.argmax(mu + np.sqrt(var) * kappa)
            value = mu[i] + np.sqrt(var[i]) * kappa
            if value > best_value:
                best_value = value
                best_i = i

        best_res = X2[best_i] * scale + delta
        if processes > 1:  # get the bost one of all processes
            if rank == 0:
                for i in range(1, processes):
                    comp_res = comm.recv(source=i)
                    comp_val = comm.recv(source=i)
                    if comp_val > best_value:
                        best_value = comp_val
                        best_res = comp_res
            else:
                comm.send(X2[best_i] * scale + delta, dest=0)
                comm.send(best_value, dest=0)

            best_res = comm.bcast(best_res, root=0)

        return best_res

    mu, var = GP_pred(X1, X2, Y, sigma, tau)

    return acquisition_function(X2, mu, var, kappa, acq) * scale + delta


def find_zero(X, Y, sigma=0.05, tau=0.5, tol=0.01):
    avg_Y = sum(Y) / len(Y)
    Y = np.array(Y) - avg_Y
    scaleY = sum(abs(Y)) / len(Y)
    if scaleY == 0:
        scaleY = 1
    Y = Y / scaleY
    zero_value = -avg_Y / scaleY
    N = len(X)
    # M = len(X2)
    sigma2 = sigma**2
    K11 = K(X, X, tau, same=True)
    matrix = np.linalg.solve(K11 + sigma2 * np.eye(N), Y)

    lb = min(X)
    ub = max(X)
    val = (lb + ub) / 2
    pred = np.matmul(K([val], X, tau), matrix)
    while abs(pred - zero_value) > tol:
        if pred > zero_value:
            lb = val
        else:
            ub = val
        val = (lb + ub) / 2
        pred = np.matmul(K([val], X, tau), matrix)

    return val



def acquisition_function(X2, mu, var, kappa, acq="UCB"):
    """
    Returns the maximal value of the acquisition function
    Parameters
    ----------
    X2: values to look for
    mu: mean
    var: variance
    kappa:
    acq: type of acquisition function. Only Upper Confidence Bound is implemented

    Returns
    -------

    """
    if acq == "UCB":
        return X2[np.argmax(mu + np.sqrt(var) * kappa)]
    else:
        return ValueError("Only UCB is implemented in this version")


def GP_pred(X1, X2, Y, sigma=0.2, tau=0.15, only_mu=False, ground_level=None):
    if ground_level is None:
        avg_Y = sum(Y) / len(Y)
    else:
        avg_Y = ground_level
    Y = np.array(Y) - avg_Y
    scaleY = sum(abs(Y)) / len(Y)
    if scaleY == 0:
        scaleY = 1
    Y = Y / scaleY
    N = len(X1)
    M = len(X2)
    sigma2 = sigma**2
    K11 = K(X1, X1, tau, same=True)
    K21 = K(X2, X1, tau)
    # K22 = K(X2, X2, same=True)
    mu = np.matmul(K21, np.linalg.solve(K11 + sigma2 * np.eye(N), Y))
    matrix2 = np.linalg.solve(K11 + sigma2 * np.eye(N), np.transpose(K21))
    if not only_mu:
        var = (1 + sigma2) * np.ones(M) - np.array([np.inner(K21[i], matrix2[:, i]) for i in range(len(K21))])
    else:
        var = None
    mu = mu * scaleY + avg_Y
    Y = Y * scaleY + avg_Y
    return mu, var


def GP_max_mu(X1, Y, bounds, X2, fixed_d, prec=40, sigma=0.2, tau=0.2, ref=None):
    # print(X1)
    if ref is None:
        avg_Y = sum(Y) / len(Y)
    else:
        avg_Y = ref
    # avg_Y = 0.
    # avg_Y = 0
    Y = Y - avg_Y
    scaleY = sum(abs(Y)) / len(Y)
    if scaleY == 0:
        scaleY = 1
    Y = Y / scaleY
    N = len(X1)
    d = len(bounds)
    sigma2 = sigma**2
    K11 = K(X1, X1, tau, same=True)
    matrix1 = np.linalg.solve(K11 + sigma2 * np.eye(N), Y)

    go = True
    steps = (np.array(bounds)[:, 1] - np.array(bounds)[:, 0]) / prec
    test_ds = list(filter(lambda x: x not in fixed_d, [i for i in range(d + len(fixed_d))]))
    X3 = [x[0] for x in bounds]
    best_mu = -np.inf
    best_x = None
    j = 0

    xs = np.meshgrid(*np.vstack([np.linspace(0, 1, prec)] * d))
    xs = np.transpose([np.reshape(Xi, [product(np.shape(Xi)), 1]) for Xi in xs])[0]
    for X3 in xs:
        x = np.zeros(d + len(fixed_d))
        for i in range(len(fixed_d)):
            x[fixed_d[i]] = X2[i]
        for i in range(len(test_ds)):
            x[test_ds[i]] = X3[i]

        K21 = K([x], X1, tau)
        mu = np.matmul(K21, matrix1)[0]
        # print(mu)
        if mu > best_mu:
            best_mu = mu
            best_x = x
            # print(best_x, best_mu)
        # print(X3)
        # X3[j] += steps[j]
        # if X3[j] > bounds[j][1]:
        #     X3[j] = bounds[j][0]
        #     j += 1
        #     if j > len(test_ds):
        #         go = False
        #     else:
        #         X3[j] += steps[j]
    # while go:
    #     x = np.zeros(d + len(fixed_d))
    #     for i in range(len(fixed_d)):
    #         x[fixed_d[i]] = X2[i]
    #     for i in range(len(test_ds)):
    #         x[test_ds[i]] = X3[i]
    #
    #     K21 = K([x], X1, tau)
    #     mu = np.matmul(K21, matrix1)[0]
    #     # print(mu)
    #     if mu > best_mu:
    #         best_mu = mu
    #         best_x = x
    #         # print(best_x, best_mu)
    #     print(X3)
    #     X3[j] += steps[j]
    #     if X3[j] > bounds[j][1]:
    #         X3[j] = bounds[j][0]
    #         j += 1
    #         if j > len(test_ds):
    #             go = False
    #         else:
    #             X3[j] += steps[j]

    return best_mu * scaleY + avg_Y, best_x




def K(X1, X2, tau=1, same=False):
    tau2 = 2*tau**2
    size = [len(X1), len(X2)]
    K_r = np.zeros(size)
    if same:
        for i, x1 in enumerate(X1):
            K_r[i, i] = ki(x1, x1, tau2)
            for j, x2 in enumerate(X2[:i]):
                K_r[i, j] = ki(x1, x2, tau2)
                K_r[j, i] = K_r[i, j]
    else:
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K_r[i, j] = ki(x1, x2, tau2)

    return K_r


def ki(x1, x2, tau2):
    if isinstance(x1, float):
        return exp(-(x2 - x1)**2 / tau2)
    else:
        return exp(-sum((x2 - x1)**2) / tau2)


def product(X):
    p = 1
    for xi in X:
        p *= xi
    return p