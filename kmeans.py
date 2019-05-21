

import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

def load_data(path, to_float32=True):
    data = np.loadtxt(path, dtype=np.uint8, delimiter=',')
    if to_float32:
        data = data.astype(np.float32)
        data /= 255
    return data

def sse(x_1, x_2):
    return np.sum((x_2 - x_1) ** 2)

def kmeans_predict(centroids, x):
    d = np.array([sse(x, centroid) for centroid in centroids])
    return np.argmin(d), np.amin(d)

def _kmeans_update_centroids(centroids, clusters):
    for i, cluster in enumerate(clusters):
        centroids[i] = np.average(cluster, axis=0)

def kmeans(data, k, tol=1e-4, max_iter=128):
    centroids = np.array([data[i] for i in np.random.choice(np.arange(data.shape[0]), size=k)])
    clusters = None
    prev_ave_sse = np.inf
    n_iter = 0
    ave_sses = []
    while True:
        clusters = [[] for _ in range(k)]
        preds = [0 for _ in range(k)]
        sses = []
        for x in data:
            argmin, sse_ = kmeans_predict(centroids, x)
            clusters[argmin].append(x)
            preds[argmin] += 1
            sses.append(sse_)
        _kmeans_update_centroids(centroids, clusters)
        n_iter += 1
        ave_sse = np.average(sses)
        ave_sses.append(ave_sse)
        print('{}: {:.6f}'.format(n_iter, ave_sse))
        if prev_ave_sse - ave_sse <= tol:
            break
        if n_iter > max_iter:
            print('Max iter ({}) exceeded.'.format(max_iter))
            break
        prev_ave_sse = ave_sse
    return clusters, n_iter, ave_sses

def kmeans_optimal(data, ks, n_init=10, tol=1e-4):
    min_sse = np.inf
    argmin_k = np.inf
    min_sses = []
    n_iters = []
    for k in ks:
        sses = []
        for _ in range(n_init):
            _, n_iter, sse_ = kmeans(data, k, tol=tol)
            sses.append(sse_[-1])
            n_iters.append(n_iter)
        k_min_sse = np.amin(sses)
        min_sses.append(k_min_sse)
        if k_min_sse < min_sse:
            min_sse = k_min_sse
            argmin_k = k
            print('k={}, {:.6f}'.format(k, min_sse))
    n_iters = np.average(np.array(n_iters).reshape((len(ks), n_init)), axis=1)
    # min_sses = np.array(min_sses)
    return argmin_k, n_iters, min_sses

def plot(x, y, x_label='', y_label=''):
    fig, left_axis = plt.subplots()
    left_axis.set_xlabel(x_label)
    left_axis.set_ylabel(y_label, color='tab:blue')
    left_axis.plot(x, y, color='tab:blue')
    left_axis.tick_params(axis='y', labelcolor='tab:blue')
    plt.show()

if __name__ == '__main__':

    k = int(sys.argv[1])

    data = load_data('./data/p4-data.txt')
    np.random.shuffle(data)

    # Q 2.1

    _, n_iters, sses = kmeans(data, k, tol=1e-2)
    plot(np.arange(n_iters), sses, x_label='Iters', y_label='SSE')

    # Notes on 2.1:
    # Figure is not smoothed

    # Q 2.2

    ks = [k_ for k_ in range(2, k + 1)]
    _, _, sses = kmeans_optimal(data, ks, n_init=1, tol=1e-2)
    plot(ks, sses, x_label='k', y_label='SSE')

    # Notes on 2.2:
    # *** By default, k={2,...,<command line k>} will be evaluated when
    #   kmeans.py is run
    # Figure is smoothed
    # Figure is minimum average SSE over 10 trials for each k={2,...,15}
