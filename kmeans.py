# CS 434 Implementation 4 - K-means
# Jesse Chick, Jonah Siekmann, David Skeen

import sys
import numpy as np

def load_data(path, to_float32=False):
    # Loads data from the path provided as 8-bit pixel values.
    data = np.loadtxt(path, dtype=np.uint8, delimiter=',')
    # If desired, convert to [0, 1] - adds overhead, but makes SSEs easier to
    #   interpret, w/r/t the order of magnitude observed in the SSEs. In our
    #   report, all figures were generated via data of type np.float32, [0, 1].
    if to_float32:
        data = data.astype(np.float32)
        data /= 255
    return data

def kmeans_predict(centroids, x):
    # Calculates distances between x and each centroid with SSE.
    d = np.array([np.sum((x - centroid) ** 2) for centroid in centroids])
    # Return the predicted cluster and distance between x and the center of the
    #   cluster.
    pred_idx, pred_sse = np.argmin(d), np.amin(d)
    return pred_idx, pred_sse

# Implements the standard k-means algorithm.
def kmeans(data, k, tol=1e-4, max_iter=128):
    clusters, n_iter, ave_sses = None, 0, []
    prev_ave_sse = np.inf
    # Selects inital centroids randomly from provided data.
    centroids = np.array([
        data[i] for i in np.random.choice(np.arange(data.shape[0]), size=k)])
    # Performs indeterminate number of iterations; constrained by 'max_iter'
    #   parameter.
    while True:
        clusters = [[] for _ in range(k)]
        preds = [0 for _ in range(k)]
        sses = []
        # Partition data by nearest centroid.
        for x in data:
            # Repurpose prediction function to determine the nearest cluster to
            #   'x'.
            argmin, sse = kmeans_predict(centroids, x)
            clusters[argmin].append(x)
            preds[argmin] += 1
            sses.append(sse)
        # Update centroids.
        centroids = [np.average(cluster, axis=0) for cluster in clusters]
        ave_sse = np.average(sses)
        ave_sses.append(ave_sse)
        n_iter += 1
        print('Iter. {}:\t{:.6f}'.format(n_iter, ave_sse))
        # Ceases iteration when the centroids have sufficiently converged.
        if prev_ave_sse - ave_sse <= tol:
            break
        # Prevents infinite loop by enforcing an interation limit.
        if n_iter > max_iter:
            print('Max iter ({}) exceeded.'.format(max_iter))
            break
        prev_ave_sse = ave_sse
    print()
    return clusters, n_iter, ave_sses

# Helper for analysis for 2.2.
def kmeans_optimal(data, ks, n_init=10, tol=1e-4):
    min_sse = np.inf
    argmin_k = np.inf
    min_sses = []
    n_iters = []
    for k in ks:
        sses = []
        for _ in range(n_init):
            _, n_iter, sse = kmeans(data, k, tol=tol)
            sses.append(sse[-1])
            n_iters.append(n_iter)
        k_min_sse = np.amin(sses)
        min_sses.append(k_min_sse)
        if k_min_sse < min_sse:
            min_sse = k_min_sse
            argmin_k = k
            print('k={}, {:.6f}'.format(k, min_sse))
    n_iters = np.average(np.array(n_iters).reshape((len(ks), n_init)), axis=1)
    return argmin_k, n_iters, min_sses

def plot(x, y, x_label='', y_label=''):
    _, axes = plt.subplots()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label, color='tab:blue')
    axes.plot(x, y, color='tab:blue')
    axes.tick_params(axis='y', labelcolor='tab:blue')
    plt.show()

if __name__ == '__main__':

    submission = True

    k = int(sys.argv[1])
    print('''
        Performing single iteration of kmeans with k = {}
    '''.format(k))
    data = load_data('./data/p4-data.txt', to_float32=True)
    # Q 2.1
    _, n_iters, sses = kmeans(data, k, tol=1e-2)
    if not submission:
        plot(np.arange(n_iters), sses, x_label='Iters', y_label='SSE')
        # Q 2.2
        ks = [k_ for k_ in range(2, k + 1)]
        _, _, sses = kmeans_optimal(data, ks, n_init=1, tol=1e-2)
        if hasattr(sys, 'real_prefix'):
            plot(ks, sses, x_label='k', y_label='SSE')
