"""
DPSearch clustering algorithm
"""
import argparse
from heapq import heappush, heappop
from itertools import count

import numpy as np
from scipy.special import gammaln


def data_likelihood(xi, alpha):
    """
    Parameters
    ----------
    xi : ndarray, shape (n_features,)
    alpha : float
    """
    n_features, = xi.shape
    return (gammaln(n_features * alpha) - n_features * gammaln(alpha) +
            np.sum(gammaln(alpha + xi)) - gammaln(np.sum(alpha + xi)) +
            gammaln(np.sum(xi) + 1) - np.sum(gammaln(xi + 1)))

def data_likelihood_update(X, alpha, c, phi, cluster_sizes):
    """
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    alpha : float
    c : ndarray
        c[i] = cluster of sample i.
    phi : ndarray
    cluster_sizes : ndarray
        cluster_sizes[k] = size of cluster k.
    """
    assert len(c) > 1
    idx = len(c) - 1
    cdx = c[idx]
    # new cluster
    if cluster_sizes[cdx] == 1:
        dll = data_likelihood(X[idx], alpha)
    # existing cluster
    else:
        xm = X[idx]
        sx = phi[cdx]
        dll = (gammaln(np.sum(xm) + 1) - np.sum(gammaln(xm + 1)) +
               np.sum(gammaln(alpha + sx)) - gammaln(np.sum(alpha + sx)) +
               gammaln(np.sum(alpha + sx - xm)) - np.sum(gammaln(alpha + sx - xm)))
    return dll


def order_by_marginal(X, marginals):
    """
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    marginals : ndarray
        marginals[i] = log marginal posterior of sample i.
    """
    idxs = np.argsort(marginals)
    return X[idxs], marginals[idxs]


def log_DP_prior_count_complete2(N0, alpha, N, size_counts, cache={}):
    """
    Parameters
    ----------
    N0 : int
        Number of samples already clustered.
    alpha : float
    N : int
        Number of samples.
    size_counts : ndarray
        size_counts[s] = number of clusters of size s.
    """
    h = hash_(size_counts, N)
    if h not in cache:
        cache[h] = compute_it(size_counts, N0, N, alpha)
    return cache[h]


def compute_it(size_counts, N0, N, alpha):
    """
    Parameters
    ----------
    size_counts : ndarray
        size_counts[s] = number of clusters of size s.
    N0 : int
        Number of samples already clustered.
    N : int
        Number of samples.
    alpha : float
    """
    size_counts = np.array(size_counts)

    dd1 = np.arange(1., N) / np.arange(2., N + 1)
    logs = np.log(np.arange(1, N + 1))

    finish_up = -1
    last_n = -1
    if size_counts[-1] != 0:
        size_counts = np.concatenate((size_counts, [0]))
    lsc = len(size_counts)
    for n in range(N0 + 1, N + 1):
        scores = (dd1[:lsc-1] *
                  size_counts[:lsc-1] /
                  (size_counts[1:lsc] + 1))
        idx = np.argmax(scores)
        val = scores[idx]
        if val < alpha / (size_counts[0] + 1):
            idx = -1

        size_counts[idx + 1] += 1
        if idx > -1:
            size_counts[idx] -= 1

        if lsc == idx + 2:
            if len(size_counts) < idx + 3:
                diff = idx + 3 - len(size_counts)
                size_counts = np.concatenate((size_counts, [0 for _ in range(diff)]))
            lsc = idx + 3
        jdx = idx + 1
        v = ((jdx / (jdx + 1.)) *
             (size_counts[jdx] / (size_counts[jdx + 1] + 1.)))
        valid = (jdx > 0 and
                 v > alpha / (size_counts[0] + 1.) and
                 size_counts[jdx - 1] == 0 and
                 np.all(scores < v))
        if valid:
            if np.all(size_counts[jdx + 1:] == 0):
                tmp = (dd1[jdx-1:lsc] * size_counts[jdx-1:lsc] /
                       (size_counts[jdx:lsc+1] + 1))
                if np.all(tmp <= v):
                    finish_up = jdx
                    last_n = n
                    break

    if finish_up > -1:
        size_counts[finish_up] = 0
        size_counts[finish_up + N - last_n] = 1

    idxs, = np.where(size_counts > 0)
    return (np.sum(size_counts[idxs]) * np.log(alpha) -
            np.sum(size_counts[idxs] * logs[idxs]) -
            np.sum(gammaln(size_counts[idxs] + 1)))


def hash_(size_counts, N):
    """
    Parameters
    ----------
    size_counts : ndarray
        size_counts[s] = number of clusters of size s.
    N : int
        Number of samples.
    """
    logs = np.log(7 + 3 * np.arange(1, N + 1))
    return int(np.floor(np.sum(size_counts * logs[:len(size_counts)])) % 97) + 1


def compute_heur_inad(s, marginals):
    """
    Parameters
    ----------
    s : dict
    marginals : ndarray
        marginals[i] = log marginal posterior of sample i.
    """
    N0 = len(s['c'])
    return - marginals[N0]


def compute_g_score(s, X, alpha, G0_alpha):
    """
    Parameters
    ----------
    s : dict
    X : ndarray, shape (n_samples, n_features)
    alpha : float
        Concentration parameter of the DP
    G0_alpha : float
        Concentration parameter of the base distribution
    """
    N0 = len(s['c'])
    if N0 == 1:
        dl = data_likelihood(X[0], G0_alpha)
    else:
        dl = s['parent']['dl'] + data_likelihood_update(
            X, G0_alpha, s['c'], s['phi'], s['cluster_sizes'])
    score = - dl - log_DP_prior_count_complete2(
        N0, alpha, len(X), s['size_counts'])
    return {'dl': dl, 'g_score': score}


def dpsearch(X, alpha, G0_alpha, beam_size):
    """
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    alpha : float
        Concentration parameter of the DP
    G0_alpha : float
        Concentration parameter of the base distribution
    beam_size : int
    """
    n_samples, n_features = X.shape
    marginals = np.array([data_likelihood(xi, G0_alpha) for xi in X])
    X, marginals = order_by_marginal(X, marginals)
    for idx in range(n_samples):
        marginals[idx] = np.sum(marginals[idx:])
    marginals = np.concatenate((marginals, [0]))

    s0 = {
        'c': np.array([0]),
        'size_counts': np.array([1]),
        'nb_clusters': 1,
        'phi': np.array(X[0]).reshape(1, len(X[0])),
        'cluster_sizes': np.array([1])
    }
    s0.update(compute_g_score(s0, X, alpha, G0_alpha))
    s0['h_score'] = compute_heur_inad(s0, marginals)
    s0['score'] = s0['g_score'] + s0['h_score']
    heap = []
    tiebreaker = count()
    heappush(heap, (s0['score'], next(tiebreaker), s0))
    while True:
        _, _, s = heappop(heap)
        N0 = len(s['c'])

        if N0 == n_samples:
            break

        # expand by existing cluster
        for k in range(s['nb_clusters']):
            size_ = sum(1 for c in s['c'] if c == k)
            size_counts = np.array(s['size_counts'])
            size_counts[size_ - 1] -= 1
            if size_ == len(size_counts):
                size_counts = np.concatenate((size_counts, [0]))
            size_counts[size_] += 1

            phi = np.array(s['phi'])
            if k >= len(phi):
                phi = np.vstack(phi, phi[0] * 0)

            phi[k] += X[N0]

            cluster_sizes = np.array(s['cluster_sizes'])
            cluster_sizes[k] += 1

            s1 = {
                'c': np.concatenate((s['c'], [k])),
                'size_counts': size_counts,
                'nb_clusters': s['nb_clusters'],
                'phi': phi,
                'cluster_sizes': cluster_sizes,
                'parent': s
            }
            s1.update(compute_g_score(s1, X, alpha, G0_alpha))
            s1['h_score'] = compute_heur_inad(s1, marginals)
            s1['score'] = s1['g_score'] + s1['h_score']
            s1['parent'] = None
            heappush(heap, (s1['score'], next(tiebreaker), s1))

        # expand by new cluster
        phi = np.array(s['phi'])
        if s['nb_clusters'] >= len(phi):
            phi = np.vstack((phi, phi[0] * 0))
        phi[s['nb_clusters']] += X[N0]
        cluster_sizes = np.concatenate((s['cluster_sizes'], [1]))
        size_counts = np.array(s['size_counts'])
        size_counts[0] += 1
        s1 = {
            'c': np.concatenate((s['c'], [s['nb_clusters']])),
            'size_counts': size_counts,
            'nb_clusters': s['nb_clusters'] + 1,
            'phi': phi,
            'cluster_sizes': cluster_sizes,
            'parent': s
        }
        s1.update(compute_g_score(s1, X, alpha, G0_alpha))
        s1['h_score'] = compute_heur_inad(s1, marginals)
        s1['score'] = s1['g_score'] + s1['h_score']
        s1['parent'] = None
        heappush(heap, (s1['score'], next(tiebreaker), s1))

        if len(heap) > beam_size:
            heap = heap[:beam_size]

    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='Concentration parameter of the DP')
    parser.add_argument('-g', '--g0-alpha', type=float, default=10,
                        help='Concentration parameter of G0')
    parser.add_argument('-b', '--beam-size', type=int, default=100)
    args = parser.parse_args()

    X = np.loadtxt(args.input_file, dtype=int)
    s = dpsearch(X, args.alpha, args.g0_alpha, args.beam_size)
    print(s)


if __name__ == '__main__':
    main()
