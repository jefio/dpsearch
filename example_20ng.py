"""
Example: clustering the 20 newsgroups text dataset.
"""
import argparse
from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dpsearch import dpsearch


def get_dataset(keep_words):
    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories)

    # filter terms
    tvec = TfidfVectorizer(
        max_df=0.5, max_features=keep_words, stop_words='english')
    X_tfidf = tvec.fit_transform(dataset.data).toarray()
    terms = tvec.get_feature_names()

    # DPSearch needs integer data
    cvec = CountVectorizer(vocabulary=terms, dtype=int)
    X_count = cvec.fit_transform(dataset.data).toarray()
    return {
        'X_tfidf': X_tfidf,
        'X_count': X_count,
        'y': dataset.target,
        'target_names': dataset.target_names,
        'terms': terms
    }


def plot_clustering(dataset, pred_clusters, exp_name):
    pxt_mat = get_pxt_mat(dataset['y'], pred_clusters)
    df = pd.DataFrame(pxt_mat, columns=dataset['target_names'])
    df.plot.bar(stacked=True)
    plt.xlabel('Cluster ID')
    plt.ylabel('Nb of samples')
    plt.title('Clusters composition')
    filename = "exp_{}_composition.png".format(exp_name)
    plt.savefig(filename)
    plt.close()


def get_pxt_mat(y, pred_clusters):
    n_classes = len(set(y))
    n_pred_clusters = len(set(pred_clusters))
    pxt_mat = np.zeros((n_pred_clusters, n_classes), int)
    for cdx in range(n_pred_clusters):
        idxs, = np.where(pred_clusters == cdx)
        class_counter = Counter(y[idxs])
        pxt_mat[cdx] = [class_counter[c] for c in range(n_classes)]

    # plot clusters ordered by size
    cdxs = np.argsort(pxt_mat.sum(axis=1))[::-1]
    pxt_mat = pxt_mat[cdxs]

    return pxt_mat


def write_clusters_top_words(dataset, pred_clusters, exp_name):
    n_pred_clusters = len(set(pred_clusters))
    filename = "exp_{}_words.csv".format(exp_name)
    with open(filename, 'w') as fwrite:
        for cdx in range(n_pred_clusters):
            idxs, = np.where(pred_clusters == cdx)
            xis = dataset['X_tfidf'][idxs]
            fdxs = np.argsort(xis.max(axis=0))[-20:]
            top_terms = [dataset['terms'][fdx] for fdx in fdxs]
            line = ','.join(top_terms)
            fwrite.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    # DP options
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='Concentration parameter of the DP')
    parser.add_argument('-g', '--g0-alpha', type=float, default=10,
                        help='Concentration parameter of G0')
    parser.add_argument('-b', '--beam-size', type=int, default=100)
    # pre-processing options
    parser.add_argument('-k', '--keep-words', type=int, default=1000)
    args = parser.parse_args()

    dataset = get_dataset(args.keep_words)
    s = dpsearch(dataset['X'], args.alpha, args.g0_alpha, args.beam_size)
    print(s)


if __name__ == '__main__':
    main()
