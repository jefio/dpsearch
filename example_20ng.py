"""
Example: clustering the 20 newsgroups text dataset.
"""
import argparse

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from dpsearch import dpsearch


def get_dataset(keep_words):
    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories)

    # filter terms
    tvec = TfidfVectorizer(
        max_df=0.5, max_features=keep_words, stop_words='english')
    tvec.fit(dataset.data)
    terms = tvec.get_feature_names()

    # DPSearch needs integer data
    cvec = CountVectorizer(vocabulary=terms, dtype=int)
    X = cvec.fit_transform(dataset.data).toarray()
    return {
        'X': X,
        'y': dataset.target,
        'terms': terms
    }


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
