""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25_self(object):
    def __init__(self, b=0.75, k1=0.8):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.y = y
        print("self.y.shape: ", self.y.shape)
        # self.avdl = y.sum(1).mean()
        # print("self.avdl: ", self.avdl)
        # print("len of self.avdl: ", len(self.avdl))

    def transform(self, q, index_list):
        """ Calculate BM25 between query q and documents X """
        b, k1 = self.b, self.k1

        # apply CountVectorizer
        # X = super(TfidfVectorizer, self.vectorizer).transform(X)
        X = self.y[index_list]
        avdl = X.sum(1).mean()
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1



#------------ End of library impl. Followings are the example -----------------

# from sklearn.datasets import fetch_20newsgroups
#
# print(1)
# texts = fetch_20newsgroups(subset='train').data
# print(texts[0])
# print(2)
# bm25 = BM25_self()
# print(3)
# bm25.fit(texts[1:])
# print(bm25.transform(texts[0], [i for i in range(11313)]))
# # bm25.fit(texts[5:])
# print(4)
# print(bm25.transform(texts[1], [i for i in range(1, 11313)] + [0]))
# # bm25.fit(texts[7:])
# print(5)
# print(bm25.transform(texts[2], [i for i in range(11313)]))