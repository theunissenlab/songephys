from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.lda import LDA


def _reassign(Clusterer, labels, data, i, j):
    selector = (labels == i) | (labels == j)
    if data.ndim == 1:
        data_in = data[selector, None]
        data = data[:, None]
    else:
        data_in = data[selector]

    if Clusterer is KMeans:
        new_labels = Clusterer(n_clusters=2).fit(data_in).predict(data)
    else:
        new_labels = Clusterer(n_components=2).fit(data_in, labels[selector]).predict(data)
    labels[selector & (new_labels == 0)] = i
    labels[selector & (new_labels == 1)] = j
    return labels


reassign_fns = {
        "gmm": lambda labels, data, i, j: _reassign(GaussianMixture, labels, data, i, j),
        "kmeans": lambda labels, data, i, j: _reassign(KMeans, labels, data, i, j),
        "lda": lambda labels, data, i, j: _reassign(LDA, labels, data, i, j)
}


cluster_fns = {
    "gmm": lambda data, n: GaussianMixture(n_components=n).fit(data).predict(data),
    "kmeans": lambda data, n: KMeans(n_clusters=n, n_jobs=2).fit_predict(data)
}


def merge(labels, label1, label2):
    labels[labels == label2] = label1
    return labels


class Projector(object):

    def __init__(self):
        self.lda = LDA(n_components=2)
        
    def fit_transform(self, data, labels, i, j):
        selector = (labels == i) | (labels == j)
        self.projected = self.lda.fit_transform(data[selector], labels[selector])[:, 0]
        self.original_x = self.projected[labels[selector] == i]
        self.original_y = self.projected[labels[selector] == j]

    def project(self, data, ndim=1):
        if ndim == 1:
            return self.lda.transform(data)[:, 0]
        else:
            return self.lda.transform(data)

