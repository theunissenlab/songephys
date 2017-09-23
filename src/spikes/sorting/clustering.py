import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class KMeansClusterer(object):

    def __init__(self, pcs, n_clusters):
        self.fitted = False

        self.pca = PCA(n_components=pcs)
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, waveforms):
        """Fit the clusterer to a dataset of waveforms

        Params
        ======
        waveforms (np.ndarray; (n_datapoints, waveform_width)):
            Dataset of spike waveforms to run clustering on.
            Currently slices the 10th to 30th samples of each waveform
        """
        x = self.pca.fit_transform(waveforms[:, 10:30])  # TODO: maybe dont hard code the slice to use?
        self.model.fit(x)

        y = self.model.predict(x)

        self.bad_labels = set(np.unique(y))  # Set of labels which do not meet our criteria
        self.good_labels = set()

        self.fitted = True

    def add_good_label(self, x):
        self.bad_labels.remove(x)
        self.good_labels.add(x)

    def predict(self, waveforms):
        if not self.fitted:
            raise Exception("Model not fit yet; have you run fit() yet?")

        x = self.pca.transform(waveforms[:, 10:30])
        y = self.model.predict(x)

        return y, None

    def predict_good(self, waveforms):
        if not self.fitted:
            raise Exception("Model not fit yet; have you run fit() yet?")

        x = self.pca.transform(waveforms[:, 10:30])
        y = self.model.predict(x)

        for label in self.bad_labels:
            y[y == label] = -1

        return y, None
