import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

from preprocess import collect_waveforms


class GaussianMixtureClustererPruned(object):
    """Spike clustering using Bayesian Gaussian Mixture model

    Excludes clusters with low weights or if they have too few members
    """

    def __init__(self, pcs, max_clusters, p_min=0.9, n_min=200, w_min=0.1):
        """Initialize GaussianMixtureClustererPruned and set thresholds

        Params
        ======
        pcs (int):
            number of principal components to use before clustering
        max_components (int):
            maximum number of clusters to fit when approximating Dirichlet process
        p_min (float, default=0.9):
            fix minimum probability of datapoints to consider when deciding on keeping
            a cluster
        n_min (int, default=200):
            reject clusters with fewer than n_min datapoints with probability greater
            than p_min of belonging to that cluster
        w_min (float, default=0.1):
            reject clusters with weight lower than w_min
        """
        self.fitted = False

        self.p_min = p_min
        self.n_min = n_min
        self.w_min = w_min

        self.pca = PCA(n_components=pcs)
        self.model = BayesianGaussianMixture(n_components=max_clusters, n_init=3, max_iter=200)

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
        p = np.max(self.model.predict_proba(x), axis=1)

        self.bad_labels = set()  # Set of labels which do not meet our criteria
        for label in range(self.model.n_components):
            # compute number of points with p above our threshold
            count = np.where((p > self.p_min) & (y == label))[0].size
            if count < self.n_min:
                self.bad_labels.add(label)
            if self.model.weights_[label] < self.w_min:
                self.bad_labels.add(label)
        self.good_labels = set(range(self.model.n_components)) - self.bad_labels
        self.bad_labels = np.array(sorted(self.bad_labels))
        self.good_labels = np.array(sorted(self.good_labels))

        self.fitted = True

    def predict(self, waveforms):
        """Predict labels for waveforms

        Any waveforms categorized as one of the excluded clusters is labelled with a -1

        Params
        ======
        waveforms (np.ndarray; (n_datapoints, waveform_width)):
            Dataset of spike waveforms to assign cluster labels to
            Currently slices the 10th to 30th samples of each waveform

        Returns
        =======
        y (np.ndarray; (n_datapoints)):
            Array of integers corresponding to labels for each spike
            Label == -1 when cluster should be ignored
        p (np.ndarray; (n_datapoints)):
            Probability that each datapoint belongs to its assigned cluster
        """
        if not self.fitted:
            raise Exception("Model not fit yet; have you run fit() yet?")

        x = self.pca.transform(waveforms[:, 10:30])
        y = self.model.predict(x)
        p = self.model.predict_proba(x)

        for label in self.bad_labels:
            p[y == label] = 0.0
            y[y == label] = -1

        return y, np.max(p, axis=1)


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
