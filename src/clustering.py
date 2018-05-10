"""Python implementation of superparamagnetic clustering

Detailed by Blatt et. al. 1996
Used for spike sorting in Quiroga et al. 2004
Algorithm based on Bergheim 2011 (Masters thesis)

Example Usage
=============

from clustering import SuperparamagneticClustering

spc = SuperparamagneticClustering(data, k_neighbors=10, n_states=2)
clusters = spc.cluster(T=0.5, iters=500)

for cluster in clusters:
    plt.scatter(*data[cluster].T)
"""

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def pairs_equal(arr):
    """Find the pairs in an array that are equal

    Params
    ======
    arr (np.ndarray, size=(N))

    Returns
    =======
    result (np.ndarray, size=(N, N))
        2d array whose elements (i, j) are True if arr[i] == arr[j]

    Example
    =======
    >>> pairs_equal([6, 5, 1, 5])
    array([
        [True, False, False, False],
        [False, True, False, True],
        [False, False, True, False],
        [False, True, False, True],
    ])
    """
    n = len(arr)
    return (
        np.tile(arr[None, :], (n, 1)) ==
        np.tile(arr[:, None], (1, n))
    )


def sample_spin_state(spins, p_spin, spin_states):
    """Sample a new spin state given a current state

    Updates passed spin state array in place.

    Params
    ======
    spins (np.ndarray, size=(N))
        current spin states. UPDATED IN PLACE.
    p_spin (np.ndarray, size=(N, N))
        probability that a pair of points i, j share the same spin state
    spin_states (np.ndarray, size=(N_states))
        array of possible spin states

    Returns
    =======
    spins (np.ndarray, size=(N))
        Reference to input array, updated with new spin states
    """
    g = nx.Graph()
    edges = (
        pairs_equal(spins) &
        (p_spin > np.random.random(size=p_spin.shape))
    )
    for i, j in zip(*np.where(edges)):
        g.add_edge(i, j)

    subgraphs = np.array([
        np.array(subgraph) for subgraph in
        nx.connected_component_subgraphs(g)
    ])

    for subgraph in subgraphs:
        new_spin = np.random.choice(spin_states)
        spins[subgraph] = new_spin

    return spins


class SuperparamagneticClustering(object):

    def __init__(self, data, k_neighbors=10, n_states=2):
        """Initialize SuperparamagneticClustering object

        Params
        ======
        data (np.ndarray, size=(N_datapoints, N_features))
        k_neighbors (int, default=10)
            Number of nearest neighbors for interactions
        n_states (int, default=2)
            Number of magnetic states to use
        """
        self.data = data
        self.k_neighbors = k_neighbors
        self.n_states = n_states

        self.N = len(data)
        self.spin_states = np.arange(n_states).astype(np.int)

        # Using k+1 to manually exclude points from being their own neighbor
        self._knn = NearestNeighbors(k_neighbors + 1)

        # Cached properties
        self._dist = None
        self._interaction_strength = None
        self._neighbors = None
        self._spins = np.random.choice(self.spin_states, size=(self.N))

    @property
    def dist(self):
        """Distance of each datapoint to its k neighbors"""
        if self._dist is None:
            self._knn.fit(self.data)
            self._dist, self._neighbors = self._knn.kneighbors(self.data)
            # Excluding points from being their own neighbors
            self._dist = self._dist[:, 1:]
            self._neighbors = self._neighbors[:, 1:]
        return self._dist

    @property
    def neighbors(self):
        """Indices of nearest neighbors for each datapoint"""
        if self._neighbors is None:
            # Trigger computation by accessing distance matrix
            self.dist
        return self._neighbors

    @property
    def interaction_strength(self):
        """Interaction strength based on pairwise distance

        Cached because it only needs to be computed once per dataset
        """
        if self._interaction_strength is None:
            J = (1.0 / self.k_neighbors) * np.exp(- self.dist / (2 * np.mean(self.dist)))
            self._interaction_strength = np.zeros((self.N, self.N))
            for i, idx in enumerate(self.neighbors):
                np.put(self._interaction_strength[i], idx, J[i])

        return self._interaction_strength

    def p_spin(self, T):
        """Matrix defining probability that two neighbors are in the same spin state

        Params
        ======
        T (float)
            temperature parameter; lower more ordered (probabilities high) while
            higher is highly disordered (probabilities of sharing state low)
        """
        return 1 - np.exp(- self.interaction_strength / T)

    def susceptibility(self, T, iters=500, reuse_spin_state=False):
        """Compute the magnetic susceptibility of the data given a certain temperature

        Use this to find ideal temperature values. High susceptibility corresponds to
        low temperatures and highly ordered states. Low susceptibility (tending toward
        zero) corresponds to high temperatures and disordered, random states.

        Params
        ======
        T (float)
            temperature parameter; lower T is more ordered (higher susceptibility) while
            higher T is more disordered (lower susceptibility)
        iters (int)
            maximum number of times to sample new spin states
        reuse_spin_state (bool)
            If False, reinitialize spin states randomly. Otherwise use the state from
            previous call to cluster() or susceptibility()

        Returns
        =======
        susceptibility (float)
        """
        p_spin = self.p_spin(T)
        spins = (
            self._spins
            if reuse_spin_state
            else np.random.choice(self.spin_states, size=self.N)
        )

        m_arr = np.zeros(iters)
        for iteration in range(iters):
            spins = sample_spin_state(spins, p_spin, self.spin_states)

            _, counts = np.unique(spins, return_counts=True)
            n_max = np.max(counts)

            magnetization = (
                (self.n_states * n_max - self.N) /
                ((self.n_states - 1) * self.N)
            )

            m_arr[iteration] = magnetization

        self._spins = spins
        return (
            (self.N / T) *
            (np.mean(m_arr ** 2) - np.mean(m_arr) ** 2)
        )

    def cluster(self, T, threshold=0.5, iters=500, reuse_spin_state=False):
        """Perform superparamagnetic clustering

        Params
        ======
        T (float)
            temperature parameter; lower T is more ordered (fewer clusters) while
            higher T is more disordered (many/smaller clusters)
        threshold (float)
            correlated units (similar spin states) will be clustered together. set
            the threshold at which a two correlated points will be considered
            as within the same cluster
        iters (int)
            maximum number of times to sample new spin states
        reuse_spin_state (bool)
            If False, reinitialize spin states randomly. Otherwise use the state from
            previous call to cluster() or susceptibility()

        Returns
        =======
        clusters (np.ndarray, size=(N_clusters))
            Array of arrays; each outer array represents one cluster. Inner arrays
            are integers corresponding to the indices of datapoints belonging to
            each cluster.
        susceptibility (float)
        """
        p_spin = self.p_spin(T)
        spins = (
            self._spins
            if reuse_spin_state
            else np.random.choice(self.spin_states, size=self.N)
        )
        correlation = np.zeros(shape=(self.N, self.N))
        threshold = (self.n_states * threshold - 1) / (self.n_states - 1)
        magnetizations = np.zeros(iters)

        for iteration in range(iters):
            spins = sample_spin_state(spins, p_spin, self.spin_states)

            # Computing susceptibility
            _, counts = np.unique(spins, return_counts=True)
            n_max = np.max(counts)

            magnetization = (
                (self.n_states * n_max - self.N) /
                ((self.n_states - 1) * self.N)
            )
            magnetizations[iteration] = magnetization

            # Clustering
            matching_neighbors = np.where(
                pairs_equal(spins) &
                (p_spin > 0.0)
            )
            correlation[matching_neighbors] = correlation[matching_neighbors] + 1

        # Susceptibility
        susceptibility = (
            (self.N / T) *
            (np.mean(magnetizations ** 2) - np.mean(magnetizations) ** 2)
        )

        # Clustering
        correlation = correlation / iters
        g = nx.Graph()
        for i, j in zip(*np.where(correlation >= threshold)):
            g.add_edge(i, j)

        clusters = np.array([
            np.array(subgraph) for subgraph in nx.connected_component_subgraphs(g)
        ])

        self._spins = spins

        return clusters, susceptibility
