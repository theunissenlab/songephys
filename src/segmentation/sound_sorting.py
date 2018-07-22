import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from soundsig.sound import plot_spectrogram, spectrogram

from segmentation import autodetect
from block_io import load_block
from segment_analysis import MultiSegment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture

import config


def get_threshold(segments):
    multiseg = MultiSegment(segments)
    all_sounds = []
    for seg in multiseg.segments:
        t_arr, sound, fs = multiseg.get_analogsignal(
            seg, 
            signal=0,
            channel=1)
        sound = sound - np.mean(sound)
        all_sounds.append(sound)
    all_sounds = np.concatenate(all_sounds)
    
    return autodetect.get_amp_thresh(all_sounds, fs=fs)    


def find_syllables(segments, signal=0, channel=1):
    sample_start = np.floor(config.PAD_SONG_BEFORE * 30000.0).astype(np.int)
    sample_stop = np.ceil(config.PAD_SONG_AFTER * 30000.0).astype(np.int)
    
    threshold = get_threshold(segments)

    multiseg = MultiSegment(segments)
    syllable_times = list()
    for seg in multiseg.segments:
        _, sound, fs = multiseg.get_analogsignal(
            seg, 
            signal=signal,
            channel=channel)
        sound = sound[sample_start:-sample_stop]
        sound = sound - np.mean(sound)
        syllables, _ = autodetect.find_syllables(
            sound,
            fs=fs,
            amp_thresh=threshold)
        syllables += sample_start
        
        if len(syllables) > 0:
            syllable_times.append(syllables)
        else:
            syllable_times.append(np.zeros((0, 2)))
    
    return syllable_times


def get_center_of_mass(t_arr, amp_env):
    return np.sum(amp_env * t_arr) / np.sum(amp_env)


def entropy(spec):
    p = np.sum(np.abs(spec) ** 2, axis=1)
    p = p / np.sum(p)
    return np.sum(- p * np.log2(p))


class Syllable(object):
    """
    Defines a syllable by its position in a segment

    Attrs
        start_idx, stop_idx: indices delimiting the syllable

    """
    def __init__(self, segment, start_idx, stop_idx):
        self.segment = segment
        self._start = start_idx
        self._stop = stop_idx
        self._amp_env = autodetect.get_amplitude_envelope(
            self.signal()
        )
        self._center = int(get_center_of_mass(
            np.arange(self._start, self._stop),
            self._amp_env[self._start:self._stop]
        ))
        
        self.t_start = self._start / 30000.0
        self.t_stop = self._stop / 30000.0
        self.t_center = self._center / 30000.0
        
    def __repr__(self):
        return "Syllable {:.2f}s to {:.2f}s".format(
                self.segment.t_start.item() + self.t_start,
                self.segment.t_start.item() + self.t_stop
        )

    def get_amp_env(self, channel=1):
        return autodetect.get_amplitude_envelope(
            self.signal(channel=channel)
        )

    def amp_env_ratio(self):
        return np.mean(self.get_amp_env(1) / self.get_amp_env(0))
    
    def signal(self, channel=1):
        return self.segment.analogsignals[0].as_array()[:, channel]
    
    def get_sig(self, dur=3000, channel=1):
        """
        Return a signal centered (dur/2) on t_center
        """
        result = np.zeros(dur)
        result[:dur//2] = self.signal(channel=channel)[self._center - dur//2:self._center]
        result[dur//2:] = self.signal(channel=channel)[self._center:self._center + dur//2]

        t_arr = np.arange(0, dur)
        t_arr = t_arr / 30000.0
        t_arr -= ((dur // 2) / 30000.0)

        return t_arr, result


class SpectrogramCollection(object):
    def __init__(self, t_spec, f_spec, specs):
        self.t_spec = t_spec
        self.f_spec = f_spec
        self.data = specs

    def logged(self):
        return np.log10(np.abs(self.data))

    def mean(self):
        return np.mean(
                self.logged(),
                axis=0
        )

    def select(self, x):
        return SpectrogramCollection(
                self.t_spec, 
                self.f_spec,
                self.data[x]
        )

    def plot_mean(self, ax, cmap):
        ax.pcolormesh(
                self.t_spec,
                self.f_spec,
                self.mean(),
                cmap=cmap
        )


def collect_syllable_objects(multiseg, signal=0, channel=1):
    syllable_times = find_syllables(multiseg.segments, signal=signal, channel=channel)
    syllables = []
    for seg_idx, syllable_idxs in enumerate(syllable_times):
        for start_idx, stop_idx in syllable_idxs:
            syllables.append(
                Syllable(
                    multiseg.segments[seg_idx],
                    start_idx,
                    stop_idx
                )
            )

    return np.array(syllables)


def plot_syllable(syllable, dur=3000):
    """
    Dur (duration) in samples
    """
    t_arr, sig = syllable.get_sig(dur=3000)
    t_spec, f_spec, spec, rms = spectrogram(
        sig - sig.mean(),
        30000.0,
        1000.0,
        50.0,
        min_freq=250,
        max_freq=10000,
        nstd=6
    )
    t_spec = t_spec + t_arr[0]
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0, 0, 0.1, 1])
    plot_spectrogram(
        t_spec, 
        f_spec, 
        spec, 
        colormap="gray_r", 
        ax=ax, 
        colorbar=False)
    
    ax.vlines([
        - (syllable.t_center - syllable.t_start),
        0,
        (syllable.t_stop - syllable.t_center)
    ], *ax.get_ylim(), "r", linestyle="--")
    
    plt.show()
    plt.close(fig)


# colelct spectrograms
# do pca
# do gmm
# select noise clusters
# filter spectrograms
# repeat
def collect_spectrograms(syllables, channel=1, dur=3000):
    spectrograms = []
    for syl_idx, syl in enumerate(syllables):
        t_arr, sig = syl.get_sig(channel=channel, dur=dur)
        t_spec, f_spec, spec, _ = spectrogram(
                sig - sig.mean(),
                30000.0,
                1000.0,
                50.0,
                min_freq=250,
                max_freq=10000,
                nstd=6
        )
        t_spec += t_arr[0]
        spectrograms.append(spec)

    return SpectrogramCollection(t_spec, f_spec, np.array(spectrograms))


def collect_spectrogram_diffs(syllables, dur=3000):
    spec0 = collect_spectrograms(syllables, channel=0, dur=dur)
    spec1 = collect_spectrograms(syllables, channel=1, dur=dur)

    return spec0, spec1, SpectrogramCollection(
            spec0.t_spec,
            spec0.f_spec,
            spec0.logged() / spec1.logged()
    )


def sort_sounds(syllables, spectrograms=None, spectrogram_diffs=None):
    if spectrograms is None:
        spectrograms = collect_spectrograms(syllables)

    data = spectrograms.logged().reshape(len(spectrograms.data), -1)
    if spectrogram_diffs is not None:
        diff_data = spectrogram_diffs.data.reshape(len(spectrogram_diffs.data), -1)
        data = np.hstack([data, diff_data])

    pca = PCA(n_components=20)
    spectro_pca = pca.fit_transform(
        scipy.stats.zscore(
            data,
            axis=0
        )
    )

    gmm = BayesianGaussianMixture(n_components=30, covariance_type="full")
    gmm.fit(spectro_pca)
    labels = gmm.predict(spectro_pca)

    return spectro_pca, syllables, spectrograms, labels


def plot_sort_results(syllables, spectrograms, labels, figsize=(8, 3)):

    for label in np.unique(labels):
        fig = plt.figure(figsize=figsize)
        ax_mean = fig.add_axes([0, 0, 0.1, 1])
        specs = spectrograms.select(labels == label)
        print("Label", label)
        print("n =", len(specs.data))
        specs.plot_mean(ax_mean, cmap="gray_r")
        
        for i in range(len(specs.data))[:9]:
            ax = fig.add_axes([0.1 + i * 0.1, 0, 0.1, 1])
            plot_spectrogram(
                specs.t_spec, 
                specs.f_spec, 
                specs.data[i], 
                colormap="gray_r", 
                ax=ax, 
                colorbar=False)
            ax.axis("off")
        plt.show()
        plt.close(fig)


def remove_noise(noise_labels, labels, select_arrays=tuple(), arrays=tuple()):
    selector = np.logical_not(np.isin(labels, noise_labels))
    return (
            [labels[selector]] +
            [arr.select(selector) for arr in select_arrays] +
            [arr[selector] for arr in arrays]
    )

