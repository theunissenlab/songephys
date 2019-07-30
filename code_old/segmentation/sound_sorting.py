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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import BayesianGaussianMixture

import config


def get_threshold(segments, signal=0, channel=0):
    multiseg = MultiSegment(segments)
    all_sounds = []
    for seg in multiseg.segments:
        t_arr, sound, fs = multiseg.get_analogsignal(
            seg, 
            signal=signal,
            channel=channel)
        sound = sound - np.mean(sound)
        if signal != "playback":
            all_sounds.append(sound[:15000])  # Take only from the beginning
        else:
            all_sounds.append(sound)
    all_sounds = np.concatenate(all_sounds)
   
    if signal != "playback":
        return autodetect.get_amp_thresh(all_sounds, fs=fs, mode="median")
    else:
        return autodetect.get_amp_thresh(all_sounds, fs=fs, mode="mean")


    sample_start = np.floor(config.PAD_SONG_BEFORE * 30000.0).astype(np.int)
    
    # FIXME: pick threshold better?
    # this allows for manual threshold override
    if threshold is None:
        threshold = get_threshold(segments, signal=signal, channel=channel)

    multiseg = MultiSegment(segments)
    syllable_times = list()

    playback_cache = {}
    for seg in multiseg.segments:
        if signal == "playback":
            pb_file = os.path.basename(seg.annotations["playback_file"])

        if signal == "playback" and pb_file in playback_cache:
            syllables = playback_cache[pb_file]
            if len(syllables) > 0:
                syllable_times.append(syllables)
            else:
                syllable_times.append(np.zeros((0, 2)))
            continue

        t, sound, fs = multiseg.get_analogsignal(
            seg, 
            signal=signal,
            channel=channel,
            original_times=True)

        sound = sound[sample_start:-sample_stop]

        # FIXME: if the segment got cut off, sound might be empty after
        # slicing.... for now just ignore it
        if not len(sound):
            print("uhhh", sample_start, sample_stop)
            continue

        sound = sound - np.mean(sound)
        syllables, _ = autodetect.find_syllables(
            sound,
            fs=fs,
            amp_thresh=threshold,
            shortest_duration=0.04)

        syllables += sample_start

        if signal == "playback":
            playback_cache[pb_file] = syllables
        
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
    def __init__(self, segment, start_idx, stop_idx, center_idx=None, is_playback=False):
        self.is_playback = is_playback
        self.segment = segment
        self._start = start_idx
        self._stop = stop_idx
        self._amp_env = autodetect.get_amplitude_envelope(
            self.signal()
        )
        if center_idx is None:
            self._center = int(get_center_of_mass(
                np.arange(self._start, self._stop),
                self._amp_env[self._start:self._stop]
            ))
        else:
            self._center = center_idx
        
        self.t_start = self._start / 30000.0
        self.t_stop = self._stop / 30000.0
        self.t_center = self._center / 30000.0
        
    def __repr__(self):
        return "Syllable {:.2f}s to {:.2f}s".format(
                self.segment.t_start.item() + self.t_start,
                self.segment.t_start.item() + self.t_stop
        )

    def shift(self, samples):
        """Shift the syllable in time by {samples}"""
        self._start += samples
        self._center += samples
        self._stop += samples

        self.t_start = self._start / 30000.0
        self.t_stop = self._stop / 30000.0
        self.t_center = self._center / 30000.0

    def get_amp_env(self, channel=1):
        return autodetect.get_amplitude_envelope(
            self.get_sig(dur=3000, channel=channel)[1]
        )

    def amp_env_ratio(self):
        diff = self.get_amp_env(1) - self.get_amp_env(0)
        i = np.argmax(np.abs(diff)[1000:-1000])
        return diff[1000:-1000][i]
    
    def signal(self, channel=1):
        _multiseg = MultiSegment([self.segment])
        if self.is_playback:
            return _multiseg.get_analogsignal(0, signal="playback", channel=None)[1]
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


def collect_syllable_objects(multiseg, signal=0, channel=1, threshold=None):
    syllable_times = find_syllables(multiseg.segments, signal=signal, channel=channel, threshold=threshold)
    syllables = []
    i = 0
    for seg_idx, syllable_idxs in enumerate(syllable_times):
        for start_idx, stop_idx in syllable_idxs:
            i += 1
            try:
                syllables.append(
                    Syllable(
                        multiseg.segments[seg_idx],
                        start_idx,
                        stop_idx,
                        is_playback=(signal == "playback")
                    )
                )
            except ValueError as e:
                print("dodgin error {}".format(e))
                continue

    return np.array(syllables)


def plot_syllable(syllable, ax=None, dur=3000, channel=0):
    """
    Dur (duration) in samples
    """
    t_arr, sig = syllable.get_sig(dur=dur, channel=channel)
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
    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_axes([0, 0, 0.1, 1])
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
    
    # plt.show()
    # plt.close(fig)


# colelct spectrograms
# do pca
# do gmm
# select noise clusters
# filter spectrograms
# repeat
def collect_spectrograms(syllables, channel=0, dur=3000):
    spectrograms = []
    bad_index = []
    for syl_idx, syl in enumerate(syllables):
        # if syl.segment.annotations.get("bird") == "tutor":
        #     channel = 0
        # elif syl.segment.annotations.get("bird") == "subject":
        #     channel = 1
        try:
            t_arr, sig = syl.get_sig(channel=channel, dur=dur)
        except ValueError:
            bad_index.append(syl_idx)
            continue
        t_spec, f_spec, spec, _ = spectrogram(
                sig - sig.mean(),
                30000.0,
                1000.0,
                50.0,
                min_freq=250,
                max_freq=10000,
                nstd=6,
                cmplx=False,
        )
        t_spec += t_arr[0]
        spectrograms.append(spec)

    return np.array(syllables)[np.logical_not(np.isin(np.arange(len(syllables)), bad_index))], SpectrogramCollection(t_spec, f_spec, np.array(spectrograms))


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
        syllables, spectrograms = collect_spectrograms(syllables)

    data = spectrograms.logged().reshape(len(spectrograms.data), -1)
    if spectrogram_diffs is not None:
        diff_data = spectrogram_diffs.data.reshape(len(spectrogram_diffs.data), -1)
        data = np.hstack([data, diff_data])

    pca = PCA(n_components=40)
    spectro_pca = pca.fit_transform(
        scipy.stats.zscore(
            data,
            axis=0
        )
    )

    gmm = BayesianGaussianMixture(n_components=40, covariance_type="full")
    gmm.fit(spectro_pca)
    labels = gmm.predict(spectro_pca)

    for _ in range(5):
        knn = KNeighborsClassifier(n_neighbors=20).fit(spectro_pca[::2], labels[::2])
        labels = knn.predict(spectro_pca)

    return pca, spectro_pca, syllables, spectrograms, labels


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
        yield label, fig, ax
        plt.show()
        plt.close(fig)


def remove_noise(noise_labels, labels, select_arrays=tuple(), arrays=tuple()):
    selector = np.logical_not(np.isin(labels, noise_labels))
    return (
            [labels[selector]] +
            [arr.select(selector) for arr in select_arrays] +
            [arr[selector] for arr in arrays]
    )


def save_syllables(filename, syllable_dict):
    """
    Saves a dict where keys are labels and values are lists of syllables
    """
    to_save = {}

    for key, syllables in syllable_dict.items():
        segment_starts = [syl.segment.t_start for syl in syllables]
        start_ind = [syl._start for syl in syllables]
        center_ind = [syl._center for syl in syllables]
        stop_ind = [syl._stop for syl in syllables]
        playbacks = [syl.is_playback for syl in syllables]
        to_save[key] = list(zip(
                segment_starts,
                start_ind,
                center_ind,
                stop_ind,
                playbacks
        ))

    np.save(filename, to_save)


def load_syllables(filename, segments):
    saved_data = np.load(filename)[()]
    overall_result = {}

    for key, data in saved_data.items():
        _idx = 0
        result = []
        for row in data:
            if len(row) == 5:
                seg_start, start_ind, center_ind, stop_ind, is_playback = row
            elif len(row) == 4:
                seg_start, start_ind, stop_ind, is_playback = row
                center_ind = None
            for seg in segments[_idx:]:
                if seg.t_start == seg_start:
                    result.append(Syllable(
                        seg, start_ind, stop_ind, center_ind, is_playback=is_playback
                    ))
                    break
                elif seg.t_start > seg_start:
                    _idx -= 1
                    break
                _idx += 1 
            else:
                raise Exception("Couldnt find segment for syllable at {}".format(seg_start))
        overall_result[key] = result
        
    return overall_result
