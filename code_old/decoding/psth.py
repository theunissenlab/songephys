from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import numpy as np
from spikes.utils import bin_spikes
from spikes.psth import conv, gaussian_convolver, exponential_convolver
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.metrics import confusion_matrix


def sample_psth(rows, p=0.5, repeat=10):
    psths = []
    for _ in np.arange(repeat):
        select = np.random.choice(np.arange(rows.shape[1]), size=int(np.ceil(rows.shape[1] * p)))
        psths.append(np.mean(rows[:, select], axis=1))

    return np.mean(psths, axis=0) * 1000, np.std(psths, axis=0) * 1000


def get_raster(spikes, times, before, after, filt=False):
    spike_times = []
    for t in times:
        t0 = t - before
        t1 = t + after
        sts = spikes[(spikes > t0) & (spikes < t1)] - t0
        if filt and not len(sts):
            continue
        spike_times.append(sts)

    if not len(spike_times) or not len(np.concatenate(spike_times)):
        return None, None, None
    
    dur = before + after
    n_bins = int(dur / 0.001)
    t, raster = bin_spikes(spike_times, min_time=0, dt=0.001, n_bins=n_bins, binary=True)
    return t - before, raster, np.array([st - before for st in spike_times])


def get_counts(df):
    if "syllable_id" in df:
        return df.groupby(
            ['label','sound_type','vocalizer', "syllable_id"]
        ).size().reset_index().rename(columns={0:'count'})
    return df.groupby(
        ['label','sound_type','vocalizer']
    ).size().reset_index().rename(columns={0:'count'})


def decode(psths, labels, shuffle=False):
    """Run a simple nearest centroid decoder;
    
    Run this multiple times to estimate mutual information
    
    Split each set into 3:
        Choose the mean on 2/3
        Test on the last third
    """
    results = []
    strat = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    labels = np.copy(labels)
    if shuffle:
        np.random.shuffle(labels)
    for train_idx, test_idx in strat.split(psths, labels):
        X_train = psths[train_idx]
        y_train = labels[train_idx]
        X_test = psths[test_idx]
        y_test = labels[test_idx]
        
        nc = NearestCentroid()
        nc.fit(X_train, y_train)
        y_pred = nc.predict(X_test)
        results.append((y_pred, y_test))
    
    return results


def mean_spec(syllables, loader, channel, max_count=50, range=(-0.05, 0.35)):
    all_specs = []
    
    for idx, row in syllables.sample(n=max_count).iterrows():
        data = loader.time_slice(row["t_start"] + range[0], row["t_start"] + range[1])
        sig = data["mic"].data[:, channel]
        sig = sig - np.mean(sig)
        t_spec, f_spec, spec, _ = spectrogram(
            sig,
            fs, 
            1000, 50, 500, 8000, cmplx=False)
        all_specs.append(spec)
        
    width = int(round(np.median([x.shape[1] for x in all_specs])))
    full_spec = np.zeros((all_specs[0].shape[0], width))
    
    for i, spec in enumerate(sorted(all_specs, key=lambda s: s.shape[1], reverse=True)):
        if np.abs(width - spec.shape[1]) % 2 == 1:
            d = np.random.choice([0, 1]) 
        else:
            d = 0 

        if spec.shape[1] < width:
            start = d + (width - spec.shape[1]) // 2
            full_spec[:, start:start + spec.shape[1]] *= (i / (i + 1)) 
            full_spec[:, start:start + spec.shape[1]] += (1 / (i + 1)) * spec

        elif spec.shape[1] >= width:
            start = d + (spec.shape[1] - width) // 2
            full_spec *= (i / (i + 1)) 
            full_spec += (1 / (i + 1)) * spec[:, start:start + width]
            
    return t_spec + range[0], f_spec, full_spec


def peak_align(syllables, loader, channel, max_count=50, range=(-0.05, 0.35)):
    raise Exception

