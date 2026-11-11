"""Find Kilosort clusters that are probably two units merged together

A merge of two neurons with similar waveform shapes but different sizes shows up
as two separate, persistent amplitude bands in the cluster's amplitude-vs-time
plot (Kilosort4's merge test compares amplitude-normalised templates, so it
cannot tell such units apart).

For each cluster:

1. split the recording into time bins (``bin_s``);
2. in each bin with enough spikes, fit two Gaussians to log(amplitude) and
   measure their separation (Ashman's D) and the weight of the smaller one;
   a bin is "two-band" if D >= ``min_d`` and the smaller band has >= ``min_weight``
   of the spikes;
3. persistence = fraction of populated bins that are two-band;
4. valley: every spike's log amplitude relative to its bin's upper band, pooled
   over the two-band bins; the density at the dip between the bands divided by
   the smaller band's peak. Two real bands give a deep valley (near 0); a single
   skewed or threshold-truncated distribution has none (near 1).

Candidates have high persistence and a deep valley. Amplitude drift is handled
by fitting each time bin separately.

Usage::

    from songephys.postprocess.merge_check import scan_merge_errors
    df = scan_merge_errors('sites/<site>/ks4_xxx')
    df[df.candidate]
"""
import os

import numpy as np
import pandas as pd


def fit_two_gaussians(x, n_iter=60):
    """1-D two-component Gaussian mixture by EM. Returns (mu, sigma, weight) arrays sorted by mu."""
    x = np.asarray(x, float)
    mu = np.percentile(x, [25, 75]).astype(float)
    if mu[1] - mu[0] < 1e-9:
        mu = mu + np.array([-1e-3, 1e-3])
    sd = np.full(2, max(x.std() / 2, 1e-6))
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        ll -= ll.max(1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        w = nk / len(x)
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-6
    o = np.argsort(mu)
    return mu[o], sd[o], w[o]


def ashman_d(mu, sd):
    return abs(mu[1] - mu[0]) * np.sqrt(2) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)


def _valley(rel, mu_lo, step=0.04, smooth=3):
    """Density at the dip between the bands / the smaller band's peak, from pooled
    log-amplitudes relative to the upper band (upper band at 0, lower at mu_lo < 0)."""
    if not np.isfinite(mu_lo) or mu_lo >= -2 * step:
        return 1.0
    edges = np.arange(mu_lo - 1.0, 1.0 + step, step)
    h, _ = np.histogram(rel, edges)
    h = np.convolve(h, np.ones(smooth) / smooth, mode='same')
    c = edges[:-1] + step / 2
    i_lo, i_hi = np.argmin(np.abs(c - mu_lo)), np.argmin(np.abs(c))
    if i_hi - i_lo < 2:
        return 1.0
    lo_peak = h[max(i_lo - 2, 0):i_lo + 3].max()
    hi_peak = h[max(i_hi - 2, 0):i_hi + 3].max()
    dip = h[i_lo:i_hi + 1].min()
    return float(dip / max(min(lo_peak, hi_peak), 1e-9))


def cluster_band_metrics(times_s, amps, t_end, bin_s=300.0, min_spikes_per_bin=100,
                         max_spikes_per_bin=2000, min_d=2.0, min_weight=0.05, rng=None):
    """Two-band metrics for one cluster (see module docstring). Returns a dict."""
    rng = np.random.default_rng(0) if rng is None else rng
    amps = np.asarray(amps, float)
    good = amps > 0
    times_s, la = np.asarray(times_s, float)[good], np.log(amps[good])
    edges = np.arange(0, t_end + bin_s, bin_s)
    b = np.clip(np.searchsorted(edges, times_s, side='right') - 1, 0, len(edges) - 2)
    rows, rel = [], []
    for k in range(len(edges) - 1):
        x = la[b == k]
        if len(x) < min_spikes_per_bin:
            continue
        xs = x if len(x) <= max_spikes_per_bin else rng.choice(x, max_spikes_per_bin, replace=False)
        mu, sd, w = fit_two_gaussians(xs)
        d = ashman_d(mu, sd)
        two = d >= min_d and w.min() >= min_weight
        rows.append((k, mu[0], mu[1], sd[0], sd[1], w[0], w[1], d, two))
        if two:
            rel.append(x - mu[1])
    if not rows:
        return dict(n_bins=0, persistence=np.nan, valley=np.nan, median_d=np.nan,
                    amp_ratio=np.nan, low_band_frac=np.nan, ratio_cv=np.nan, bins=None)
    R = pd.DataFrame(rows, columns=['bin', 'mu_lo', 'mu_hi', 'sd_lo', 'sd_hi', 'w_lo', 'w_hi', 'd', 'two_band'])
    two = R[R.two_band]
    persistence = float(R.two_band.mean())
    if len(two):
        gap = (two.mu_hi - two.mu_lo).values
        amp_ratio = float(np.exp(np.median(gap)))
        ratio_cv = float(np.std(np.exp(gap)) / np.mean(np.exp(gap)))
        valley = _valley(np.concatenate(rel), -float(np.median(gap)))
        low_frac = float(np.median(two.w_lo))
    else:
        amp_ratio = ratio_cv = low_frac = np.nan
        valley = 1.0
    return dict(n_bins=len(R), persistence=persistence, valley=valley, median_d=float(np.median(R.d)),
                amp_ratio=amp_ratio, low_band_frac=low_frac, ratio_cv=ratio_cv, bins=R)


def scan_merge_errors(ks_dir, by='clusters', bin_s=300.0, min_spikes=1000, min_spikes_per_bin=100,
                      min_d=2.0, min_weight=0.05, min_persistence=0.5, max_valley=0.5,
                      keep_bins=False, progress=True):
    """Scan a Kilosort/phy output folder for clusters with two persistent amplitude bands.

    by : 'clusters' (spike_clusters.npy: phy's current clusters) or 'templates'
         (spike_templates.npy: Kilosort's original output)
    Returns a DataFrame sorted best candidates first, with columns
    cluster_id, n_spikes, persistence, valley, median_d, amp_ratio (upper/lower band),
    low_band_frac, ratio_cv, n_bins, candidate, score (= persistence * (1 - valley)),
    and peak_channel if templates.npy exists.
    """
    load = lambda f: np.load(os.path.join(ks_dir, f))
    st = load('spike_times.npy').ravel()
    amps = load('amplitudes.npy').ravel()
    ids = load('spike_clusters.npy' if by == 'clusters' else 'spike_templates.npy').ravel()
    fs = 30000.0
    pp = os.path.join(ks_dir, 'params.py')
    if os.path.exists(pp):
        ns = {}
        exec(open(pp).read(), {}, ns)
        fs = float(ns.get('sample_rate', fs))
    t = st / fs
    t_end = float(t.max()) + 1e-6
    peak_of_template = tmpl = None
    if os.path.exists(os.path.join(ks_dir, 'templates.npy')):
        tmpl = load('spike_templates.npy').ravel()
        peak_of_template = np.abs(load('templates.npy')).max(1).argmax(1)
    order = np.argsort(ids, kind='stable')
    uniq, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
    rows, bins = [], {}
    for n_done, (cid, s0, n) in enumerate(zip(uniq, starts, counts)):
        if n < min_spikes:
            continue
        idx = order[s0:s0 + n]
        m = cluster_band_metrics(t[idx], amps[idx], t_end, bin_s, min_spikes_per_bin,
                                 min_d=min_d, min_weight=min_weight)
        row = dict(cluster_id=int(cid), n_spikes=int(n), **{k: v for k, v in m.items() if k != 'bins'})
        if peak_of_template is not None:
            row['peak_channel'] = int(np.bincount(peak_of_template[tmpl[idx]]).argmax())
        rows.append(row)
        if keep_bins:
            bins[int(cid)] = m['bins']
        if progress and len(rows) % 100 == 0:
            print('  scanned %d clusters' % len(rows))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['candidate'] = (df.persistence >= min_persistence) & (df.valley <= max_valley)
    # rank: a deep valley is the strongest evidence; persistence can be lower when
    # one of the units stops firing part-way through
    df['score'] = df.persistence * (1 - df.valley.clip(0, 1))
    df = df.sort_values(['candidate', 'score'], ascending=[False, False]).reset_index(drop=True)
    if keep_bins:
        df.attrs['bins'] = bins
    return df


def write_phy_column(df, ks_dir, name='merge_suspect'):
    """Write cluster_<name>.tsv so phy shows the score as a sortable column
    (persistence for candidates, 0 otherwise)."""
    out = pd.DataFrame({'cluster_id': df.cluster_id,
                        name: np.where(df.candidate, df.persistence.round(2), 0.0)})
    path = os.path.join(ks_dir, 'cluster_%s.tsv' % name)
    out.to_csv(path, sep='\t', index=False)
    return path
