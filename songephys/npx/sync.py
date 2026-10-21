"""Put everything on one clock: seconds along the CatGT-concatenated NI file.

Why not just TPrime? CatGT glues triggered recordings together, so the 1 Hz sync
pulse train jumps phase at every join where time was skipped. TPrime pairs edges
across the whole file and mis-pairs them near those joins (spikes in ~75/192
segments of Choc57_d1.1 moved by up to +-500 ms). Here each input file (one
piece of the tcat output, as listed in ct_offsets) is aligned on its own:

* the piece's start in each stream comes from ct_offsets,
* imec and NI sync edges inside the piece are paired by nearest-neighbour, and
* the imec->NI offset is linearly interpolated between paired edges (held
  constant past the first/last pair), which is what TPrime does inside a piece.
"""
import numpy as np
import pandas as pd


class NIClock(object):
    """Maps imec times and absolute (acquisition) sample counts onto NI tcat seconds."""

    def __init__(self, ct_offsets, imec_edges, ni_edges, imec_fs, ni_fs,
                 ni_total_samples, imec_total_samples, raw_files=None, probe=0,
                 max_pair_err_s=0.05):
        self.ni_fs = float(ni_fs)
        self.imec_fs = float(imec_fs)
        self.ni_smp = ct_offsets['smp_nidq'].to_numpy()
        self.imec_smp = ct_offsets[f'smp_imap{probe}'].to_numpy()
        self.ni_start = self.ni_smp / self.ni_fs
        self.imec_start = self.imec_smp / self.imec_fs
        self.ni_total = int(ni_total_samples)
        self.imec_total = int(imec_total_samples)
        self.raw_files = raw_files
        self.n_pieces = len(self.ni_start)

        ni_end = np.r_[self.ni_start[1:], self.ni_total / self.ni_fs]
        imec_end = np.r_[self.imec_start[1:], self.imec_total / self.imec_fs]
        imec_edges = np.sort(np.asarray(imec_edges, dtype=float))
        ni_edges = np.sort(np.asarray(ni_edges, dtype=float))

        self.pairs = []
        rows = []
        for k in range(self.n_pieces):
            ie = imec_edges[(imec_edges >= self.imec_start[k]) & (imec_edges < imec_end[k])]
            ne = ni_edges[(ni_edges >= self.ni_start[k]) & (ni_edges < ni_end[k])]
            pi, pn = np.array([]), np.array([])
            if len(ie) > 0 and len(ne) > 0:
                pred = ie - self.imec_start[k] + self.ni_start[k]
                j = np.clip(np.searchsorted(ne, pred), 1, len(ne) - 1) if len(ne) > 1 else np.zeros(len(pred), int)
                if len(ne) > 1:
                    j = np.where(np.abs(ne[j - 1] - pred) < np.abs(ne[j] - pred), j - 1, j)
                err = ne[j] - pred
                ok = np.abs(err) < max_pair_err_s
                # one NI edge per imec edge
                ok &= ~pd.Series(j).duplicated(keep=False).to_numpy()
                pi, pn = ie[ok], ne[j[ok]]
            self.pairs.append((pi, pn - pi))
            rows.append(dict(piece=k, n_imec_edges=len(ie), n_ni_edges=len(ne), n_pairs=len(pi),
                             start_offset_ms=(self.ni_start[k] - self.imec_start[k]) * 1e3,
                             median_edge_offset_ms=np.median(pn - pi) * 1e3 if len(pi) else np.nan))
        self.piece_summary = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def imec_to_ni(self, t_imec):
        """imec tcat seconds -> NI tcat seconds."""
        t = np.asarray(t_imec, dtype=float)
        flat = t.ravel()
        out = np.full(flat.shape, np.nan)
        piece = np.clip(np.searchsorted(self.imec_start, flat, side='right') - 1, 0, self.n_pieces - 1)
        for k in np.unique(piece):
            m = piece == k
            pi, off = self.pairs[k]
            if len(pi) == 0:
                out[m] = flat[m] - self.imec_start[k] + self.ni_start[k]
            else:
                out[m] = flat[m] + np.interp(flat[m], pi, off)
        return out.reshape(t.shape)

    # ------------------------------------------------------------------
    def _abs_to_tcat(self, s, stream):
        if self.raw_files is None:
            raise ValueError("Raw file table is needed to map absolute sample counts")
        first = self.raw_files[f'{stream}_first_sample'].to_numpy()
        nsamp = self.raw_files[f'{stream}_n_samples'].to_numpy()
        smp = self.ni_smp if stream == 'ni' else self.imec_smp
        total = self.ni_total if stream == 'ni' else self.imec_total
        kept = np.minimum(nsamp, np.diff(np.r_[smp, total]))
        s = np.asarray(s, dtype=np.int64)
        k = np.searchsorted(first, s, side='right') - 1
        kc = np.clip(k, 0, len(first) - 1)
        rel = s - first[kc]
        valid = (k >= 0) & (rel >= 0) & (rel < kept[kc])
        tc = np.where(valid, smp[kc] + rel, -1)
        return tc, valid

    def ni_abs_sample_to_time(self, s):
        """Absolute NI sample (since acquisition start) -> NI tcat seconds (NaN if not recorded)."""
        tc, valid = self._abs_to_tcat(s, 'ni')
        return np.where(valid, tc / self.ni_fs, np.nan)

    def imec_abs_sample_to_time(self, s):
        """Absolute imec sample (since acquisition start) -> NI tcat seconds (NaN if not recorded)."""
        tc, valid = self._abs_to_tcat(s, 'imec')
        t = self.imec_to_ni(np.where(valid, tc / self.imec_fs, 0.0))
        return np.where(valid, t, np.nan)

    # ------------------------------------------------------------------
    def segments(self):
        """Contiguous recording segments (runs of overlapping/abutting trigger files).

        Returns a DataFrame with start_time/stop_time in NI tcat seconds, plus the
        absolute first samples and t-indices, so the real time between triggers
        can be recovered."""
        nf = self.n_pieces
        ni_end = np.r_[self.ni_start[1:], self.ni_total / self.ni_fs]
        if self.raw_files is None:
            blocks = [[k] for k in range(nf)]
        else:
            first = self.raw_files['ni_first_sample'].to_numpy()
            nsamp = self.raw_files['ni_n_samples'].to_numpy()
            blocks = [[0]]
            for k in range(1, nf):
                if first[k] <= first[k - 1] + nsamp[k - 1] + 1:
                    blocks[-1].append(k)
                else:
                    blocks.append([k])
        rows = []
        for b in blocks:
            row = dict(start_time=float(self.ni_start[b[0]]), stop_time=float(ni_end[b[-1]]),
                       first_t_index=int(b[0]), last_t_index=int(b[-1]))
            if self.raw_files is not None:
                rf = self.raw_files.iloc[b[0]]
                row.update(ni_abs_first_sample=int(rf['ni_first_sample']),
                           imec_abs_first_sample=int(rf['imec_first_sample']),
                           acq_time_s=float(rf['ni_first_sample']) / self.ni_fs,
                           file_create_time=str(rf['file_create_time']))
            rows.append(row)
        return pd.DataFrame(rows)


def build_ni_clock(catgt_run, raw_files=None):
    """Construct an NIClock from a spikeglx.CatGTRun."""
    from .spikeglx import read_edges, sample_rate, n_samples
    ni_meta = catgt_run.ni_meta
    ap_meta = catgt_run.ap_meta
    return NIClock(
        ct_offsets=catgt_run.ct_offsets,
        imec_edges=read_edges(catgt_run.imec_sync_edges_path),
        ni_edges=read_edges(catgt_run.ni_sync_edges_path),
        imec_fs=sample_rate(ap_meta), ni_fs=sample_rate(ni_meta),
        ni_total_samples=n_samples(ni_meta), imec_total_samples=n_samples(ap_meta),
        raw_files=raw_files, probe=catgt_run.probe,
    )
