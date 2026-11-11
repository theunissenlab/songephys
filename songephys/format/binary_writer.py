"""Write a SpikeInterface recording to a flat binary file (samples x channels)

Drop-in replacement for ``kilosort.io.spikeinterface_to_binary`` for the large
ephys binaries. That function maps the whole output file into memory and
flushes it after every chunk from many threads, which on a 131 GB recording ran
about 5x slower than the preprocessing itself, and it silently leaves zeros
where a chunk raises an error. Here chunks are computed in a thread pool and
written in order through a normal file handle:

- no memory-mapped output, no per-chunk flush
- errors in any chunk are raised, not swallowed
- output goes to ``<path>.partial`` and is renamed only when complete, so an
  interrupted run never leaves a full-size file that looks finished

The format is the same: raw ``dtype`` values, samples x channels, C order, no
header, segments concatenated in order.
"""
import os
import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def write_recording_binary(recording, path, dtype=np.int16, chunksize=30000,
                           max_workers=None, progress_every_s=30.0):
    """Write ``recording`` to ``path`` as a flat binary file

    Parameters
    ----------
    recording : SpikeInterface recording (anything with get_num_segments,
        get_num_samples(segment_index), get_num_channels, get_traces)
    path : str, output file
    dtype : output dtype; traces are cast with ``astype`` (as numpy does when
        assigning into kilosort's memmap)
    chunksize : samples per chunk
    max_workers : threads computing chunks. Default min(16, cpu count): the
        preprocessing chain is limited by Python's GIL beyond about 8 threads.
    progress_every_s : seconds between progress log lines

    Returns
    -------
    (n_samples, n_channels) written
    """
    dtype = np.dtype(dtype)
    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 1)
    n_seg = recording.get_num_segments()
    n_ch = recording.get_num_channels()
    jobs = []
    for seg in range(n_seg):
        n = recording.get_num_samples(segment_index=seg)
        jobs += [(seg, i, min(i + chunksize, n)) for i in range(0, n, chunksize)]
    total = sum(b - a for _, a, b in jobs)

    def compute(job):
        seg, a, b = job
        t = recording.get_traces(start_frame=a, end_frame=b, segment_index=seg)
        if t.shape != (b - a, n_ch):
            raise ValueError("Chunk %s returned shape %s, expected %s" % (job, t.shape, (b - a, n_ch)))
        return np.ascontiguousarray(t.astype(dtype, copy=False))

    partial = path + ".partial"
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    t_start = t_log = time.perf_counter()
    done = 0
    logger.info("Writing %s: %d samples x %d channels, %d chunks, %d threads",
                path, total, n_ch, len(jobs), max_workers)
    try:
        with open(partial, "wb") as f, ThreadPoolExecutor(max_workers=max_workers) as exe:
            pending = deque()
            it = iter(jobs)
            # keep a bounded number of chunks in flight; write them in order
            for job in it:
                pending.append(exe.submit(compute, job))
                if len(pending) >= 2 * max_workers:
                    break
            while pending:
                data = pending.popleft().result()  # raises if the chunk failed
                f.write(data.data)
                done += data.shape[0]
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(exe.submit(compute, nxt))
                now = time.perf_counter()
                if now - t_log >= progress_every_s:
                    t_log = now
                    rate = done / (now - t_start)
                    msg = "  %s: %.1f%% (%.1fx real time), about %.0f min left" % (
                        os.path.basename(path), 100.0 * done / total,
                        rate / recording.get_sampling_frequency(), (total - done) / rate / 60)
                    logger.info(msg)
                    print(msg, flush=True)
        if os.path.getsize(partial) != total * n_ch * dtype.itemsize:
            raise IOError("%s has the wrong size after writing" % partial)
        os.replace(partial, path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    dt = time.perf_counter() - t_start
    logger.info("Wrote %s in %.1f min", path, dt / 60)
    print("Wrote %s in %.1f min" % (path, dt / 60), flush=True)
    return total, n_ch
