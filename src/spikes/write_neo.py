from __future__ import print_function

import numpy as np

from neo.core import SpikeTrain, Unit, Segment, Block, ChannelIndex


def write_neo(old_block, new_spike_times, new_spike_waveforms):
    """Create a new neo.core.Block with new spiketrain data

    Params
    old_block (neo.core.Block)
        old block with thresholded, unsorted units
    spike_times (list of lists)
        electrode_idx ->
            array of spike arrival times (in seconds) for each putative unit

    Returns
    new_block
        New block with all segments and multiple units per electrode
        channel index for sorted units, waveforms
        And has audio
    """
    new_block = Block()

    channel_indexes = [channel_index
                       for channel_index in old_block.channel_indexes
                       if channel_index.name.startswith("Amp Elec")]

    # basically copy over all segments
    new_segments = [
            Segment(
                name=seg.name,
                description=seg.description,
                file_origin=seg.file_origin,
                file_datetime=seg.file_datetime,
                rec_datetime=seg.rec_datetime,
                index=seg.index,
                **seg.annotations
            ) for seg in old_block.segments]
    new_channel_index_units = dict((i, []) for i in range(len(channel_indexes)))

    new_audio_channel_index = ChannelIndex(name=old_block.channel_indexes[0].name, index=100)

    for i, old_segment in enumerate(old_block.segments):
        old_signal = old_segment.analogsignals[0]
        old_signal.segment = new_segments[i]
        old_signal.channel_index = new_audio_channel_index
        new_audio_channel_index.analogsignals.append(old_signal)
        new_segments[i].analogsignals.append(old_signal)

    iterator = enumerate(zip(new_spike_times, new_spike_waveforms, channel_indexes))
    for electrode_idx, (spike_times_list, spike_waveforms_list, old_channel_index) in iterator:
        if not spike_times_list:
            continue

        old_unsorted_unit = old_channel_index.units[0]

        print("Creating {} units for electrode {}".format(len(spike_times_list[0]), electrode_idx))

        for new_unit_idx in range(len(spike_times_list[0])):
            new_unit = Unit(name="e{}u{}".format(electrode_idx, new_unit_idx))

            for segment_idx, old_spiketrain in enumerate(old_unsorted_unit.spiketrains):
                if not len(spike_times_list):
                    continue

                spike_train = SpikeTrain(
                        spike_times_list[segment_idx][new_unit_idx],
                        t_start=old_spiketrain.t_start,
                        t_stop=old_spiketrain.t_stop,
                        units=old_spiketrain.units,
                        waveforms=np.array(spike_waveforms_list[segment_idx][new_unit_idx]),
                        # uhhh... how can we get the waveforms at this point?
                        # sampling_rate=old_spiketrain.sampling_rate
                )

                new_unit.spiketrains.append(spike_train)
                new_segments[segment_idx].spiketrains.append(spike_train)

                # add backwards links

                spike_train.segment = new_segments[segment_idx]
                spike_train.unit = new_unit

            new_channel_index_units[electrode_idx].append(new_unit)
            new_unit.channel_index = channel_index

        new_channel_index = ChannelIndex(name=old_channel_index.name, index=electrode_idx)
        new_channel_index.units += new_channel_index_units[electrode_idx]
        new_block.channel_indexes.append(new_channel_index)
    new_block.channel_indexes.append(new_audio_channel_index)

    new_block.segments += new_segments

    return new_block
