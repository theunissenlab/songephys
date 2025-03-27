# now lets find onsets and offsets

from scipy.signal import find_peaks

# lets get all stims
import glob
from pathlib import Path

import soundfile as sf
import numpy as np
import pandas as pd
import luigi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import windows
from pynwb import NWBHDF5IO, NWBFile
from pynwb import validate
from pynwb.file import LabMetaData
from pynwb.base import TimeSeries, ProcessingModule#, #DynamicTable, 
from hdmf.common import DynamicTable, VectorData, VectorIndex, ElementIdentifiers

from strfpy.preprocSound import timefreq_raw

from .pipeline_utils import (
    LocalTask,
    RemoteTask,
    SiteParameter,
    parse_filelist,
    session_basename,
)
from .nwb_utils import (
    NWBProcessingTarget,
    CreateSiteNWB,
    AddDigitalEventsNWB,
    AddMotionNWB,
    AddAudioNWB,
    AddPyoperantTrials,
    AddLFPNWB,
)
# this is hard coded for now..
mbr_rends = pd.read_csv('/mnt/condor/Data/OperantEphys/XXXBlu121XF/Stimuli/Milbrook2018Renditions.csv')



def find_long_zero_segments(arr, min_length=10):
    # Find where the array is zero
    is_zero = (arr == 0)
    
    # Get the indices where values change
    changes = np.diff(is_zero.astype(int))
    start_idx = np.where(changes == 1)[0] + 1
    end_idx = np.where(changes == -1)[0] + 1
    
    # Handle edge cases where array starts or ends with zeros
    if is_zero[0]:
        start_idx = np.insert(start_idx, 0, 0)
    if is_zero[-1]:
        end_idx = np.append(end_idx, len(arr))
    
    # Filter segments by length and return as list of (start, end) tuples
    segments = [(start, end) for start, end in zip(start_idx, end_idx) 
               if end - start > min_length]
    
    return segments

def get_peaks_troughs(rendition_row, DBNOISE=50, minSound=25,derivativeThresh=0.2, respChunkLen=75, segmentBuffer=25):
    spectro = np.copy(rendition_row.tfrep)
    # first set the noise floor for the spectrogram
    dBMax = spectro.max()
    spectro[spectro < dBMax - DBNOISE] = dBMax - DBNOISE

    # and lets blur it a bit in 2d
    spectro = gaussian_filter1d(spectro, 1, axis=0)
    spectro = gaussian_filter1d(spectro, 1, axis=1)

    nFreqs = len(rendition_row.tfrep_freq)
    ampenv = np.mean(spectro, axis=0)
    ampfs = 1000

    wHann = windows.hann(
            21, sym=True
        )  # The 21 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann / sum(wHann)
    ampenv = np.convolve(ampenv, wHann, mode="same")
    ampenv[0:10] = ampenv[10]
    ampenv[-10:] = ampenv[-10]
    ampdev = ampenv[1:] - ampenv[0:-1]

    # Find peaks and troughs
    peakInd, peakVals = find_peaks(
        ampdev, height=derivativeThresh, distance=minSound
    )
    troughInd, troughVals = find_peaks(
        -ampdev, height=derivativeThresh, distance=minSound
    )

    # now we have our onsets and offsets, we return the spectrograms around those points

    # pad the spectrogram with zeros to make sure we have enough points for windowing
    padded_spect = np.pad(
        spectro,
        ((0, 0), (respChunkLen, respChunkLen)),
        "constant",
        constant_values=(dBMax - DBNOISE, dBMax - DBNOISE),
    )
    # now get the sliding window view
    spect_windows = np.lib.stride_tricks.sliding_window_view(
        padded_spect, respChunkLen, axis=1
    )

    # now we can get the spectrograms around the onsets and offsets
    onsets = spect_windows[:, peakInd+respChunkLen-segmentBuffer].swapaxes(0, 1)
    onset_inds = peakInd
    offsets = spect_windows[:, troughInd+respChunkLen-segmentBuffer].swapaxes(0, 1)
    offset_inds = troughInd

    df = pd.DataFrame({
        'type': ['onset']*len(onsets) + ['offset']*len(offsets),
        'spec': list(np.concatenate([onsets,offsets],axis=0)),
        'inds': np.concatenate([onset_inds,offset_inds]),
        'bird_name': [rendition_row.bird_name]*len(onsets) + [rendition_row.bird_name]*len(offsets),
        'rend': [rendition_row.rend]*len(onsets) + [rendition_row.rend]*len(offsets),
    })
    return df.sort_values(by='inds')

# now lets find onsets and offsets

from scipy.signal import find_peaks


def get_peaks_troughs(rendition_row, DBNOISE=50, minSound=25,derivativeThresh=0.2, respChunkLen=75, segmentBuffer=25):
    spectro = np.copy(rendition_row.tfrep)
    # first set the noise floor for the spectrogram
    dBMax = spectro.max()
    spectro[spectro < dBMax - DBNOISE] = dBMax - DBNOISE

    # and lets blur it a bit in 2d
    spectro = gaussian_filter1d(spectro, 1, axis=0)
    spectro = gaussian_filter1d(spectro, 1, axis=1)

    nFreqs = len(rendition_row.tfrep_freq)
    ampenv = np.mean(spectro, axis=0)
    ampfs = 1000

    wHann = windows.hann(
            21, sym=True
        )  # The 21 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann / sum(wHann)
    ampenv = np.convolve(ampenv, wHann, mode="same")
    ampenv[0:10] = ampenv[10]
    ampenv[-10:] = ampenv[-10]
    ampdev = ampenv[1:] - ampenv[0:-1]

    # Find peaks and troughs
    peakInd, peakVals = find_peaks(
        ampdev, height=derivativeThresh, distance=minSound
    )
    troughInd, troughVals = find_peaks(
        -ampdev, height=derivativeThresh, distance=minSound
    )

    # now we have our onsets and offsets, we return the spectrograms around those points

    # pad the spectrogram with zeros to make sure we have enough points for windowing
    padded_spect = np.pad(
        spectro,
        ((0, 0), (respChunkLen, respChunkLen)),
        "constant",
        constant_values=(dBMax - DBNOISE, dBMax - DBNOISE),
    )
    # now get the sliding window view
    spect_windows = np.lib.stride_tricks.sliding_window_view(
        padded_spect, respChunkLen, axis=1
    )

    # now we can get the spectrograms around the onsets and offsets
    onsets = spect_windows[:, peakInd+respChunkLen-segmentBuffer].swapaxes(0, 1)
    onset_inds = peakInd
    offsets = spect_windows[:, troughInd+respChunkLen-segmentBuffer].swapaxes(0, 1)
    offset_inds = troughInd

    df = pd.DataFrame({
        'type': ['onset']*len(onsets) + ['offset']*len(offsets),
        'spec': list(np.concatenate([onsets,offsets],axis=0)),
        'inds': np.concatenate([onset_inds,offset_inds]),
        'bird_name': [rendition_row.bird_name]*len(onsets) + [rendition_row.bird_name]*len(offsets),
        'rend': [rendition_row.rend]*len(onsets) + [rendition_row.rend]*len(offsets),
    })
    return df

def process_stim_dir(stim_dir, wavs_to_include=None):
    wavs = list(Path(stim_dir).rglob('*.wav'))
    rend_info_df = dict({
        'bird_folder': [],
        'stim_type': [],
        'bird_name': [],
        'rend': [],
        'fs': [],
        'n_samples': [],
        'data' : [],
        'tfrep': [],
        'tfrep_time': [],
        'tfrep_freq': [],
    })
    stim_rend_info_df = dict({
        'bird_folder': [],
        'stim_type': [],
        'bird_name': [],
        'stim_name': [],
        'rends': [],
        'rend_intervals': [],
        'fs': [],
        'n_samples': [],
        'data': [],
        'tfrep': [],
        'tfrep_time': [],
        'tfrep_freq': [],
    })

    used_names = []
    for wav in wavs:
        if wav.name in used_names:
            print("Duplicate name %s"%wav)
        used_names.append(wav.name)


    if wavs_to_include is not None:
        wavs = [w for w in wavs if w.name in wavs_to_include]
        
        # lets check that all wavs_to_include are included
        all_wavs_included = np.all([w in [wav.name for wav in wavs] for w in wavs_to_include])
        print("All wavs included: %s"%all_wavs_included)
        assert(all_wavs_included)

    for wav in wavs:
        relpath = wav.relative_to(stim_dir)
        bird_folder_name = relpath.parts[0]
        stim_name = relpath.parts[-1]

        if stim_name in stim_rend_info_df['stim_name']:
            print("Duplicate stim name %s, skipping"%stim_name)
            continue
        bird_name = stim_name.split('_')[3]
        rend_str = stim_name.split('_')[-1].split('.')[0]
        #print(bird_name, bird_folder_name, rend_str)

        
        # get the duration of the stim
        stim, fs = sf.read(wav)
        # lets convert it to uint16 if it is float
        if stim.dtype == np.float32 or stim.dtype == np.float64:
            stim = (stim*32767).astype(np.int16)

        duration = len(stim)/fs
        n_samples = len(stim)
        # get the time frequency representation
        stim_params = dict()
        stim_params['fband'] = 120
        stim_params['nstd'] = 6
        stim_params['high_freq'] = 8000
        stim_params['low_freq'] = 250
        stim_params['log'] = 1
        stim_params['stim_rate'] = 1000
        tfrep = timefreq_raw(stim, fs, 'ft', stim_params)
        spectro = tfrep['spec'].copy()
        tfrep_time = tfrep['t']
        tfrep_freq = tfrep['f']
        #tfrep_time = tfrep['time']
        # there are broadly three types of stim
        # 1. BOS
        if 'BOS' == bird_folder_name[:3]:
            stim_type = 'Colony'
            nrends = 1
            rend_ids = [int(rend_str)]
        # 2. Mbr
        elif 'Mbr' == bird_name[:3]:
            stim_type = 'Milbrook'
            possible_rends = mbr_rends[mbr_rends.bird_name == bird_name].rendition.values
            # find the rendition id that matches the beginning of the stim name
            rend_ids = []
            rend_str_tmp = rend_str
            for i in range(2):
                if rend_str_tmp == '':
                    break
                for pr in possible_rends:
                    str_pr = str(pr)
                    if str_pr == rend_str_tmp[:len(str_pr)]:
                        rend_ids.append(pr)
                        rend_str_tmp = rend_str_tmp[len(str_pr):]
                        break
                
            #print("Milbrook len %d"%len(rend_ids))
            nrends = len(rend_ids)
        else:
            stim_type = 'Operant'
            if len(rend_str) == 4:
                nrends = 2
                rend_ids = [int(rend_str[:2]), int(rend_str[2:])]
            elif len(rend_str) < 4:
                nrends = 1
                rend_ids = [int(rend_str)]
            elif len(rend_str) == 6:
                nrends = 3
                rend_ids = [int(rend_str[:2]), int(rend_str[2:4]), int(rend_str[4:])]
            else:
                raise ValueError('Unknown stim type %s'%stim_name)


        # now find the zero blocks
        if nrends == 1:
            rend = rend_ids[0]

            # find when the rend stops
            zero_blocks = find_long_zero_segments(stim, min_length=20)
            # take the last one
            if len(zero_blocks) == 0:
                stop = len(stim)
            else:
                stop = zero_blocks[-1][0]
            # get the stim
            stim_rend = stim[:stop]
            # get the time frequency representation
            tfrep_rend = timefreq_raw(stim_rend, fs,'ft', stim_params)
            spectro_rend = tfrep_rend['spec'].copy()
            tfrep_rend_time = tfrep_rend['t']
            tfrep_rend_freq = tfrep_rend['f']

            # store the info
            rend_info_df['bird_folder'].append(bird_folder_name)
            rend_info_df['stim_type'].append(stim_type)
            rend_info_df['bird_name'].append(bird_name)
            rend_info_df['rend'].append(rend)
            rend_info_df['fs'].append(fs)
            rend_info_df['n_samples'].append(stop)
            rend_info_df['data'].append(stim_rend)
            rend_info_df['tfrep'].append(spectro_rend)
            rend_info_df['tfrep_time'].append(tfrep_rend_time)
            rend_info_df['tfrep_freq'].append(tfrep_rend_freq)

            # store the rend mapping info
            stim_rend_info_df['bird_folder'].append(bird_folder_name)
            stim_rend_info_df['stim_type'].append(stim_type)
            stim_rend_info_df['bird_name'].append(bird_name)
            stim_rend_info_df['stim_name'].append(stim_name)
            stim_rend_info_df['rends'].append([rend])
            stim_rend_info_df['rend_intervals'].append([(0, stop)])
            stim_rend_info_df['fs'].append(fs)
            stim_rend_info_df['n_samples'].append(n_samples)
            stim_rend_info_df['data'].append(stim)
            stim_rend_info_df['tfrep'].append(spectro)
            stim_rend_info_df['tfrep_time'].append(tfrep_time)
            stim_rend_info_df['tfrep_freq'].append(tfrep_freq)
        else:
            print(wav)
            zero_blocks = find_long_zero_segments(stim, min_length=20)
            print(zero_blocks)
            # if len(zero_blocks)+1 != nrends:
            #     print("Zero blocks %d, nrends %d"%(len(zero_blocks), nrends))
            #     # taking top nrends-1
            #     zb_sizes = [zb[1]-zb[0] for zb in zero_blocks]
            #     # take the largest nrends-1 zero blocks
            #     zero_blocks = [zb for zb in zero_blocks if zb[1]-zb[0] in sorted(zb_sizes, reverse=True)[:nrends-1]]
                
            #     # and restort them by zb[0]
            #     zero_blocks = sorted(zero_blocks, key=lambda x: x[0])
            #     print(zero_blocks)
            assert(len(zero_blocks) +1 == nrends)
            rend_start_idx = []
            rend_stop_idx = []
            for rend_ind in range(nrends):
                if rend_ind == 0:
                    start = 0
                else:
                    start = zero_blocks[rend_ind-1][1]
                if rend_ind == nrends - 1:
                    stop = len(stim)
                else:
                    stop = zero_blocks[rend_ind][0]
                rend_start_idx.append(start)
                rend_stop_idx.append(stop)
                # get the stim
                stim_rend = stim[start:stop]
                rend_id = rend_ids[rend_ind]
                # get the time frequency representation
                tfrep_rend = timefreq_raw(stim_rend, fs,'ft', stim_params)
                spectro_rend = tfrep_rend['spec'].copy()
                tfrep_rend_time = tfrep_rend['t']
                tfrep_rend_freq = tfrep_rend['f']

                # store the info
                rend_info_df['bird_folder'].append(bird_folder_name)
                rend_info_df['stim_type'].append(stim_type)
                rend_info_df['bird_name'].append(bird_name)
                rend_info_df['rend'].append(rend_id)
                rend_info_df['fs'].append(fs)
                rend_info_df['n_samples'].append(stop - start)
                rend_info_df['data'].append(stim_rend)
                rend_info_df['tfrep'].append(spectro_rend)
                rend_info_df['tfrep_time'].append(tfrep_rend_time)
                rend_info_df['tfrep_freq'].append(tfrep_rend_freq)
            stim_rend_info_df['bird_folder'].append(bird_folder_name)
            stim_rend_info_df['stim_type'].append(stim_type)
            stim_rend_info_df['bird_name'].append(bird_name)
            stim_rend_info_df['stim_name'].append(stim_name)
            stim_rend_info_df['rends'].append(rend_ids)
            stim_rend_info_df['rend_intervals'].append(list(zip(rend_start_idx, rend_stop_idx)))
            stim_rend_info_df['fs'].append(fs)
            stim_rend_info_df['n_samples'].append(n_samples)
            stim_rend_info_df['data'].append(stim)
            stim_rend_info_df['tfrep'].append(spectro)
            stim_rend_info_df['tfrep_time'].append(tfrep_time)
            stim_rend_info_df['tfrep_freq'].append(tfrep_freq)
    stim_rend_info_df = pd.DataFrame(stim_rend_info_df)
    rend_info_df = pd.DataFrame(rend_info_df)
    # get rid of duplicates
    rend_info_df = rend_info_df.drop_duplicates(subset=['rend', 'bird_name'])
    return stim_rend_info_df, rend_info_df


def create_dynamic_table_from_df(df, name='my_table', description=''):
    # Create the base DynamicTable
    
    columns = []
    
    for column in df.columns:
        data = df[column].values
        
        # Check if the column contains ragged arrays
        if isinstance(data[0], (list, np.ndarray)):
            # Flatten the ragged arrays and create index pointers
            flat_data = []
            indices = []
            current_index = 0
            
            for arr in data:
                arr = np.asarray(arr)
                print(arr.shape)
                flat_data.extend(arr)
                indices.append(current_index)
                current_index += len(arr)
            
            # Create VectorData for the flattened data
            vector_data = VectorData(
                name=f'{column}_data',
                description=f'Flattened data for {column}',
                data=flat_data
            )
            
            # Create VectorIndex to map back to ragged structure
            vector_index = VectorIndex(
                name=column,
                data=indices,
                target=vector_data
            )
            
            columns.append(vector_data)
            columns.append(vector_index)
            
        else:
            # Handle regular columns normally
            vector_data = VectorData(
                name=column,
                description=f'Column {column} from dataframe',
                data=data
            )
            columns.append(vector_data)
    table = DynamicTable(
        name=name,
        description=description,
        columns = columns
    )
    return table

# lets make a luigi task that generates these dataframes
class AddStimSpectrogramsNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir),
            #'pyoperant_trials': ParsePyoperantTrials(site_dir=self.site_dir),
        }
    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'stimuli_info', 'rend_info','stim_info','rend_onsets_offsets')
    
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()

            op_trials = nwbfile.intervals['operant_trials'].to_dataframe()
            pb_trials = nwbfile.intervals['playback_trials'].to_dataframe()
            all_trials = pd.concat([op_trials, pb_trials])
            all_stim = all_trials['stimuli_name'].unique()
            
            stim_dir = self.stim_dir
            stim_rend_info_df, rend_info_df = process_stim_dir(stim_dir, wavs_to_include=all_stim)

            if 'stimuli' not in nwbfile.processing:
                nwbfile.create_processing_module(
                    name="stimuli_info", description="Processed stimuli, including specgrams and onsets/offsets"
                )
            if 'stim_spectrograms' not in nwbfile.processing:
                nwbfile.create_processing_module(
                    name="stimuli_spectrograms", description="Spectrograms of the stimuli"
                )
            
            # lets store the spectrograms of the full stimuli
            for ix, row in stim_rend_info_df.iterrows():
                stim_name = row.stim_name
                spectro = row.tfrep
                spectro_time = row.tfrep_time
                spectro_freq = row.tfrep_freq
                # lets make a time series
                ts = TimeSeries(
                    name=stim_name,
                    description='Spectrogram of the full stimulus evenly spaced freq from %s to %s'%(spectro_freq[0], spectro_freq[-1]),
                    data=spectro.T,
                    timestamps=spectro_time,
                    unit='dB'
                )  
                nwbfile.processing['stimuli_spectrograms'].add_data_interface(ts)
            

            
            # now lets find the onsets and offsets
            print("Finding onsets and offsets")
            all_onoff_dfs = []
            for ix, rend in rend_info_df.iterrows():
                onoff_df = get_peaks_troughs(rend)
                onoff_df['rendition_index'] = ix
                all_onoff_dfs.append(onoff_df)
            all_onoff_df = pd.concat(all_onoff_dfs)
            
            
            rend_info_df.drop(columns=['data', 'tfrep', 'tfrep_time', 'tfrep_freq'], inplace=True)
            stim_rend_info_df.drop(columns=['rends','rend_intervals','data', 'tfrep', 'tfrep_time', 'tfrep_freq'], inplace=True)
            rend_info_table = DynamicTable.from_dataframe(rend_info_df, name='rend_info')
            stim_rend_info_table = DynamicTable.from_dataframe(stim_rend_info_df,name='stim_info')
            nwbfile.processing['stimuli_info'].add(rend_info_table)
            nwbfile.processing['stimuli_info'].add(stim_rend_info_table)
            all_onoff_df.drop(columns=['spec'], inplace=True)
            all_onoff_table = DynamicTable.from_dataframe(all_onoff_df, name='rend_onsets_offsets')
            nwbfile.processing['stimuli_info'].add(all_onoff_table)
            nwb_io.write(nwbfile)



class AddStimuliInfoNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir),
        }
    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'stimuli', 'rend_info','stim_info','rend_onsets_offsets')

    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()

            op_trials = nwbfile.intervals['operant_trials'].to_dataframe()
            pb_trials = nwbfile.intervals['playback_trials'].to_dataframe()
            all_trials = pd.concat([op_trials, pb_trials])
            all_stim = all_trials['stimuli_name'].unique()
            
            stim_dir = self.stim_dir
            stim_rend_info_df, rend_info_df = process_stim_dir(stim_dir, wavs_to_include=all_stim)
            rend_info_table = create_dynamic_table_from_df(rend_info_df, name='rend_info', description='Rendition info')
            stim_rend_info_table = create_dynamic_table_from_df(stim_rend_info_df, name='stim_info', description='Stimulus rendition info')

            # ragged_arrays = ['data', 'tfrep', 'tfrep_time', 'tfrep_freq']
            # rend_info_nonragged = rend_info_df.drop(columns=ragged_arrays)
            # rend_info_table = DynamicTable.from_dataframe(rend_info_nonragged, name='rend_info')
            # for ra in ragged_arrays:
            #     rend_info_table.add_column(name=ra, description='Ragged array', data=rend_info_df[ra].tolist(), index=True)

            # ragged_arrays = ['rends','rend_intervals','data', 'tfrep', 'tfrep_time', 'tfrep_freq']
            # stim_rend_info_nonragged = stim_rend_info_df.drop(columns=ragged_arrays)
            # stim_rend_info_table = DynamicTable.from_dataframe(stim_rend_info_nonragged,name='stim_info')
            # for ra in ragged_arrays:
            #     stim_rend_info_table.add_column(name=ra, description='Ragged array', data=stim_rend_info_df[ra].tolist(), index=True)

            

            # rend_info_table = DynamicTable(
            #     name='rend_info',
            #     description='Rendition info',
            # )
            # rend_info_table.add_column(name='bird_folder', description='Bird folder name', data=rend_info_df['bird_folder'].tolist())
            # rend_info_table.add_column(name='stim_type', description='Stimulus type', data=rend_info_df['stim_type'].tolist())
            # rend_info_table.add_column(name='bird_name', description='Bird name', data=rend_info_df['bird_name'].tolist())
            # rend_info_table.add_column(name='rend', description='Rendition id', data=rend_info_df['rend'].tolist())
            # rend_info_table.add_column(name='fs', description='Sampling frequency', data=rend_info_df['fs'].tolist())
            # rend_info_table.add_column(name='n_samples', description='Number of samples', data=rend_info_df['n_samples'].tolist())
            # rend_info_table.add_column(name='data', description='Stimulus data', data=rend_info_df['data'].tolist(), index=True)
            # rend_info_table.add_column(name='tfrep', description='Time frequency representation', data=rend_info_df['tfrep'].tolist(), index=True)
            # rend_info_table.add_column(name='tfrep_time', description='Time frequency representation time', data=rend_info_df['tfrep_time'].tolist(), index=True)
            # rend_info_table.add_column(name='tfrep_freq', description='Time frequency representation frequency', data=rend_info_df['tfrep_freq'].tolist(), index=True)
            
            # stim_rend_info_table = DynamicTable(
            #     name='stim_info',
            #     description='Stimulus rendition info',
            # )
            # stim_rend_info_table.add_column(name='bird_folder', description='Bird folder name', data=stim_rend_info_df['bird_folder'].tolist())
            # stim_rend_info_table.add_column(name='stim_type', description='Stimulus type', data=stim_rend_info_df['stim_type'].tolist())
            # stim_rend_info_table.add_column(name='bird_name', description='Bird name', data=stim_rend_info_df['bird_name'].tolist())
            # stim_rend_info_table.add_column(name='stim_name', description='Stimulus name', data=stim_rend_info_df['stim_name'].tolist())
            # stim_rend_info_table.add_column(name='rends', description='Rendition ids', data=stim_rend_info_df['rends'].tolist())
            # stim_rend_info_table.add_column(name='rend_intervals', description='Rendition intervals', data=stim_rend_info_df['rend_intervals'].tolist())
            # stim_rend_info_table.add_column(name='fs', description='Sampling frequency', data=stim_rend_info_df['fs'].tolist())
            # stim_rend_info_table.add_column(name='n_samples', description='Number of samples', data=stim_rend_info_df['n_samples'].tolist())
            # stim_rend_info_table.add_column(name='data', description='Stimulus data', data=stim_rend_info_df['data'].tolist(), index=True)
            # stim_rend_info_table.add_column(name='tfrep', description='Time frequency representation', data=stim_rend_info_df['tfrep'].tolist(), index=True)
            # stim_rend_info_table.add_column(name='tfrep_time', description='Time frequency representation time', data=stim_rend_info_df['tfrep_time'].tolist(), index=True)
            # stim_rend_info_table.add_column(name='tfrep_freq', description='Time frequency representation frequency', data=stim_rend_info_df['tfrep_freq'].tolist(), index=True)


            if 'stimuli' not in nwbfile.processing:
                nwbfile.create_processing_module(
                    name="stimuli", description="Processed stimuli, including specgrams and onsets/offsets"
                )

            nwbfile.processing['stimuli'].add(rend_info_table)
            nwbfile.processing['stimuli'].add(stim_rend_info_table)

            # now lets find the onsets and offsets
            print("Finding onsets and offsets")
            all_onoff_dfs = []
            for ix, rend in rend_info_df.iterrows():
                onoff_df = get_peaks_troughs(rend)
                onoff_df['rendition_index'] = ix
                all_onoff_dfs.append(onoff_df)
            all_onoff_df = pd.concat(all_onoff_dfs)
            all_onoff_table = create_dynamic_table_from_df(all_onoff_df, name='rend_onsets_offsets', description='Rendition onsets and offsets')
            # ragged_arrays = ['spec']
            # all_onoff_nonragged = all_onoff_df.drop(columns=ragged_arrays)
            # all_onoff_table = DynamicTable.from_dataframe(all_onoff_nonragged,name='rend_onsets_offsets')
            # for ra in ragged_arrays:
            #     all_onoff_table.add_column(name=ra, description='Ragged array', data=all_onoff_df[ra].tolist(), index=True)
            # all_onoff_table = DynamicTable(
            #     name='rend_onsets_offsets',
            #     description='Rendition onsets and offsets'
            # )
            # all_onoff_table.add_column(name='type', description='Type of event (onset/offset)', data=all_onoff_df['type'].tolist())
            # all_onoff_table.add_column(name='spec', description='Spectrogram', data=all_onoff_df['spec'].tolist(), index=True)
            # all_onoff_table.add_column(name='inds', description='Indices of event', data=all_onoff_df['inds'].tolist())
            # all_onoff_table.add_column(name='bird_name', description='Bird name', data=all_onoff_df['bird_name'].tolist())
            # all_onoff_table.add_column(name='rend', description='Rendition id', data=all_onoff_df['rend'].tolist())


            nwbfile.processing['stimuli'].add(all_onoff_table)

            # now lets write the nwb file
            nwb_io.write(nwbfile)

            # nwb_write_path = self.input()['site_nwb'].path.split('.nwb')[0] + '_stimuli.nwb'
            # nwb_iow = NWBHDF5IO(nwb_write_path, 'w')
            # nwb_iow.write(nwbfile)


        
class GenerateBaseNWB(luigi.WrapperTask):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'digital': AddDigitalEventsNWB(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir),
            'audio': AddAudioNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir),
            'spectrograms': AddStimSpectrogramsNWB(site_dir=self.site_dir),
        }