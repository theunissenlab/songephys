import numpy as np
from twpca import TWPCA
from twpca.regularizers import curvature


class WindowedTimeWarp(object):
    def __init__(self, fs=25000.0, **twpca_kwargs):
        self._transform_array_fns = []
        self._transform_timestamp_fns = []
        
        self.fs = fs
        self.twpca_kwargs = twpca_kwargs
        
    def fit_transform(self, data, window, overlap, **kwargs):
        self._transform_array_fns = []
        self._transform_timestamp_fns = []

        data = np.copy(data)
        n = data.shape[1]

        index = 0
        while index < (n - window):
            model = TWPCA(**self.twpca_kwargs)
            model.fit(data[:, index:index + window], **kwargs)
            
            transformations = []
            
            fitted = model.transform(data[:, index:index + window])
            
            for seg_idx, seg_signal in enumerate(fitted):
                warp_array = model.params["warp"][seg_idx]
                
                if warp_array[-1] > window:
                    shift_from_slice = slice(index + np.argmax(warp_array >= window), None)
                    shift_to = index + window
                    transform_from_slice = slice(None, None)

                elif warp_array[-1] < window:
                    last_index_rel = int(np.floor(warp_array[-1]))
                    last_index = index + last_index_rel
                    shift_from_slice = slice(index + window, None)
                    shift_to = last_index
                    transform_from_slice = slice(None, last_index_rel)
                else:
                    data[i, index:index + window] = signal
                    shift_from_slice = slice(index + window, None)
                    shift_to = index + window
                    transform_from_slice = slice(None, None)
                
                transformations.append((shift_from_slice, shift_to, transform_from_slice))

            def _transform_array(data_array, index, transformations, model):
                output_array = np.copy(data_array)
                fitted = model.transform(data_array[:, index:index + window])
                
                for seg_idx, (seg_signal, transform_params) in enumerate(zip(fitted, transformations)):
                    shift_from_slice, shift_to, transform_from_slice = transform_params
                    
                    shift_from = data_array[seg_idx, shift_from_slice]
                    space_remaining = output_array.shape[1] - shift_to
                    output_array[seg_idx, shift_to:shift_to + shift_from.shape[0]] = shift_from[:space_remaining]
                    
                    transform_from = seg_signal[transform_from_slice]
                    output_array[seg_idx, index:index + transform_from.shape[0]] = transform_from
                    
                return output_array
            
            def _transform_timestamp(timestamp, seg_idx, index, transformations, warp_params):
                timestamp_in_samples = timestamp * self.fs
                timestamp_index = int(np.floor(timestamp_in_samples))
                shift_from_slice, shift_to, transform_from_slice = transformations[seg_idx]
                if timestamp_in_samples < index:
                    return timestamp
                elif index <= timestamp_index < shift_from_slice.start - 1:
                    frac_over = timestamp_in_samples - timestamp_index
                    interpolated_index = (
                            (1.0 - frac_over) * warp_params[seg_idx, timestamp_index - index] + 
                            frac_over * warp_params[seg_idx, timestamp_index - index + 1]
                    )
                    return (index + interpolated_index) / self.fs
                elif shift_from_slice.start <= timestamp_index:
                    return timestamp + ((shift_to - shift_from_slice.start) / self.fs)
                
            warp_params = model.params["warp"]
            data = _transform_array(data, index, transformations, model)
            self._transform_array_fns.append((index, _transform_array, tuple(transformations), model))
            self._transform_timestamp_fns.append((index, _transform_timestamp, tuple(transformations), warp_params))
            
            index = index + window - overlap
        
        return data
    
    def transform(self, data_array):
        result = np.copy(data_array)
        for i, transform_fn, slices, model in self._transform_array_fns:
            result = transform_fn(result, i, slices, model)
        return result
    
    def convert_time(self, timestamp, seg_idx):
        result = timestamp
        for i, transform_fn, slices, warp_params in self._transform_timestamp_fns:
            result = transform_fn(result, seg_idx, i, slices, warp_params)
        return result
    
    def convert_times(self, timestamps):
        """
        timestamps must be a list of lists (inner lists are spike times)
        """
        return np.array([
            [self.convert_time(t, i) for t in spike_times]
            for i, spike_times in enumerate(timestamps)
        ])
        
