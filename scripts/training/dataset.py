import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset

class EEG_Image_Dataset(Dataset):

    def __init__(self, hdf5_file, window, output_type='label'):
        self.hdf5_file = hdf5_file
        self.window = window
        self.output_type = output_type
        self.index_to_frame = self._index_to_starting_frame(hdf5_file, window)

    def __len__(self):
        return len(self.index_to_frame)

    def __getitem__(self, index):

        starting_frame = self.index_to_frame[index]
        slice_ = slice(starting_frame, starting_frame + self.window)

        with h5py.File(self.hdf5_file, "r") as f:

            images = f['images'][slice_]
            label = f['labels'][slice_][0]
            
        if self.output_type == 'label':
            return images, label
        
        if self.output_type == 'last_frame':
            return images[:-1], images[-1].flatten()

    def _index_to_starting_frame(self, hdf5_file, window):

        with h5py.File(hdf5_file, "r") as f:

            df = pd.DataFrame({
                'sid': f['subject_id'][:], 
                'tid': f['trial_id'][:], 
                'fid': f['frame_id'][:]
                })

            # discard indices too close to the end of a trial
            frames_per_trial = df.groupby(['sid', 'tid']).apply(len)
            df['reverse_fid'] = df.apply(lambda row: frames_per_trial.loc[row.sid, row.tid] - row.fid - 1, axis=1)
            df['valid_index'] = df.reverse_fid >= window - 1
            df = df[df.valid_index]
            
            # mapping between item indices and valid starting frames 
            return dict(zip(np.arange(df.shape[0]), df.index))