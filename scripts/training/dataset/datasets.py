import numpy as np
import pandas as pd
import h5py
import random
from typing import Dict, List, Union, Any
from torch.utils.data import Dataset

class EEG_Image_Batch_Dataset(Dataset):
    """
    A dataset that generates images in batches. Data is loaded in the memory for reading speed-up.
    """

    def __init__(self, 
                 hdf5_file: str, 
                 window: int, 
                 slide: int, 
                 shuffle: bool,
                 batch_size: Union[int, None], 
                 transforms: Union[List[Any], None] = None,
                 return_metadata: bool = False,
                 exclude_subject: List[int] = [],
                 output_type: str = 'label',
                 next_frame_index: int = None):
        """
        Args:
            hdf5_file (str): Path to the hdf5 file that contains the data.
            window (int): Number of stacked images (video) in a sample.
            slide (int): Number of frames between the starting frames of two consecutive videos.
            shuffle (bool): Shuffle the data once before creating the batches.
            batch_size (int): Number of samples in a batch. If set to None, get all the data in a single batch.
            return_metadata (bool): False
            exclude_subject (List[int]): Indices of subject to exclude from the dataset. Defaults to empty.
            output_type (str, optional): Can be set to {none, label, next_frame, transform}. Defaults to 'label'.
            next_frame_index (int): Index of the next frame to output (1 correspond to the frame just after the window)
        """

        random.seed(0)
        
        assert window > 0, 'Window must be strictly positive'
        assert slide > 0, 'Slide must be strictly positive'
        assert output_type in {'none', 'label', 'next_frame', 'transform'}, (
            'Output_type must one of {none, label, next_frame, transform}'
            )
        assert transforms is None or len(transforms) == 2, 'Transforms must be None or have size 2'
        
        if output_type == 'transform':   
            assert transforms is not None, (
                'If output_type is "transform", transforms argument should not be None')
            
        if output_type == 'next_frame':
            assert next_frame_index is not None, (
                'If output_type is "next_frame", "next_frame_index" should not be None'
            )
        
        
        super(EEG_Image_Batch_Dataset, self).__init__()
        
        with h5py.File(hdf5_file, "r") as f:

            # load in memory
            self.images = f['images'][:]
            self.labels = f['labels'][:]
            self.subject_id = f['subject_id'][:]
            self.trial_id = f['trial_id'][:]
            self.frame_id = f['frame_id'][:]
            self.sleep_stage = f['sleep_stage'][:]
            self.elapsed_time = f['elapsed_time'][:]
        
        self.hdf5_file = hdf5_file
        self.window = window
        self.next_frame_index = next_frame_index
        self.slide = slide
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transforms = transforms
        self.output_type = output_type
        self.return_metadata = return_metadata
        self.exclude_subject = exclude_subject
        self.starting_frames = self._get_starting_frames()
        self.batch_to_frames = self._get_batch_to_starting_frames()
        

    def __len__(self):
        return len(self.batch_to_frames)
    

    def __getitem__(self, index: int):
        
        images, labels, frame_indices, this_batch_size = self._select_data(index)

        if self.output_type == 'none':
            batch = (images, # all frames
                    np.array([])) # no output
            
        if self.output_type == 'label':
            batch = (images, # all frames
                     labels[:, 0].astype(int)) # one label per video
        
        if self.output_type == 'next_frame':
            batch =  (images[:, :-1], # all frames except last one
                      images[:, -1].reshape(this_batch_size, -1)) # next frame
            
        if self.output_type == 'transform':
            batch = (self.transforms[0]['f'](images, **self.transforms[0]['kwargs']), 
                     self.transforms[1]['f'](images, **self.transforms[1]['kwargs'])) 
            
        if self.return_metadata: # add subjects ID and sleep stages in the batch
            
            nb_frames = self.window + 1 if self.output_type == 'next_frame' else self.window
            
            subject_id = self.subject_id[frame_indices]
            subject_id = subject_id.reshape(this_batch_size, nb_frames) # reshape in batch
            subject_id = subject_id[:, 0].astype(int) # one subject ID per video
            
            sleep_stage = self.sleep_stage[frame_indices]
            sleep_stage = sleep_stage.reshape(this_batch_size, nb_frames)
            sleep_stage = sleep_stage[:, 0].astype(int)
            
            trial_id = self.trial_id[frame_indices]
            trial_id = trial_id.reshape(this_batch_size, nb_frames)
            trial_id = trial_id[:, 0].astype(int)
            
            frame_id = self.frame_id[frame_indices]
            frame_id = frame_id.reshape(this_batch_size, nb_frames)
            frame_id = frame_id[:, 0].astype(int)
            
            etime = self.elapsed_time[frame_indices]
            etime = etime.reshape(this_batch_size, nb_frames)
            etime = etime[:, 0].astype(int)
            
            batch = batch + ( (subject_id, trial_id, frame_id, sleep_stage, etime), )
            
        return batch
            
    def set_exlude_subjects(self, new_subjects: List[int]):
        self.exclude_subject = new_subjects
        self.starting_frames = self._get_starting_frames()
        self.batch_to_frames = self._get_batch_to_starting_frames() # re-compute batches
        
        return self
    
    def reshuffle(self):
        self.batch_to_frames = self._get_batch_to_starting_frames() # re-compute batches
        
        return self
    
    def _select_data(self, batch_index: int):
        
        # starting frames of this batch
        batch_starting_frames = self.batch_to_frames[batch_index]
        this_batch_size = len(batch_starting_frames)
        
        # all the frames in this batch 
        # select 'window' frames after the starting frame
        frame_indices = []
        for s in batch_starting_frames:
            ix = np.arange(self.window)
            if self.output_type == 'next_frame': # add the frame at the 'next_frame index'
                ix = np.append(ix, self.window + self.next_frame_index - 1)
            frame_indices.extend(s + ix)
            
        # get flattened frames
        images = self.images[frame_indices] # (batch_size * window + 1) x 5 x 32 x 32
        labels = self.labels[frame_indices]
            
        # reshape frames into videos  
        nb_frames = self.window + 1 if self.output_type == 'next_frame' else self.window
        shape = [this_batch_size, nb_frames] + list(images.shape)[1:]
        images = images.reshape(shape)
        labels = labels.reshape(this_batch_size, nb_frames)
        
        return images, labels, frame_indices, this_batch_size

    def _get_batch_to_starting_frames(self) -> Dict[int, List[int]]:
        """ Create a mapping between a batch index and the starting frames in the batch.

        Returns:
            Dict[int, List[int]]: Mapping between batch index to starting frames.
        """
        
        starting_frames = self.starting_frames.copy()
        
        if self.shuffle:
            random.shuffle(starting_frames)
            
        if not self.batch_size: self.batch_size = len(starting_frames)
            
        batch_index = np.arange(len(starting_frames)) // self.batch_size 
        df_batch = pd.DataFrame({'frame_ix': starting_frames, 
                                 'batch_ix': batch_index })
        
        batch_to_frames = df_batch.groupby('batch_ix').frame_ix.apply(list).to_dict()
        
        return batch_to_frames
        

    def _get_starting_frames(self) -> List[int]:
        """ Find valid starting frames of the videos, given window and slide.

        Returns:
            List[int]: Indices of valid starting frames in the dataset.
        """

        df = pd.DataFrame({
            'sid': self.subject_id, 
            'tid': self.trial_id, 
            'fid': self.frame_id,
            'indexing': np.arange(len(self.frame_id))
            })
        
        # remove exluded subjects
        df = df[df.sid.apply(lambda subject: subject not in self.exclude_subject)]

        # trials length
        frames_per_trial = df.groupby(['sid', 'tid']) \
            .apply(len).rename('trial_len') \
            .to_frame() \
            .reset_index()
            
        # discard non valid starting frame, given slide and window
        df = df.merge(frames_per_trial, how='left')
        df['start_frame'] = df.fid % self.slide == 0 # keep modulo of slide
        
        # discard frames at the end of trial
        if self.output_type == 'next_frame':
            limit = df.trial_len + 1 - self.window - self.next_frame_index
        else:
            limit = df.trial_len + 1 - self.window
        df['valid_start'] = df.fid <= limit 
            
        df = df[df.start_frame & df.valid_start]
        
        # indices of valid starting frames 
        return df.indexing.to_list()
        