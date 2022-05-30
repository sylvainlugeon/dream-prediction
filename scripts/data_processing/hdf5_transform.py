from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
import h5py
import glob
import argparse
import yaml
import sys
import tqdm

sys.path.append('/home/lugeon/eeg_project/scripts')
from interaction.interaction import ask_for_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    if ask_for_config(config): pass
    else: return

    transform_by_subject(**config)


def transform_by_subject(root_dir, output_file, metadata, subject_pattern='*'):

    print('\nCreate an HDF5 dataset with images...')

    # mapping between trials and labels
    df_metadata = pd.read_csv(metadata, 
                              sep=' ', 
                              header=0, 
                              names=['trial', 'label', 'sleep_cycle', 'sleep_stage', 'elapsed_time'])
    
    labels_map = dict(zip(df_metadata.trial, df_metadata.label))
    cycle_map = dict(zip(df_metadata.trial, df_metadata.sleep_cycle))
    stage_map = dict(zip(df_metadata.trial, df_metadata.sleep_stage))
    etime_map = dict(zip(df_metadata.trial, df_metadata.elapsed_time))

    print('Compute size of the dataset...')

    # retrieve numpy archive containing the images, and the size of the dataset
    subjects = glob.glob(f'{root_dir}/{subject_pattern}')
    n_frames, n_gridpoints, n_channels = _get_dataset_shape(subjects, labels_map)

    # create HDF5 dataset
    f = h5py.File(output_file, 'w')
    dset_images = f.create_dataset('images', (n_frames, n_channels, n_gridpoints, n_gridpoints))
    dset_sid = f.create_dataset('subject_id', (n_frames,))
    dset_tid = f.create_dataset('trial_id', (n_frames,))
    dset_ss = f.create_dataset('sleep_stage', (n_frames,))
    dset_sc = f.create_dataset('sleep_cycle', (n_frames,))
    dset_fid = f.create_dataset('frame_id', (n_frames,))
    dset_labels = f.create_dataset('labels', (n_frames,))
    dset_etime = f.create_dataset('elapsed_time', (n_frames,))
    dset_snames = f.create_dataset('subject_name', (len(subjects),), dtype='S04')
    
    print('Fill the dataset...')

    # for each subject numpy archive
    sequence_counter = 0
    for subject_id, subject in tqdm.tqdm(enumerate(subjects), total=len(subjects), ncols=70,):

        subject_name = subject.split('/')[-1].split('.')[0]
        dset_snames[subject_id] = subject_name
        
        with np.load(subject) as images:

            # keep only trials in the labels mapping
            trials = [k for k in images.keys() if k.split('.')[0] in labels_map]

            # for each trial, fill the dataset
            for trial_id, trial in enumerate(trials):

                sequence = images[trial]
                sequence_length = sequence.shape[0]
                
                trial_key = trial.split('.')[0]
                label = labels_map[trial_key]
                sleep_stage = stage_map[trial_key]
                sleep_cycle = cycle_map[trial_key]
                etime = etime_map[trial_key]
                
                slice_ = slice(sequence_counter, sequence_counter + sequence_length)
                
                dset_images[slice_] = sequence
                dset_labels[slice_] = sequence_length * [label]
                dset_sid[slice_] = sequence_length * [subject_id]
                dset_tid[slice_] = sequence_length * [trial_id]
                dset_fid[slice_] = np.arange(sequence_length)
                dset_ss[slice_] = sequence_length * [sleep_stage]
                dset_sc[slice_] = sequence_length * [sleep_cycle]
                dset_etime[slice_] = sequence_length * [etime]

                sequence_counter += sequence.shape[0]

    f.close()

    print('Done.\n')
    print(f'Dataset {output_file} created with {n_frames} frames.\n')


def _get_dataset_shape(subjects, labels_map):

    n_frames, n_gridpoints, n_channels = 0, 0, 0
    for subject_id, subject in tqdm.tqdm(enumerate(subjects), total=len(subjects), ncols=70):

        with np.load(subject) as images:

            trials = [k for k in images.keys() if k.split('.')[0] in labels_map]

            n_frames += sum([images[trial].shape[0] for trial in trials])
            if subject_id == 0:
                _, n_channels, n_gridpoints, _ = next(iter(images.values())).shape

    return n_frames, n_gridpoints, n_channels


if __name__ == '__main__':
    main()
