import numpy as np
import pandas as pd
import h5py
import glob
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    transform_by_subject(**config)


def transform_by_subject(root_dir, output_file, labels, subject_pattern='*'):

    print('\nCreate an HDF5 dataset with images...')

    # mapping between trials and labels
    labels = pd.read_csv(labels, sep=' ', names=['trial', 'label'])
    labels_map = dict(zip(labels.trial, labels.label))

    # retrieve numpy archive containing the images, and the size of the dataset
    subjects = glob.glob(f'{root_dir}/{subject_pattern}')
    n_frames, n_gridpoints, n_channels = _get_dataset_shape(subjects, labels_map)

    # create HDF5 dataset
    f = h5py.File(output_file, 'w')
    dset_images = f.create_dataset('images', (n_frames, n_channels, n_gridpoints, n_gridpoints))
    dset_sid = f.create_dataset('subject_id', (n_frames,))
    dset_tid = f.create_dataset('trial_id', (n_frames,))
    dset_fid = f.create_dataset('frame_id', (n_frames,))
    dset_labels = f.create_dataset('labels', (n_frames,))
    dset_snames = f.create_dataset('subject_name', (len(subjects),), dtype='S04')

    # for each subject numpy archive
    sequence_counter = 0
    for subject_id, subject in enumerate(subjects):

        subject_name = subject.split('/')[-1].split('.')[0]
        dset_snames[subject_id] = subject_name
        
        with np.load(subject) as images:

            # keep only trials in the labels mapping
            trials = [k for k in images.keys() if k.split('.')[0] in labels_map]

            # for each trial, fill the dataset
            for trial_id, trial in enumerate(trials):

                sequence = images[trial]
                sequence_length = sequence.shape[0]
                label = labels_map[trial.split('.')[0]]
                slice_ = slice(sequence_counter, sequence_counter + sequence_length)

                dset_images[slice_] = sequence
                dset_labels[slice_] = sequence_length * [label]
                dset_sid[slice_] = sequence_length * [subject_id]
                dset_tid[slice_] = sequence_length * [trial_id]
                dset_fid[slice_] = np.arange(sequence_length)

                sequence_counter += sequence.shape[0]

    f.close()

    print('Done.\n')


def _get_dataset_shape(subjects, labels_map):

    n_frames, n_gridpoints, n_channels = 0, 0, 0
    for subject_id, subject in enumerate(subjects):

        with np.load(subject) as images:

            trials = [k for k in images.keys() if k.split('.')[0] in labels_map]

            n_frames += sum([images[trial].shape[0] for trial in trials])
            if subject_id == 0:
                _, n_channels, n_gridpoints, _ = next(iter(images.values())).shape

    return n_frames, n_gridpoints, n_channels


if __name__ == '__main__':
    main()
