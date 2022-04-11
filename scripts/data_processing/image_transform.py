import numpy as np
import argparse
import yaml
import os
import shutil
import tqdm
import glob
import sys
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

sys.path.append('/home/lugeon/eeg_project/scripts')
from data_processing.aep import map_to_2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:   
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    transform_by_subject(**config)


def transform_by_subject(root_dir, output_dir, subject_pattern, e_locs_path, e_locs_read_format, gen_images_opts):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    electrode_locs = _read_electrode_locs(e_locs_path, e_locs_read_format)
    coord = map_to_2d(electrode_locs)

    print('\nCreate images for each subject...')

    for subject_freq_file in tqdm.tqdm(glob.glob(f'{root_dir}/{subject_pattern}'), ncols=70):

        subject_name = subject_freq_file.split('/')[-1].split('.')[0]
        subject_output_path = f'{output_dir}/{subject_name}'

        _save_images(subject_freq_file, subject_output_path, coord, gen_images_opts)

    print('Done.\n')


def _save_images(input_path, output_path, coord, gen_images_opts):

    images = {}
    with np.load(input_path) as signals:
    
        for sname in signals:

            signal = signals[sname]
            n_samples = signal.shape[0]
            signal = signal.reshape((n_samples, -1), order='F') # concatenate frequencies into single dim

            image = _gen_images(coord, signal, **gen_images_opts)
            iname = f'{sname.split(".")[0]}.img'
            images[iname] = image

    np.savez_compressed(output_path, **images)


def _read_electrode_locs(e_locs_path, read_format='scipy'):

    if read_format == 'scipy':
        locs = scipy.io.loadmat(e_locs_path)['locstemp']

    else: raise NotImplementedError(f'Format {read_format} is not implemented')
    return locs


def _gen_images(locs, features, n_gridpoints, normalize=False, edgeless=False, interpolation = 'cubic'):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    n_electrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    #assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / n_electrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * n_electrodes : n_electrodes * (c+1)])

    n_samples = features.shape[0]

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method=interpolation, fill_value=np.nan)
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    return np.swapaxes(np.asarray(temp_interp), 0, 1) 



if __name__ == '__main__':
    main()