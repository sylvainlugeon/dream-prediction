import numpy as np
import argparse
import yaml
import os
import shutil
import tqdm
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:   
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    transform_by_subject(**config)


def transform_by_subject(root_dir, output_dir, subject_pattern, norm_opts):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
        
    print('\nNormalize signals for each subject...')

    for subject_freq_file in tqdm.tqdm(glob.glob(f'{root_dir}/{subject_pattern}'), ncols=70):

        subject_name = subject_freq_file.split('/')[-1].split('.')[0]
        subject_output_path = f'{output_dir}/{subject_name}'

        _normalize(subject_freq_file, subject_output_path, **norm_opts)

    print('Done.\n')


def _normalize(input_path, output_path, log, mode):

    normed = {}
    with np.load(input_path) as signals:
        
        trials = [signals[sname] for sname in signals]
            
        concat = np.concatenate(trials) # (n_trials * n_frames) x n_electrodes x n_channels
        if log:
            concat = np.log(concat)
        mean = concat.mean(axis=0)
        std = concat.std(axis=0)
                        
        for sname in signals:

            signal = signals[sname]
            
            if log:
                signal = np.log(signal)
            
            if mode == 'zscore':
                signal = (signal - mean) / std
            else: raise NotImplementedError(f'Mode {mode} not implemented.')
                
            name = f'{sname.split(".")[0]}.norm'
            normed[name] = signal

    np.savez_compressed(output_path, **normed)


if __name__ == '__main__':
    main()