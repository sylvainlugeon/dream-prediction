import numpy as np
import argparse
import yaml
import os
import shutil
import tqdm
import glob
import sys
from typing import Dict, Any

sys.path.append('/mlodata1/lugeon/eeg_project/scripts')
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


def transform_by_subject(root_dir: str, 
                         output_dir: str, 
                         subject_pattern: str, 
                         norm_opts: Dict[str, Any]) -> None:
    """ For each subject-specific numpy archive in the root directory, normalize
    all the signals contained in the archive and write the result on disk

    Args:
        root_dir (str): root directory with the subject-specific numpy archives
        output_dir (str): where to write the normalized archives
        subject_pattern (str): pattern for filtering subjects
        norm_opts (Dict[str, Any]): options for normalization
    """
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
        
    print('\nNormalize signals for each subject...')

    for subject_freq_file in tqdm.tqdm(glob.glob(f'{root_dir}/{subject_pattern}'), ncols=70):

        subject_name = subject_freq_file.split('/')[-1].split('.')[0]
        subject_output_path = f'{output_dir}/{subject_name}'

        _normalize(subject_freq_file, subject_output_path, **norm_opts)

    print('Done.\n')


def _normalize(input_path: str, 
               output_path: str, 
               log: bool, 
               mode: str) -> None:
    """Read an numpy archive that can contains many signals, and normalize 
    all the signals with metrics computed on all signals. Typically, a archive
    contains all the signals for a given subject.

    Args:
        input_path (str): path to the numpy archive
        output_path (str): where to write the normalize archive
        log (bool): apply a log-transformation to the signal
        mode (str): normalization method, either 'zscore' or 'none'

    Raises:
        NotImplementedError: if the normalization method doesn't exist
    """

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

            elif mode == 'none':
                pass
            
            else: raise NotImplementedError(f'Mode {mode} not implemented.')
                
            name = f'{sname.split(".")[0]}.norm'
            normed[name] = signal

    np.savez_compressed(output_path, **normed)


if __name__ == '__main__':
    main()