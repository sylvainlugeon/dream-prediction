
import numpy as np 
import os
import shutil
import glob
import h5py
import yaml
import sys
import tqdm
import argparse
import traceback
import logging
import scipy.io as sio
from typing import Dict, Any, List

sys.path.append('/mlodata1/lugeon/eeg_project/scripts')
from data_processing.bandwidths_power import bandwidths_power
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
                         read_format: str, 
                         signal_name: str, 
                         transpose: bool, 
                         output_dir: str, 
                         filter_opts: Dict[str, Any], 
                         bw_power_opts: Dict[str, Any]) -> None:
    """For a root directory that contains multiple subject-specifc sub-directories,
    transform the time signals for each subjects into bandwidths power signal and group
    the results in a single numpy archive file for each subject. 
    
    Exemple of root dir:
    root_dir/
    |-- subject_1/
    |   |-- trial_1a.mat
    |   |-- trial_1b.mat
    |-- subject_2/
        |-- trial_2a.mat
        |-- trial_2b.mat
        
    Exemple of output dir:
    output_dir/
    |-- subject_1.npz/
    |-- subject_2.npz/
     
    Args:
        root_dir (str): root directory that contains the subjects sub-directories
        read_format (str): reading format for the files, either 'h5py' or 'scipy'
        signal_name (str): key for the signal in a file
        transpose (bool): if the signal should be transposed
        output_dir (str): output directory to write the transformed signals
        filter_opts (Dict[str, Any]): options for filtering
        bw_power_opts (Dict[str, Any]): options for bandwidths power computation
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('\nExtract bandwidths power for each subject...')
    
    subject_pattern = filter_opts['subject_pattern']
    data_pattern = filter_opts['data_pattern']
    n_electrodes = filter_opts['n_electrodes']

    for subject in tqdm.tqdm(glob.glob(f'{root_dir}/{subject_pattern}'), ncols=70):

        subject_name = subject.split('/')[-1]
        subject_output_path = f'{output_dir}/{subject_name}'
        
        eeg_files = glob.glob(f'{subject}/{data_pattern}')
        
        signals = []
        for file in eeg_files:
            try:
                signal = _read_time_signal(file, read_format, signal_name, transpose, n_electrodes)
                signals.append(signal)
            
            except Exception as e:
                print(f'\nProblem while reading {file}, discard file.')
                logging.error(traceback.format_exc())
                        
        names = [file.split('/')[-1].split('.')[0] for file in eeg_files]

        _save_frequencies(signals, names, subject_output_path, bw_power_opts)

    print('Done.\n')
    
def _read_time_signal(file_path: str, 
                      read_format: str, 
                      signal_name: str, 
                      transpose: bool, 
                      n_electrodes: int) -> np.array:
    """Read a time signal on disk 

    Args:
        file_path (str): path to the file that contains the signal
        read_format (str): reading format for the file, either 'h5py' or 'scipy'
        signal_name (str): key for the signal in the file
        transpose (bool): if the signal should be transposed
        n_electrodes (int): number of electrodes

    Raises:
        NotImplementedError: if the provided reading format doesn't exist

    Returns:
        np.array: time signal, of shape [n_samples, n_electrodes]
    """
    
    if read_format == 'h5py':
            with h5py.File(file_path,'r') as file:
                time_signal = np.array(file[signal_name])
                
    elif read_format == 'scipy':
        time_signal = sio.loadmat(file_path)[signal_name]
                
    else: raise NotImplementedError(f'Format {read_format} is not implemented.')
    
    if transpose:
        time_signal = time_signal.T
    
    assert time_signal.shape[1] == n_electrodes, \
        f'Signal at {file_path} has {time_signal.shape[1]} dimension, ' \
        f'but should have {n_electrodes}.'
        
    return time_signal


def _save_frequencies(time_signals: List[np.array], 
                      signal_names: List[str], 
                      output_path: str, 
                      bw_power_opts: Dict[str, Any]) -> None:
    """Transform time signal into bandwidths power signal and save them under
    a numpy archive on disk

    Args:
        time_signals (List[np.array]): List of time signals, each of shape [n_samples, n_electrodes]
        signal_names (List[str]): List of signal names
        output_path (str): location of the numpy archive
        bw_power_opts (Dict[str, Any]): options for bandwidths power computation
    """

    freq_signals = {}

    for time_signal, sname in zip(time_signals, signal_names):

        # remove last unit, it is the reference for voltage (always zero)
        time_signal = time_signal[:, :-1] 

        bandwiths_power = bandwidths_power(time_signal, **bw_power_opts)
        freq_signals[f'{sname}.freq'] = bandwiths_power

        np.savez_compressed(output_path, **freq_signals)


if __name__ == '__main__':
    main()