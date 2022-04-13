
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

sys.path.append('/home/lugeon/eeg_project/scripts')
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


def transform_by_subject(root_dir, read_format, output_dir, filter_opts, bw_power_opts):

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
                signal = _read_time_signal(file, read_format, n_electrodes)
                signals.append(signal)
            
            except Exception as e:
                print(f'\nProblem while reading {file}, discard file.')
                logging.error(traceback.format_exc())
                        
        names = [file.split('/')[-1].split('.')[0] for file in eeg_files]

        _save_frequencies(signals, names, subject_output_path, bw_power_opts)

    print('Done.\n')
    
def _read_time_signal(file_path, read_format, n_electrodes):
    
    if read_format == 'h5py':
            with h5py.File(file_path,'r') as file:
                time_signal = np.array(file['datavr'])
                
                assert time_signal.shape[1] == n_electrodes, \
                       f'Signal at {file_path} has {time_signal.shape[1]} dimension, ' \
                       f'but should have {n_electrodes}.'
                
    else: raise NotImplementedError(f'Format {read_format} is not implemented.')
        
    return time_signal


def _save_frequencies(time_signals, signal_names, output_path, bw_power_opts):

    freq_signals = {}

    for time_signal, sname in zip(time_signals, signal_names):

        # remove last unit, it is the reference for voltage (always zero)
        time_signal = time_signal[:, :-1] 

        bandwiths_power = bandwidths_power(time_signal, **bw_power_opts)
        freq_signals[f'{sname}.freq'] = bandwiths_power

        np.savez_compressed(output_path, **freq_signals)


if __name__ == '__main__':
    main()