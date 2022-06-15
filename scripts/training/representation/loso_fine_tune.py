from locale import strcoll
import sys
import argparse
import yaml
import os
import random
import h5py
import torch
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from torch.utils.data import DataLoader
sys.path.append('/home/lugeon/eeg_project/scripts')
from training.representation.process_config import (
    process_train_config, 
    get_dataset)
from training.representation.train_deep import train_routine
from interaction.interaction import ask_for_config, ask_for_dir
from training.representation import models

random.seed(0)
torch.manual_seed(0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    save_dir = config['save_dir']
    
    if ask_for_config(config): pass
    else: raise RuntimeError('User stopped the program.')
    
    if ask_for_dir(save_dir): pass
    else: raise RuntimeError('User stopped the program.')
    
    with h5py.File(config['data']['dset_kwargs']['hdf5_file'], 'r') as f:
        subjects = np.arange(len(f['subject_name'])).tolist()
        
    with open(f'{save_dir}/train_config.yaml', 'w') as f:
        f.write(yaml.dump(config))
        
    _train_for_each_subject(config, subjects, save_dir)
    _evaluate_for_each_subject(config, subjects, save_dir)
    
    
def _train_for_each_subject(config: Dict[str, Any], 
                            subjects: List[int], 
                            save_dir: str):
    for sub in subjects:
        
        sub_save_dir = f'{save_dir}/{sub}'
        os.makedirs(sub_save_dir)
        
        sub_config = copy.deepcopy(config)
        sub_config['save_dir'] = sub_save_dir
        sub_config['data']['dset_kwargs']['exclude_subject'] = [sub]
        
        train_routine_kwargs = process_train_config(sub_config, ask_user=False)
        
        print(f'Training for subject {sub}')
        
        train_routine(**train_routine_kwargs)
        
    
        
def _evaluate_for_each_subject(config: Dict[str, Any], 
                               subjects: List[int], 
                               save_dir: str):
    
    device = torch.device(config['device'])
    dataset = get_dataset(config['data']['dset_name'], config['data']['dset_kwargs'])
    dataset.return_metadata = True # force to return metadata
    
    preds, labels, sid, tid, fid, ss, et = [], [], [], [], [], [], []
    
    for sub in subjects:

        sub_save_dir = f'{save_dir}/{sub}'
        with open(f'{sub_save_dir}/train_config.yaml') as file:
            sub_config = yaml.load(file, Loader=yaml.FullLoader)

        sub_model = _get_model(sub_config, device)
        exlude_subjects = subjects.copy()
        exlude_subjects.remove(sub)
        dataset.set_exlude_subjects(exlude_subjects) # exlude all subjects except current one
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        
        result_df = pd.DataFrame()
        
        with torch.no_grad():
            for batch in loader:
                
                input_batch, output_batch, metadata = batch
                sid_batch, tid_batch, fid_batch, ss_batch, et_batch = metadata
                
                input_batch = input_batch.to(device) 
                preds_batch = sub_model.forward(input_batch)
                
                if isinstance(sub_model, models.FineTuner):
                    if sub_model.adverserial:
                        preds_batch = preds_batch[0]
                        
                # normalize scores into probabilities and keep only positive part
                preds_batch = torch.softmax(preds_batch, dim=1)[:, 1] 
                        
                preds.extend(preds_batch.cpu().detach().tolist())
                labels.extend(output_batch.cpu().detach().tolist())
                sid.extend(sid_batch.tolist())
                tid.extend(tid_batch.tolist())
                fid.extend(fid_batch.tolist())
                ss.extend(ss_batch.tolist())
                et.extend(et_batch.tolist())
                
                    
    result_df = pd.DataFrame({'sid': sid, 
                                'tid': tid, 
                                'fid': fid, 
                                'ss': ss, 
                                'et': et,
                                'label': labels,
                                'score': preds})
    result_df.to_csv(f'{save_dir}/res.csv', index=False)        
    
def _get_model(config: Dict[str, Any], device):
    
    # get model from model name
    model_function = getattr(models, config['model']['name'])
    model = model_function(**config['model']['kwargs'])
    model.load_state_dict(torch.load(f'{config["save_dir"]}/checkpoint.pt'))

    if isinstance(model, models.MaskedAutoEncoder):
        model.masking_ratio = 0.

    model.to(device)
    model.eval();
    
    return model
    
if __name__ == '__main__':
    main()