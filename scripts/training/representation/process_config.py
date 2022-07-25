import sys
import yaml
from typing import Tuple, Dict, Any
import torch

from torch.utils.data import Dataset, DataLoader, random_split

#sys.path.append('/home/lugeon/eeg_project/scripts')
sys.path.append('/mlodata1/lugeon/eeg_dream/scripts')

from training.representation import models, losses
from training.representation.early_stopping import EarlyStopping
from training.representation.schedulers import GradualWarmupScheduler
from training.dataset import datasets, cl_transforms
from training.dataset.datasets import EEG_Image_Batch_Dataset
from interaction.interaction import ask_for_config, ask_for_dir

def process_train_config(config: Dict[str, Any], ask_user: bool = True) -> Dict[str, Any]:
    
    save_dir = config['save_dir']

    if ask_user:
        if ask_for_config(config): pass
        else: raise RuntimeError('User stopped the program.')
                
        if ask_for_dir(save_dir): pass
        else: raise RuntimeError('User stopped the program.')
    
    with open(f'{save_dir}/train_config.yaml', 'w') as f:
        f.write(yaml.dump(config))
        
    # get device
    device = torch.device(config['device'])
    
    # get model from model name
    model_function = getattr(models, config['model']['name'])
    model = model_function(**config['model']['kwargs'])
    
    # get loss function from function name
    loss_name = config['loss']['name']
    loss_function = getattr(losses, loss_name, None)
    if not loss_function:
        loss_function = getattr(torch.nn, loss_name, None)
        
    assert loss_function is not None, (
        f'Could not find a loss function that corresponds to {loss_name}')

    _send_weight_to_device(config['loss']['kwargs'], device) # send all weight tensor to device
    criterion = loss_function(**config['loss']['kwargs'])
    
    # get optimizer from optimizer name
    optimizer_function = getattr(torch.optim, config['optim']['name'])
    optimizer = optimizer_function(model.parameters(), **config['optim']['kwargs'])
    
    # get early stopping module
    if config['early_stop']['enabled']:
        path = config['early_stop']['kwargs']['path']
        config['early_stop']['kwargs']['path'] = f'{save_dir}/{path}' # change path to write in save_dir
        early_stopping = EarlyStopping(**config['early_stop']['kwargs'])
    else: early_stopping = None
    
    # get scheduler
    if config['scheduler']['enabled']:
        scheduler_function = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])
        scheduler = scheduler_function(optimizer, **config['scheduler']['kwargs'])
    else: scheduler = None

    # get warmup scheduler
    if config['scheduler']['warmup']:
        scheduler = GradualWarmupScheduler(
            optimizer=optimizer, 
            total_epoch=config['scheduler']['warmup'],
            after_scheduler=scheduler
        )
        
    # get data loaders
    train_loader, val_loader = get_loaders(**config['data'])
    
    
    if isinstance(model, models.FineTuner):
        if model.adverserial:
            assert isinstance(criterion, losses.AdverserialLoss), (
                f'Fine-tuning model enables adverserial training, but loss {criterion} does not support it'
            )
            assert train_loader.dataset.dataset.return_metadata, (
                'Fine-tuning model enables adverserial training, but the dataset returns no adverserial metadata'
            )
    
    train_routine_kwargs = {}
    train_routine_kwargs['model'] = model
    train_routine_kwargs['n_epochs'] = config['n_epochs']
    train_routine_kwargs['save_epoch'] = config['save_epoch']
    train_routine_kwargs['train_loader'] = train_loader
    train_routine_kwargs['val_loader'] = val_loader
    train_routine_kwargs['reshuffle'] = config['reshuffle']
    train_routine_kwargs['optimizer'] = optimizer
    train_routine_kwargs['criterion'] = criterion
    train_routine_kwargs['n_losses'] = config['loss']['n_losses']
    train_routine_kwargs['early_stopping'] = early_stopping
    train_routine_kwargs['scheduler'] = scheduler
    train_routine_kwargs['step_value'] = config['scheduler']['step_value']
    train_routine_kwargs['device'] = device
    train_routine_kwargs['save_dir'] = save_dir
    train_routine_kwargs['verbose'] = config['verbose']
    
    return train_routine_kwargs

def get_dataset(dset_name: str, 
                dset_kwargs: Dict[str, Any]) -> EEG_Image_Batch_Dataset:
    
    if dset_kwargs['transforms']: # get transform functions if needed
        for t in dset_kwargs['transforms']:
            t['f'] = getattr(cl_transforms, t['f'])
    
    dataset = getattr(datasets, dset_name)(**dset_kwargs)
    
    return dataset

def split_train_test(dataset: Dataset, 
                      fraction: float) -> Tuple[DataLoader, DataLoader]:
    
    val_size = int(fraction * len(dataset))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=None, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False)
    
    return train_loader, val_loader


def get_loaders(dset_name: str, 
                 dset_kwargs: Dict[str, Any], 
                 fraction: float) -> Tuple[DataLoader, DataLoader]:
    
    dataset = get_dataset(dset_name, dset_kwargs)
    loaders = split_train_test(dataset, fraction)
    
    return loaders


def _send_weight_to_device(root_dict: Dict[str, Any], device: torch.device):
    for key, value in root_dict.items():
        if isinstance(value, dict):
            _send_weight_to_device(value, device)
        else:
            if key == 'weight':
                root_dict[key] = torch.Tensor(value).to(device)