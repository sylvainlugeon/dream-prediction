import sys
import argparse
import yaml
import time
import os
import shutil
import random
import tqdm
from typing import Union, Tuple, Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

sys.path.append('/home/lugeon/eeg_project/scripts')
from training.representation import models, losses, datasets, cl_transforms
from training.representation.early_stopping import EarlyStopping
from interaction.interaction import ask_for_config

random.seed(0)
torch.manual_seed(0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    
    print()
    print('Process configuration...')
    train_routine_kwargs = _process_train_config(args.config)

    print('Start training...')
    train_routine(**train_routine_kwargs)
    

def train_routine(model: nn.Module, 
                  n_epochs: int, 
                  train_loader: DataLoader, 
                  val_loader: DataLoader, 
                  reshuffle: bool,
                  optimizer: torch.optim.Optimizer, 
                  criterion: nn.Module, 
                  early_stopping: Union[None, EarlyStopping], 
                  scheduler: Union[None, object],
                  device: torch.device, 
                  save_dir: str) -> None:
    
    for p in model.parameters():
        p.requires_grad = True
        
    model.to(device) 
    
    for epoch in range(n_epochs):
        
        training_loss = 0
        validation_loss = 0
        
        # training
        model.train()
                
        for input_batch, output_batch in tqdm.tqdm(train_loader, ncols=70):
                        
            # send to device
            input_batch = input_batch.to(device, dtype=torch.float) 
            output_batch = output_batch.to(device, dtype=torch.float)
            
            output_type = train_loader.dataset.dataset.output_type
            
            # forward pass for traditional learning
            if output_type in {'label', 'last_frame'}:
                prediction = model(input_batch)
                loss = criterion(prediction, output_batch)
                
            # forward pass for contrastive learning
            elif output_type in {'transform'}:
                first_transformed = model(input_batch)   
                second_transformed = model(output_batch)
                loss = criterion(first_transformed, second_transformed)
            
            else:
                raise NotImplementedError
            
            # backward pass
            if epoch > 0: # don't update weights at first epoch, to get the baseline loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            training_loss += loss.item()
            
        # evaluate 
        with torch.no_grad():
            model.eval()
            for input_batch, output_batch in iter(val_loader):
                
                # send to device
                input_batch = input_batch.to(device, dtype=torch.float)
                output_batch = output_batch.to(device, dtype=torch.float)
                
                # forward pass only   
                if output_type in {'label', 'last_frame'}:     
                    prediction = model(input_batch)
                    loss = criterion(prediction, output_batch)
                    
                # forward pass for contrastive learning
                elif output_type in {'transform'}:
                    first_transformed = model(input_batch)   
                    second_transformed = model(output_batch)
                    loss = criterion(first_transformed, second_transformed)
                
                else:
                    raise NotImplementedError
                
                validation_loss += loss.item()
                
        training_loss = training_loss / len(train_loader.dataset)
        validation_loss = validation_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch:>4} **' 
              f'Train loss: {training_loss:>10.4f} ** '
              f'Val loss: {validation_loss:>10.4f}')
                
        # if early stopping is enabled
        if early_stopping:
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # if scheduler is enabled
        if scheduler: 
            scheduler.step(validation_loss)
            
        if reshuffle:
            train_loader.dataset.dataset.reshuffle()
            val_loader.dataset.dataset.reshuffle()
            
        with open(f'{save_dir}/loss.txt', 'a') as f:
            if epoch == 0:
                f.write(f'epoch training validation\n')
            f.write(f'{epoch} {training_loss:.4f} {validation_loss:.4f}\n')
            
        
def _process_train_config(config: str) -> Dict[str, Any]:
    
    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    if ask_for_config(config): pass
    else: return
        
    save_dir = config['save_dir']
    
    if os.path.exists(save_dir):
        while True:
            delete = input(f'Are you sure you want to overwrite {save_dir}? (y/n): ')
            if delete == 'y':
                break
            if delete == 'n':
                raise RuntimeError('User stopped the program.')
            else:
                print('Answer must be "y" or "n"')
            
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    with open(f'{save_dir}/train_config.yaml', 'w') as f:
        f.write(yaml.dump(config))
    
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
        
    # get data loaders
    train_loader, val_loader = _get_loaders(**config['data'])
    
    train_routine_kwargs = {}
    train_routine_kwargs['model'] = model
    train_routine_kwargs['n_epochs'] = config['n_epochs']
    train_routine_kwargs['train_loader'] = train_loader
    train_routine_kwargs['val_loader'] = val_loader
    train_routine_kwargs['reshuffle'] = config['reshuffle']
    train_routine_kwargs['optimizer'] = optimizer
    train_routine_kwargs['criterion'] = criterion
    train_routine_kwargs['early_stopping'] = early_stopping
    train_routine_kwargs['scheduler'] = scheduler
    train_routine_kwargs['device'] = torch.device(config['device'])
    train_routine_kwargs['save_dir'] = save_dir
    
    return train_routine_kwargs
    

def _get_loaders(dset_name: str, 
                 dset_kwargs: int, 
                 fraction: int) -> Tuple[DataLoader, DataLoader]:
    
    if dset_kwargs['transforms']: # get transform functions if needed
        for t in dset_kwargs['transforms']:
            t['f'] = getattr(cl_transforms, t['f'])
    
    dataset = getattr(datasets, dset_name)(**dset_kwargs)
    
    val_size = int(fraction * len(dataset))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=None, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False)

    
    return train_loader, val_loader
    

if __name__ == '__main__':
    main()