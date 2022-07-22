import sys
import argparse
import yaml
import random
import os
from tqdm import tqdm
from typing import Union, Any
import torch
from torch import nn
from torch.utils.data import DataLoader

#sys.path.append('/home/lugeon/eeg_project/scripts')
sys.path.append('/mlodata1/lugeon/eeg_project/scripts')

from training.representation import losses
from losses import RunningLoss
from training.representation.early_stopping import EarlyStopping
from training.representation.process_config import process_train_config

random.seed(0)
torch.manual_seed(0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    
    print()
    print('Process configuration...')
    
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    train_routine_kwargs = process_train_config(config)

    print('Start training...')
    train_routine(**train_routine_kwargs)
        

def train_routine(model: nn.Module, 
                  n_epochs: int, 
                  train_loader: DataLoader, 
                  val_loader: DataLoader, 
                  reshuffle: bool,
                  optimizer: torch.optim.Optimizer, 
                  criterion: nn.Module, 
                  n_losses: int,
                  early_stopping: Union[None, EarlyStopping], 
                  save_epoch: Union[None, int],
                  scheduler: Union[None, Any],
                  step_value: str,
                  device: torch.device, 
                  save_dir: str,
                  verbose: bool = True) -> None:
        
    model.to(device) 
    output_type = train_loader.dataset.dataset.output_type
    metadata = train_loader.dataset.dataset.return_metadata
    
    training_loss = RunningLoss(len(train_loader.dataset), n_losses)
    validation_loss = RunningLoss(len(val_loader.dataset), n_losses)
    
    for epoch in range(-1, n_epochs):
        
        ############################ training ############################

        model.train()
        if verbose: pbar = tqdm(total=len(train_loader), ncols=70)
                
        for batch in train_loader:
            
            if metadata: 
                    input_batch, output_batch, metadata = batch
                    sid_batch, _, _, _, _ = metadata
                    sid_batch = sid_batch.to(device)
            else: 
                input_batch, output_batch = batch
                sid_batch = None
                        
            # send to device
            input_batch = input_batch.to(device) 
            output_batch = output_batch.to(device)
                        
            # forward pass
            loss = _forward_pass(input_batch, output_batch, sid_batch, model, criterion, output_type)
            training_loss.update(loss)
                        
            # backward pass
            if epoch >= 0: # don't update weights at first epoch, to get the baseline loss
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()
                
            if verbose: pbar.update(1)
        if verbose: pbar.close()
               
        ############################ evaluate ############################
        
        with torch.no_grad():
            model.eval()
            for batch in iter(val_loader):
                
                if metadata: 
                    input_batch, output_batch, metadata = batch
                    sid_batch, _, _, _, _ = metadata
                    sid_batch = sid_batch.to(device)
                else: 
                    input_batch, output_batch = batch
                    sid_batch = None
                
                # send to device
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                                
                # forward pass
                loss = _forward_pass(input_batch, output_batch, sid_batch, model, criterion, output_type)                    
                validation_loss.update(loss)
                
        if verbose: print(f'Epoch {epoch:>4} ** ' 
                          f'Train loss: {training_loss.export():>20} ** '
                          f'Val loss: {validation_loss.export():>20}')
        
        watched_loss = validation_loss.watch()

        # if saving per epoch
        if save_epoch:
            if epoch % save_epoch == 0:
                os.makedirs(f'{save_dir}/epochs/{epoch}')
                torch.save(model.state_dict(), f'{save_dir}/epochs/{epoch}/checkpoint.pt')
                
        # if early stopping is enabled
        if early_stopping:
            early_stopping(watched_loss, model)
            if early_stopping.early_stop:
                if verbose: print("Early stopping")
                break
        
        # if scheduler is enabled and baseline epoch is over
        if scheduler and epoch >= 0: 
            if step_value == 'loss': scheduler.step(watched_loss)
            elif step_value == 'epoch': scheduler.step(epoch)
            elif step_value == 'none': scheduler.step()
            else: raise NotImplementedError(f'Step value {step_value} is not implemented')
            
        if reshuffle:
            train_loader.dataset.dataset.reshuffle()
            val_loader.dataset.dataset.reshuffle()
            
        with open(f'{save_dir}/loss.txt', 'a') as f:
            if epoch == -1:
                f.write(f'epoch training validation\n')
            f.write(f'{epoch} {training_loss.export()} {validation_loss.export()}\n')
            
        training_loss.reset()
        validation_loss.reset()
            
            
def _forward_pass(input, output, sid, model, criterion, output_type):
    
    # forward pass for reconstruction
    if output_type in {'none'}:
        if sid is not None: output = (input, sid)
        else: output = input
        prediction = model(input)

    elif output_type in {'label', 'next_frame'}:
        if sid is not None: output = (output, sid)
        else: output = output
        prediction = model(input)

    # forward pass for contrastive learning
    elif output_type in {'transform'}:
        # TODO: implement adverserial on contrastive
        prediction = model(input)   
        output = model(output)
    
    else:
        raise NotImplementedError

    loss = criterion(prediction, output)
    
    return loss 
            
            
if __name__ == '__main__':
    main()