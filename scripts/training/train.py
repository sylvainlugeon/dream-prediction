from multiprocessing import reduction
from sched import scheduler
import sys
import argparse
import yaml

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append('/home/lugeon/eeg_project/scripts')

from training import models
from training.dataset import EEG_Image_Dataset
from training.early_stopping import EarlyStopping

from interaction.interaction import ask_for_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    if ask_for_config(config): pass
    else: return
    
    print('Process configuration...')
    
    # get activation function from function name    
    config['model']['kwargs']['activation_fn'] = \
        getattr(torch.nn, config['model']['kwargs']['activation_fn'])
    
    # get model from model name
    model_function = getattr(models, config['model']['name'])
    model = model_function(**config['model']['kwargs'])
    
    # get loss function from function name
    loss_function = getattr(torch.nn, config['loss']['name'])
    criterion = loss_function(**config['loss']['kwargs'])
    
    # get optimizer from optimizer name
    optimizer_function = getattr(torch.optim, config['optim']['name'])
    optimizer = optimizer_function(model.parameters(), **config['optim']['kwargs'])
    
    # get early stopping module
    if config['early_stop']['enabled']:
        early_stopping = EarlyStopping(**config['early_stop']['kwargs'])
    else: early_stopping = None
    
    # get scheduler
    if config['scheduler']['enabled']:
        scheduler_function = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])
        scheduler = scheduler_function(optimizer, **config['scheduler']['kwargs'])
    else: scheduler = None
    
    n_epochs = config['n_epochs']
    device = torch.device(config['device'])
    
    print('Instantiate data loader...')
    
    # get data loaders
    train_loader, val_loader = _get_loaders(**config['data'])
    
    print('Start training...')
        
    # start training routine
    train_routine(model, n_epochs, 
                  train_loader, val_loader, 
                  optimizer, criterion, 
                  early_stopping, scheduler,
                  device)
    

def train_routine(model, n_epochs, 
                  train_loader, val_loader, 
                  optimizer, criterion, 
                  early_stopping, scheduler,
                  device):
    
    for p in model.parameters():
        p.requires_grad = True
        
    model.to(device) 
    
    for epoch in range(n_epochs):
        
        training_loss = 0
        validation_loss = 0
        
        # training
        model.train()
        for input_batch, output_batch in iter(train_loader):
            
            # send to device
            input_batch = input_batch.to(device) 
            output_batch = output_batch.to(device)
                       
            # forward pass
            prediction = model(input_batch)
            loss = criterion(prediction, output_batch)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
        # evaluate 
        with torch.no_grad():
            model.eval()
            for input_batch, output_batch in iter(val_loader):
                
                # send to device
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                
                # forward pass only        
                prediction = model(input_batch)
                loss = criterion(prediction, output_batch)
                
                validation_loss += loss.item()
                
        training_loss = training_loss / len(train_loader.dataset)
        validation_loss = validation_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch:>4} ** Train loss: {training_loss:>10.2f} ** Val loss: {validation_loss:>10.2f}')
        
        # if early stopping is enabled
        if early_stopping:
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # if scheduler is enabled
        if scheduler: 
            scheduler.step(validation_loss)
        
            
def _get_loaders(hdf5_file, window, output_type, fraction, batch_size):
    
    dataset = EEG_Image_Dataset(hdf5_file, window, output_type)
    val_size = int(fraction * len(dataset))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, 
                                      [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader
    

if __name__ == '__main__':
    main()