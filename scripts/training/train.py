import sys
import argparse
import yaml

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append('/home/lugeon/eeg_project/scripts')
from training import models
from training.dataset import EEG_Image_Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # get activation function from function name    
    config['model']['kwargs']['activation_fn'] = \
        getattr(torch.nn, config['model']['kwargs']['activation_fn'])
    
    # get model from model name
    model_function = getattr(models, config['model']['name'])
    model = model_function(**config['model']['kwargs'])
    
    # get loss function from function name
    loss_function = getattr(torch.nn, config['model']['loss'])
    criterion = loss_function()
    
    # get optimizer from optimizer name
    optimizer_function = getattr(torch.optim, config['optim']['name'])
    optimizer = optimizer_function(model.parameters(), **config['optim']['kwargs'])
    
    # get data loaders
    train_loader, val_loader = _get_loaders(**config['data'])
    
    n_epochs = config['n_epochs']
        
    # start training routine
    train_routine(model, n_epochs, train_loader, val_loader, optimizer, criterion)
    
    
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
    

def train_routine(model, n_epochs, train_loader, val_loader, optimizer, criterion):
    
    for p in model.parameters():
        p.requires_grad = True
    
    for epoch in range(n_epochs):
        
        training_loss = 0
        validation_loss = 0
        
        model.train()
        for input_batch, output_batch in iter(train_loader):
                        
            prediction = model(input_batch)
            loss = criterion(prediction, output_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
        with torch.no_grad():
            model.eval()
            for input_batch, output_batch in iter(val_loader):
                            
                prediction = model(input_batch)
                loss = criterion(prediction, output_batch)
                
                validation_loss += loss.item()
                
        training_loss = training_loss / len(train_loader)
        validation_loss = validation_loss / len(val_loader)
        
        print(f'Epoch {epoch:>4} ** Train loss: {training_loss:>10.2f} ** Val loss: {validation_loss:>10.2f}')
        
            

if __name__ == '__main__':
    main()