import torch
import torch.nn as nn
from torch.autograd import Function

import yaml
import sys


sys.path.append('/home/lugeon/eeg_project/scripts')
from training.representation import models
    
###################################################
#          Model container for fine-tuning        #
###################################################
    
class FineTuner(nn.Module):
    
    def __init__(self, 
                 result_dir: str, 
                 encoding_dim: int, 
                 n_classes: int, 
                 dropout: float, 
                 adverserial: bool = False,
                 adv_strength: float = 0.2,
                 n_adv_classes: int = 20):
        
        super(FineTuner, self).__init__()
        
        with open(f'{result_dir}/train_config.yaml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    
        # get pretrained model
        model_function = getattr(models, train_config['model']['name'])
        self.model = model_function(**train_config['model']['kwargs'])
        self.model.load_state_dict(torch.load(f'{result_dir}/checkpoint.pt'))
        
        self.mean_agg = False

        if isinstance(self.model, models.MaskedAutoEncoder):
            self.model.masking_ratio = 0.
            self.mean_agg = True
        
        self.clf = nn.Linear(encoding_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.adverserial = adverserial
        
        if self.adverserial:
            self.adv_clf = nn.Linear(encoding_dim, n_adv_classes)
            self.adv_strength = adv_strength

        
    def forward(self, x):
        encoding = self.model.encode(x)
        
        if self.mean_agg:
            encoding = encoding.mean(1)
        
        encoding = self.dropout(encoding)
        output = self.clf(encoding)
        
        if self.adverserial:
            reversed_encoding = ReverseLayerF.apply(encoding, self.adv_strength)
            adv_output = self.adv_clf(reversed_encoding)
            output = (output, adv_output,)
            
        return output
        
    def encode(self, x):
        return self.model.encode(x)
    
    
    
class ReverseLayerF(Function):
    '''
    Gradient Reversal Layer from `Domain-Adversarial Training of Neural Networks` (Ganin 2016).
    '''
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_

        return output, None
                    
