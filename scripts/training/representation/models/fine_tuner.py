import torch
import torch.nn as nn
from torch.autograd import Function
import yaml
import sys

sys.path.append('/mlodata1/lugeon/eeg_dream/scripts')
from training.representation import models

def test():
    pass
    
###################################################
#          Model container for fine-tuning        #
###################################################
    
class FineTuner(nn.Module):
    
    def __init__(self, 
                 result_dir: str, 
                 encoding_dim: int, 
                 n_classes: int, 
                 dropout: float, 
                 adverserial: bool,
                 adv_strength: float,
                 n_adv_classes: int,
                 freeze: bool):
        
        super(FineTuner, self).__init__()
        
        with open(f'{result_dir}/train_config.yaml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    
        # get pretrained model
        model_function = getattr(models, train_config['model']['name'])
        self.model = model_function(**train_config['model']['kwargs'])
        self.model.load_state_dict(torch.load(f'{result_dir}/checkpoint.pt'))
        self._set_dropout(self.model, dropout)
        
        self.mean_agg = False
        if isinstance(self.model, models.MaskedAutoEncoder):
            self.model.masking_ratio = 0.
            self.mean_agg = True
            
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.clf = nn.Linear(encoding_dim, n_classes)
        self.adverserial = adverserial
        
        if self.adverserial:
            self.adv_clf = nn.Linear(encoding_dim, n_adv_classes)
            self.adv_strength = adv_strength

        
    def forward(self, x):
        encoding = self.model.encode(x)
        
        if self.mean_agg:
            encoding = encoding.mean(1)
        
        output = self.clf(encoding)
        
        if self.adverserial:
            reversed_encoding = ReverseLayerF.apply(encoding, self.adv_strength)
            adv_output = self.adv_clf(reversed_encoding)
            output = (output, adv_output,)
            
        return output
        
    def encode(self, x):
        return self.model.encode(x)
    
    @staticmethod
    def _set_dropout(model, new_drop_rate):
        for _, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = new_drop_rate
            FineTuner._set_dropout(child, new_drop_rate)
        
    
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
                    
