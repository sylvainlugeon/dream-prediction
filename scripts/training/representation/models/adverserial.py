from torch import Tensor
import torch.nn as nn
from torch.autograd import Function
import sys

sys.path.append('/mlodata1/lugeon/eeg_project/scripts')


#######################################################
#   Model container for domain-adverserial learning   #
#######################################################
    
class AdverserialAutoencoder(nn.Module):
    
    def __init__(self, 
                 autoencoder_kwargs: int, 
                 adv_strength: float,
                 n_adv_classes: int):
        
        super().__init__()

        # get model
        from training.representation import models
        self.model = models.MaskedAutoEncoder(**autoencoder_kwargs)

        # adverserial classifier
        self.adv_clf = nn.Linear(self.model.emb_size, n_adv_classes)
        self.adv_strength = adv_strength

    def forward(self, x):

        encoding, shuffling_idx = self.model.encode(x, return_idx=True)
        reconstruction = self.model.decode(encoding, shuffling_idx)
                
        representation = encoding.mean(1)
        reversed = ReverseLayerF.apply(representation, self.adv_strength)
        adv_output = self.adv_clf(reversed)

        output = (reconstruction, adv_output,)
            
        return output
        
    def encode(self, x: Tensor, return_idx: bool = False) -> Tensor:
        return self.model.encode(x, return_idx)

    def decode(self, x: Tensor, idx: Tensor) -> Tensor:
        return self.model.decode(x, idx)
        
    
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