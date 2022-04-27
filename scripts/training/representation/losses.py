import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def pairwise_loss(self, 
                      similarity_matrix: torch.Tensor, 
                      i: int, 
                      j: int):
                    
        numerator = torch.exp(similarity_matrix[i, j])
        
        one_for_not_i = torch.ones(similarity_matrix.size(0)).to(similarity_matrix.device)
        one_for_not_i[i] = 0 
        
        denominator = torch.sum(
            one_for_not_i * torch.exp(similarity_matrix[i, :])
        )    
            
        pairwise_loss = - torch.log(numerator / denominator)
            
        return pairwise_loss.squeeze(0)
            
    def forward(self, 
                first_transformed: torch.Tensor, 
                second_transformed: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            first_transformed (torch.Tensor): Tensor of encodings under first transformation
            second_transformed (torch.Tensor): Tensor of encodings under second transformation

        Returns:
            torch.Tensor: Average loss in singleton tensor
        """
        
        assert first_transformed.size(0) == second_transformed.size(0), (
            f'Both tensors should have the same dimension at axis 0, '
            f'{first_transformed.size(0)} != {second_transformed.size(0)}'
        )
        
        assert first_transformed.size(1) == second_transformed.size(1), (
            f'Both tensors should have the same dimension at axis 1'
            f'{first_transformed.size(1)} != {second_transformed.size(1)}'
        )
        
        batch_size = first_transformed.size(0)
        
        # vertical stacking -> (2 * batch_size) x embedding_dim
        stacked = torch.cat([first_transformed, second_transformed], dim=0)
        
        # normalize each embedding
        stacked = F.normalize(stacked, dim=1)
        
        # similarity between embeddings
        similarity_matrix = torch.matmul(stacked, stacked.T) / self.temperature

        # sum pairwise loss
        loss = 0.0
        for k in range(0, batch_size):
            loss += self.pairwise_loss(similarity_matrix, k, k + batch_size) \
                  + self.pairwise_loss(similarity_matrix, k + batch_size, k)
                  
        # average over the batch
        return 1.0 / (2 * batch_size) * loss