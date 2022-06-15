# inspired from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Union

class PatchTransform(nn.Module):
    """
    Tranform a video into a sequence of time-space patches.
    """
    def __init__(self, 
                 emb_size: int,
                 in_channels: int, 
                 space_patch_size: int, 
                 time_patch_size: int):
        """
        Args:
            emb_size (int): Patch embedding size 
            in_channels (int): Number of channels in the video
            space_patch_size (int): Size of a patch in the space dimension
            time_patch_size (int): Size of a patch in the time dimension
        """
        
        super().__init__()
        
        patch_size = (time_patch_size, space_patch_size, space_patch_size)
        self.emb_size = emb_size
        self.projection = nn.Conv3d(in_channels, 
                                    emb_size, 
                                    kernel_size=patch_size,
                                    stride=patch_size)
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input videos of shape (batch_size, time, n_channels, x, y)

        Returns:
            Tensor: Embedded patches of shape (batch_size, n_patches, emb_size)
        """
        
        x = x.permute(0, 2, 1, 3, 4) # (batch_size, n_channels, time, x, y)
        x = self.projection(x) # (batch_size, emb, patch_time, patch_x, patch_y)
        
        # put embedding at the end
        x = x.permute(0, 2, 3, 4, 1) # (batch_size, patch_time, patch_x, patch_y, emb_size)
        
        # flatten time and space
        x = torch.flatten(x, start_dim=1, end_dim=3) # (batch_size, n_patches, emb_size)
        
        return x


class PixelTransform(nn.Module):
    """
    Transform a sequences of embedded patches back into the pixels space.
    """
    
    def __init__(self, 
                 emb_size: int,
                 in_channels: int, 
                 space_dim: int,
                 time_dim: int,
                 space_patch_size: int, 
                 time_patch_size: int):
        """
        Args:
            emb_size (int): Patch embedding size 
            in_channels (int): Number of channels in the video
            space_dim (int): Size of the space dimension in the video
            time_dim (int): Size of the time dimension in the video
            space_patch_size (int): Size of a patch in the space dimension
            time_patch_size (int): Size of a patch in the time dimension
        """
        
        super().__init__()
        
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.space_dim = space_dim
        
        n_pixels_per_patch = in_channels * time_patch_size * space_patch_size ** 2
        
        self.projection = nn.Conv2d(in_channels=1, 
                                    out_channels=n_pixels_per_patch, 
                                    kernel_size=(1, emb_size))
    
    def forward(self, x: Tensor) -> Tensor :
        """
        Args:
            x (Tensor): Patches of shape (batch_size, n_patches, emb_size)

        Returns:
            Tensor: Reconstructed videos of shape (batch_size, time, n_channels, x, y)
        """
        
        x = torch.unsqueeze(x, 1) # (batch_size x 1 x n_patches x emb_size)
        x = self.projection(x) # (batch_size x n_pixels x n_patches x 1)
        
        x = torch.squeeze(x, dim=3)
        x = x.permute(0, 2, 1) # (batch_size x n_patches x n_pixels)
        
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.in_channels, self.time_dim, self.space_dim, self.space_dim) 
        x = x.permute(0, 2, 1, 3, 4) # (batch_size, time, n_channels, x, y)
        return x
        
    
class PositionalAdder(nn.Module):
    """
    Add a positional encoding (space and time) to the patches embeddings.
    """
    
    def __init__(self, 
                 emb_size: int, 
                 time_n_patches: int, 
                 space_n_patches: int):
        """
        Args:
            emb_size (int): Patch embedding size 
            time_n_patches (int): Number of patches on the time-axis
            space_n_patches (int): Number of patches in the space-axis (flattened)
        """
              
        super().__init__()
        self.emb_size = emb_size
        
        # register the positions, automatically move them to the super module's device
        self.register_buffer('time_space_positions', 
                             self._get_time_space_positions(time_n_patches, 
                                                            space_n_patches))
        
    def forward(self, x: Tensor) -> Tensor:
        """Add patches time-space positions

        Args:
            x (Tensor): Embedded patches of shape (batch_size, n_patches, emb_size)

        Returns:
            Tensor: Input patches with positions added
        """
        return x + self.time_space_positions
        
    
    def _get_positions(self, n_pos: int, n: int = 10_000) -> Tensor:
        """Compute sin-cos positional tokens

        Args:
            n_pos (int): Number of positions to encode
            n (int, optional): Denominator factor. Defaults to 10_000.

        Returns:
            Tensor: Positional tokens of shape (n_pos, emb_size)
        """
        positions = torch.zeros(n_pos, self.emb_size)
        for i in range( int(self.emb_size / 2) ):
            denominator = np.power(n, 2*i / self.emb_size)
            for k in range(n_pos):
                positions[k, 2*i] = np.sin(k/denominator)
                positions[k, 2*i + 1] = np.cos(k/denominator)
        return positions
    
    def _get_time_space_positions(self, 
                                  time_n_patches: int, 
                                  space_n_patches: int) -> Tensor:
        """Compute time-space position tokens. It is acheived by summing the time and space positions.

        Args:
            time_n_patches (int): Number of patches on the time-axis 
            space_n_patches (int): Number of patches on the space-axis

        Returns:
            Tensor: Time-space positional tokens of shape (time_n_patches*space_n_patches, emb_size)
        """
        
        time_positions = self._get_positions(time_n_patches) # (time_n_patches x emb_size)
        space_positions = self._get_positions(space_n_patches) # (space_n_patches x emb_size)
        
        time_unsq = torch.unsqueeze(time_positions, 1) # (time_n_patches x 1 x emb_size)
        space_unsq = torch.unsqueeze(space_positions, 0) # (1 x space_n_patches x emb_size)

        time_exp = torch.cat(space_n_patches * [time_unsq], 1) # both (time_n_patches x space_n_patches x emb_size)
        space_exp = torch.cat(time_n_patches * [space_unsq], 0) 
        
        mixed_positions = torch.flatten(time_exp + space_exp, 0, 1) # (n_patches x emb_size)
        mixed_positions.requires_grad = False
        
        return mixed_positions
        

class MultiHeadAttention(nn.Module):
    """
    MultiHead self-attention module
    """
    
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    """
    Residual connection module
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    """
    Two-layers feed-forward module
    """
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class AttentionBlock(nn.Sequential):
    """
    Attention block, with residual connections between the multi-head attention module
    """
    def __init__(self,
                 emb_size: int,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 att_num_heads: int = 4,
                 att_drop_p: float = 0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=att_num_heads, dropout=att_drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )))
    
class AttentionNet(nn.Sequential):
    """
    Create a sequence of stacked attention blocks
    """
    def __init__(self, emb_size: int, depth: int = 8, **kwargs):
        """
        Args:
            emb_size (int): Patch embedding size
            depth (int, optional): Number of stacked attention blocks. Defaults to 8.
        """
        super().__init__(*[AttentionBlock(emb_size, **kwargs) for _ in range(depth)])
        
        
class MaskedAutoEncoder(nn.Module):
    """
    Transformer-based encoder and decoder, jointly trained on a masking task.
    """
    
    def __init__(self,
                 in_channels: int = 5,
                 space_dim: int = 32,
                 time_dim: int = 10,
                 space_patch_size: int = 4, 
                 time_patch_size: int = 2,
                 emb_size: int = 512, 
                 masking_ratio: float = 0.8,
                 encoder_depth: int = 8, 
                 decoder_depth: int = 2, 
                 **kwargs):
        """
        Args:
            in_channels (int, optional): Number of channels in the video. Defaults to 5.
            space_dim (int, optional): Size of the space dimension in the video. Defaults to 32.
            time_dim (int, optional): Size of the time dimension in the video. Defaults to 10.
            space_patch_size (int, optional): Size of a patch in the space dimension. Defaults to 4.
            time_patch_size (int, optional): Size of a patch in the time dimension. Defaults to 2.
            emb_size (int, optional): Patch embedding size . Defaults to 512.
            masking_ratio (float, optional): Ratio of masked patches. Defaults to 0.8.
            encoder_depth (int, optional): Number of stacked attention blocks in the encoder. Defaults to 8.
            decoder_depth (int, optional): Number of stacked attention blocks in the decoder. Defaults to 2.
        """
        
        super().__init__()
        
        assert space_dim % space_patch_size == 0, 'Space patch size must divide space dimension'
        assert time_dim % time_patch_size == 0, 'Time patch size must divide time dimension'
        assert emb_size % 2 == 0, 'Embedding size must be a factor of 2'

        self.emb_size = emb_size
        
        self.encoder = AttentionNet(emb_size, encoder_depth, **kwargs)
        self.decoder = AttentionNet(emb_size, decoder_depth, **kwargs)
        
        self.time_patch_size = time_patch_size
        self.space_patch_size = space_patch_size
        
        self.time_n_patches = int(time_dim / time_patch_size)
        self.space_n_patches = int(space_dim / space_patch_size) ** 2
        
        self.n_patches = self.time_n_patches * self.space_n_patches
        self.pos_adder = PositionalAdder(emb_size, self.time_n_patches, self.space_n_patches)
        
        self.patch_transform = PatchTransform(emb_size, 
                                              in_channels, 
                                              space_patch_size,
                                              time_patch_size)
    
        self.pixel_transform = PixelTransform(emb_size,
                                              in_channels, 
                                              space_dim,
                                              time_dim,
                                              space_patch_size, 
                                              time_patch_size)
        
        self.mask_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.masking_ratio = masking_ratio
                
    def forward(self, x: Tensor) -> Tensor: # (batch_size, time, n_channels, x, y)
        
        encoding, shuffling_idx = self.encode(x, return_idx=True)
        reconstruction = self.decode(encoding, shuffling_idx)
        
        return reconstruction
    
    def encode(self, x: Tensor, return_idx: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Randomly mask (discard) patches from the input video 
        and feed the selected ones to the encoder

        Args:
            x (Tensor): Input video of shape (batch_size, time, n_channels, x, y)
            return_idx (bool): If the shuffling indices are returned alongside the encoded patches

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output of selected patches through the decoder 
            (plus shuffling indices if specified)
        """
        
        patches = self.patch_transform(x) # (batch_size, n_patches, emb_size)
        patches_w_pos = self.pos_adder(patches)
        
        batch_size = patches_w_pos.size(0)

        # shuffle patches independently for each element in the batch
        r = torch.rand(batch_size, self.n_patches)
        idx = torch.argsort(torch.rand(*r.shape), dim=-1) # order of patches within element in the batch
        offset = (torch.arange(batch_size) * self.n_patches) \
            .unsqueeze(1).repeat(1, self.n_patches) # offset: position_in_batch * n_patches
        idx += offset
        idx = torch.flatten(idx)
        
        shuffled = torch.flatten(patches_w_pos, 0, 1)[idx, :] \
            .view(batch_size, self.n_patches, -1)
                        
        # keep only first patches 
        n_masked = int(self.n_patches * self.masking_ratio)
        keep = self.n_patches - n_masked
        masked = shuffled[:, :keep, :]
        
        encoding = self.encoder(masked)
        
        if return_idx:
            return encoding, idx
        else:
            return encoding
            
    def decode(self, x: Tensor, idx: Tensor) -> Tensor: 
        """
        Reconstruct the original video from selected encoded patches, 
        add mask tokens at discarded patches positions

        Args:
            x (Tensor): Encoded selected patches of shape (batch_size, kept_patches, emb_size)
            idx (Tensor): Indices used for shuffling patches pior to the encoder of shape (batch_size * n_patches)

        Returns:
            Tensor: Reconstructed video of shape (batch_size, time, n_channels, x, y)
        """
        
        batch_size = x.size(0)
        
        # add mask tokens at the end
        n_masked = int(self.n_patches * self.masking_ratio)
        mask_tokens = self.mask_token.repeat(batch_size, n_masked, 1)
        shuffled = torch.cat((x, mask_tokens), dim=1) # (batch_size, n_patches, emb_size)
        shuffled = torch.flatten(shuffled, 0, 1)
        
        # unshuffle the patches w.r.t to shuffling indices
        unshuffled = torch.zeros_like(shuffled)
        unshuffled[idx] = shuffled
        unshuffled = unshuffled.view(batch_size, self.n_patches, -1)
        
        # add position 
        encoding_w_pos = self.pos_adder(unshuffled)
        decoding = self.decoder(encoding_w_pos)
        
        # reconstruct pixels from each patch
        reconstruction = self.pixel_transform(decoding)
        
        return reconstruction
        
            
        

            
        
        

    
