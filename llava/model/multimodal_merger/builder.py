import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft
from einops import rearrange, repeat, einsum
import math
import numpy as np
from torch.nn.init import trunc_normal_
from transformers.models.clip.modeling_clip import CLIPAttention

def build_merger(config, vision_config, **kwargs):
    return Merger(config, vision_config)

class Merger(nn.Module):
    def __init__(self, config, vision_config):
        super().__init__()
        self.config = config
        if config.mm_merger_type == 'afno2d':
            self.fft_block = AFNO2D(config.mm_hidden_size)
        else:
            self.fft_block = CLIPAttention(vision_config)

        self.down_layer = DownLayer(config.mm_hidden_size) if config.mm_merger_conv else nn.Identity()
        self.norm1 = nn.LayerNorm(config.mm_hidden_size)
        self.norm2 = nn.LayerNorm(config.mm_hidden_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x shape: (C, H, W)
        x = x.permute(1, 2, 0).unsqueeze(0) # B, H, W, C
        if self.config.mm_merger_type == 'afno2d':
            x = x + self.fft_block(self.norm1(x), H, W)
            x = self.norm2(x)
            x = x.squeeze(0).permute(2, 0, 1) # C, H, W
        else:
            x = x.flatten(1, 2) # B, N, C
            x = x + self.fft_block(self.norm1(x))[0]
            x = self.norm2(x)
            x = x.squeeze(0).view(H, W, -1).permute(2, 0, 1) # C, H, W

        x = self.down_layer(x)
        return x

class EinFFT(nn.Module):
    r"""
    EinFFT from Simba: https://github.com/badripatro/simba/blob/main/classification/simba.py.
    """
    def __init__(self, dim):
        super().__init__()


        self.hidden_size = dim #768
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)
        self.norm = nn.LayerNorm(self.hidden_size)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        # B, C, H, W = x.shape
        x = x.flatten(2, 3).transpose(1, 2) # B, N, C

        residual = x
        x = self.norm(x)
        B, N, C = x.shape
        origin_type = x.dtype
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = x.to(torch.float32)
        x = torch.fft.fft2(x, dim=(1,2), norm='ortho') # FFT on N dimension


        x_real_1 = F.relu(self.multiply(x.real.to(origin_type), self.complex_weight_1[0]) - self.multiply(x.imag.to(origin_type), self.complex_weight_1[1]) + self.complex_bias_1[0])
        x_imag_1 = F.relu(self.multiply(x.real.to(origin_type), self.complex_weight_1[1]) + self.multiply(x.imag.to(origin_type), self.complex_weight_1[0]) + self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1, self.complex_weight_2[1]) + self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1, self.complex_weight_2[0]) + self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1,2), norm="ortho")
        
        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(origin_type)
        x = x.reshape(B, N, C)
        x = x + residual
        x = x.transpose(1, 2) # B, C, N
        x = x.view(B, C, H, W).squeeze(0).contiguous()
        return x

class DownLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_merge = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        x = self.conv_merge(x)
        return x
    
class FFTFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.hidden_size = dim #768
        self.scale = 0.02
        self.complex_weight = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, 2, dtype=torch.float32) * self.scale)


    # def multiply(self, input, weights):
    #     return torch.einsum('...bd,bdk->...bk', input, weights)
    
    def forward(self, x, H, W):
        C, a, b = x.shape
        x = x.permute(1, 2, 0) # H, W, C
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(0, 1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(0, 1), norm='ortho')

        return x.permute(2, 0, 1).contiguous()
    
class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=4, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, H, W):
        dtype = x.dtype
        x = x.float()
        B, _, _, C = x.shape
        N = H * W

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], dtype=dtype, device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], dtype=dtype, device=x.device)
        o2_real = torch.zeros(x.shape, dtype=dtype, device=x.device)
        o2_imag = torch.zeros(x.shape, dtype=dtype, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)


        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real.to(dtype), self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag.to(dtype), self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag.to(dtype), self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real.to(dtype), self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x.float())
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)
        return x