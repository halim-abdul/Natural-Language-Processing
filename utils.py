import torch
from torch import nn, Tensor

def get_parameter_dtype(module: nn.Module):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32

def get_extended_attention_mask(attention_mask: Tensor, dtype) -> Tensor:
    assert attention_mask.dim() == 2
    extended_attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - extended_attention_mask) * -10000.0
