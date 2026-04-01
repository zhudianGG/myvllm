import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    @torch.compile
    def add_rms_forward(self, x, residual):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float()) # add residual
        residual = x.to(orig_dtype) # update residual for next layer

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtpe).mul_(self.weight)
        return x, residual

