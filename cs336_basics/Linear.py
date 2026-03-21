import torch
import torch.nn as nn



class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2 / (in_features + out_features)) ** 0.5
        self.weights = nn.parameter.Parameter(
            nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            mean=0,
            std=std,
            a= -3*std,
            b=3*std
            )
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights @ x

