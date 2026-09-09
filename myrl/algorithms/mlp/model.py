import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        hidden_dim: int,
        n_hidden: int,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Linear(
                in_features=in_features,
                out_features=hidden_dim,
                bias=True,
            )
        )
        layers.append(nn.Tanh())
        for i in range(n_hidden):
            layers.append(
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    bias=True,
                )
            )
            layers.append(nn.Tanh())
        layers.append(
            nn.Linear(
                in_features=hidden_dim,
                out_features=out_features,
                bias=True,
            )
        )
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)