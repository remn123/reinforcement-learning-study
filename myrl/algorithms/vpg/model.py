import torch
import torch.nn as nn


class VanillaPolicyGradient(nn.Module):

    def __init__(self, obs_dim, actions_dim, n_hidden=3, hidden_dim=64):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            in_features=obs_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            out_features=actions_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(obs)
        return logits