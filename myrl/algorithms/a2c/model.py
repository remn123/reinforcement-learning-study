import torch
import torch.nn as nn

from myrl.algorithms.mlp.model import MultiLayerPerceptron
from typing import Tuple


class AdvantageActorCriticPolicyGAE(nn.Module):

    def __init__(self, obs_dim, actions_dim, n_hidden=3, hidden_dim=64, gamma=0.95, lambd=0.99):
        super().__init__()
        self.actor = MultiLayerPerceptron(
            in_features=obs_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            out_features=actions_dim,
        )
        self.value = MultiLayerPerceptron(
            in_features=obs_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            out_features=1,
        )
        self.gamma = gamma
        self.lambd = lambd
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_actor = self.actor(obs)
        value = self.value(obs).squeeze(-1)
        return logits_actor, value
