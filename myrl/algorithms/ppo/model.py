import torch
import torch.nn as nn

from myrl.algorithms.mlp.model import MultiLayerPerceptron
from torch.distributions.categorical import Categorical
from typing import Tuple


class ProximalPolicyOptimization(nn.Module):

    def __init__(
            self, 
            obs_dim, 
            actions_dim, 
            n_hidden=2,
            hidden_dim=64, 
        ):
        super().__init__()
        self.actor = MultiLayerPerceptron(
            in_features=obs_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            out_features=actions_dim,
        )
        self.critic = MultiLayerPerceptron(
            in_features=obs_dim,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            out_features=1,
        )
        self.log_prob_old = None
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_actor = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits_actor, value
    
    @torch.no_grad
    def save_log_prob_old(
            self, 
            observations: torch.Tensor, 
            actions: torch.Tensor
        ) -> None:
        logits_actor, _ = self(observations)
        self.log_prob_old = self.log_prob(logits_actor, actions)
        
    def log_prob(
            self, 
            logits: torch.Tensor, 
            actions: torch.Tensor
        ) -> torch.Tensor:
        return Categorical(
            logits=logits
        ).log_prob(actions)
    
    @torch.no_grad
    def sample_action(
            self, 
            logits: torch.Tensor
        ) -> torch.Tensor:
        return Categorical(
            logits=logits
        ).sample()