import torch
import numpy

import myrl.algorithms.ppo.constants as constants


def compute_value_loss(value, discounted_rewards):
    loss_value = ((value - discounted_rewards)**2.0).mean()
    return loss_value

def compute_actor_loss(log_probs, advantage_function):
    return -(log_probs * advantage_function).mean()

def compute_actor_clip_loss(
        log_prob, 
        log_prob_old, 
        advantage_function
    ):
    upper = (1.0+constants.EPS)*torch.ones_like(advantage_function)
    lower = (1.0-constants.EPS)*torch.ones_like(advantage_function)
    probability_ratio = torch.exp(log_prob - log_prob_old)
    clip = torch.where(
        advantage_function > 0.0,
        torch.min(upper, probability_ratio),
        torch.max(lower, probability_ratio)
    )
    return -(clip*advantage_function).mean()

@torch.no_grad
def calculate_discounted_rewards(rewards: numpy.ndarray) -> list:
    rewards_size = len(rewards)
    rewards_to_go = rewards_size*[0.0]
    for l in reversed(range(rewards_size)):
        future_rewards = 0.0
        if l+1 < rewards_size:
            future_rewards = rewards_to_go[l+1]
        rewards_to_go[l] = rewards[l] + constants.GAMMA*future_rewards
    return rewards_to_go

@torch.no_grad
def calculate_delta(
        rewards: numpy.ndarray, 
        values: numpy.ndarray
    ) -> list:
    rewards_size = len(rewards)
    delta = numpy.copy(rewards)
    for l in reversed(range(rewards_size)):
        values_tp1 = 0.0
        if l+1 < rewards_size:
            values_tp1 = values[l+1]
        delta[l] += constants.GAMMA*values_tp1 - values[l]
    return delta
    
@torch.no_grad
# Generalized Advantage Estimation (GAE)
# High-Dimensional Continuous Control Using Generalized Advantage Estimation
# https://arxiv.org/abs/1506.02438
def calculate_advantage_function(delta: numpy.ndarray) -> list:
    delta_size = len(delta)
    advantage_function = delta_size*[0.0]
    for i in reversed(range(delta_size)):
        future_advantage_function = 0.0
        if i+1 < delta_size:
            future_advantage_function = advantage_function[i+1]
        advantage_function[i] = delta[i] + (
            constants.GAMMA*constants.LAMBDA
        )*future_advantage_function
    return advantage_function


@torch.no_grad
def calculate_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.float32:
    KL_div = p @ (torch.log(p)-torch.log(q)).t()
    return KL_div