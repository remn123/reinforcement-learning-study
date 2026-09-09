import torch
import numpy


def compute_value_loss(value, discounted_rewards):
    discounted_rewards_tensor = torch.as_tensor(discounted_rewards, dtype=torch.float32)
    loss_value = ((value - discounted_rewards_tensor)**2.0).mean()
    return loss_value

def compute_actor_loss(log_probs, advantage_function):
    advantage_tensor = torch.as_tensor(advantage_function, dtype=torch.float32)
    return -(log_probs * advantage_tensor).mean()


@torch.no_grad
def calculate_discounted_rewards(rewards: numpy.ndarray, gamma: float=0.95) -> list:
    rewards_size = len(rewards)
    rewards_to_go = rewards_size*[0.0]
    for l in reversed(range(rewards_size)):
        future_rewards = 0.0
        if l+1 < rewards_size:
            future_rewards = rewards_to_go[l+1]
        rewards_to_go[l] = rewards[l] + gamma*future_rewards
    return rewards_to_go

@torch.no_grad
def calculate_delta(
        rewards: numpy.ndarray, 
        values: numpy.ndarray, 
        gamma: float=0.95
    ) -> list:
    rewards_size = len(rewards)
    delta = numpy.copy(rewards)
    for l in reversed(range(rewards_size)):
        values_tp1 = 0.0
        if l+1 < rewards_size:
            values_tp1 = values[l+1]
        delta[l] += gamma*values_tp1 - values[l]
    return delta
    
@torch.no_grad
# Generalized Advantage Estimation (GAE)
# High-Dimensional Continuous Control Using Generalized Advantage Estimation
# https://arxiv.org/abs/1506.02438
def calculate_advantage_function(
        delta: numpy.ndarray,
        gamma: float, 
        lambd: float
    ) -> list:
    delta_size = len(delta)
    advantage_function = delta_size*[0.0]
    for i in reversed(range(delta_size)):
        future_advantage_function = 0.0
        if i+1 < delta_size:
            future_advantage_function = advantage_function[i+1]
        advantage_function[i] = delta[i] + (gamma*lambd)*future_advantage_function
    return advantage_function