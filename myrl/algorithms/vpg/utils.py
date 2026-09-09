import torch


@torch.no_grad
def calculate_rewards_to_go(rewards: list) -> list:
    rewards_size = len(rewards)
    rewards_to_go = rewards_size*[0.0]
    for i in reversed(range(rewards_size)):
        future_rewards = 0.0
        if i+1 < rewards_size:
            future_rewards = rewards_to_go[i+1]
        rewards_to_go[i] = rewards[i] + future_rewards
    return rewards_to_go