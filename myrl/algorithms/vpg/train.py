import torch
import numpy

from torch.distributions.categorical import Categorical
from myrl.algorithms.vpg.utils import (
    calculate_rewards_to_go,
)


def train_step(model, env, optimizer, n_episodes=100):
    rewards = []
    weights = []
    observations = []
    actions = []
    rewards_total = 0
    episodes = 0
    
    done = False
    obs, _ = env.reset()
    # Run trajectories in the environment (N)
    while not done:
        with torch.no_grad():
            observations.append(obs.copy())
            obs_tensor = torch.from_numpy(obs).float()
            logits = model(obs_tensor)
            policy = Categorical(logits=logits)
            action = policy.sample()
            log_prob = policy.log_prob(action)
            
            actions.append(action.item())
            
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            # Store vars
            rewards.append(reward)
            
            # End of a trajectory
            if done:
                done=False
                episodes += 1
                # Calculate Rewards-to-go
                new_rewards = calculate_rewards_to_go(rewards)
                weights += new_rewards
                #weights += [sum(rewards)]*len(rewards)
                rewards_total += new_rewards[0]
                rewards = []
                
                # End of the trajectories sampling
                if episodes >= n_episodes:
                    done = True
                obs, _ = env.reset()
    # Calculate Loss
    observations = torch.tensor(numpy.array(observations), dtype=torch.float32)
    logits = model(observations)
    log_probs = Categorical(logits=logits).log_prob(
        torch.as_tensor(actions, dtype=torch.int64)
    )
    loss = -(log_probs * torch.as_tensor(weights, dtype=torch.float32)).mean()
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    
    # Update / optimize
    optimizer.step()
    return loss.item(), rewards_total


def train_loop(model, env, optimizer, n_episodes=100, n_epochs=100):
    rewards = [] 
    losses = []
    block = int(n_epochs//10)
    for i in range(n_epochs):
        loss, reward = train_step(
            model, 
            env, 
            optimizer,
            n_episodes
        )
        losses.append(loss)
        rewards.append(reward)
        if i % block == 0:
            print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss = {loss}")
    print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss = {loss}")
    env.close()
    return losses, rewards