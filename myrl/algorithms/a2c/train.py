import torch
import numpy

from myrl.algorithms.a2c.utils import (
    calculate_discounted_rewards,
    calculate_delta,
    calculate_advantage_function,
    compute_actor_loss,
    compute_value_loss,
)
from torch.distributions.categorical import Categorical


def train_step(model, env, optimizer, n_episodes=100):
    rewards = []
    discounted_rewards = []
    advantage_function = []
    observations = []
    actions = []
    values = []
    rewards_total = 0
    episodes = 0
    
    done = False
    obs, _ = env.reset()
    # Run trajectories in the environment (N)
    while not done:
        with torch.no_grad():
            observations.append(obs.copy())
            obs_tensor = torch.from_numpy(obs).float()
            
            logits_actor, value = model(obs_tensor)
            a_policy = Categorical(logits=logits_actor)
            action = a_policy.sample()
            
            actions.append(action.item())
            values.append(value.item())
            
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            # Store vars
            rewards.append(reward)
            
            # End of a trajectory
            if done:
                done=False
                episodes += 1
                # Calculate Rewards-to-go
                discounted_reward = calculate_discounted_rewards(rewards, gamma=model.gamma)
                discounted_rewards += discounted_reward
                delta = calculate_delta(rewards, values, gamma=model.gamma)
                advantage_function += calculate_advantage_function(
                    delta,
                    gamma=model.gamma,
                    lambd=model.lambd,
                )
                
                #weights += [sum(rewards)]*len(rewards)
                rewards_total += sum(rewards)
                rewards = []
                values = []
                
                # End of the trajectories sampling
                if episodes >= n_episodes:
                    done = True
                obs, _ = env.reset()
    # Calculate Loss
    observations = torch.tensor(numpy.array(observations), dtype=torch.float32)
    
    logits_actor, value = model(observations)
    log_probs_actor = Categorical(logits=logits_actor).log_prob(
        torch.as_tensor(actions, dtype=torch.int64)
    )
        
    loss_actor = compute_actor_loss(log_probs_actor, advantage_function)
    loss_value = compute_value_loss(value, discounted_rewards)
    # Backpropagation
    optimizer.zero_grad()
    loss_actor.backward()
    loss_value.backward()
    
    # Update / optimize
    optimizer.step()
    return loss_actor.item(), loss_value.item(), rewards_total


def train_loop(model, env, optimizer, n_episodes=100, n_epochs=100):
    rewards = [] 
    losses_actor = []
    losses_value = []
    block = int(n_epochs//10)
    for i in range(n_epochs):
        loss_actor, loss_value, reward = train_step(
            model, 
            env, 
            optimizer,
            n_episodes
        )
        losses_actor.append(loss_actor)
        losses_value.append(loss_value)
        rewards.append(reward)
        if i % block == 0:
            print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    env.close()
    return losses_actor, losses_value, rewards