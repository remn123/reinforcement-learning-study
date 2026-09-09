import torch
import numpy
import myrl.algorithms.ppo.constants as constants

from myrl.algorithms.ppo.utils import (
    calculate_discounted_rewards,
    calculate_delta,
    calculate_advantage_function,
    compute_actor_clip_loss,
    compute_value_loss,
)
from torch.distributions.categorical import Categorical


@torch.no_grad
def get_batches(observations, batch_size):
    n_observations = observations.size(0)
    batches = []
    start = 0
    end = 0
    indices = torch.randperm(n_observations)
    while end < n_observations:
        end = start+batch_size
        if end > n_observations:
            end = n_observations
        idx_batch = indices[start:end]
        batches.append(idx_batch)
        start = end
    return batches

def train_micro_batch(
        model, 
        optimizer, 
        observations, 
        actions,
        advantage_function,
        discounted_rewards,
        n_optim_epochs, 
        n_optim_batch_size
    ):
    losses_actor = []
    losses_value = []
    for _ in range(n_optim_epochs): # K epochs from PPO original paper
        losses_actor_batches = []
        losses_value_batches = []
        idx_batches = get_batches(observations, n_optim_batch_size)
        for idx_batch in idx_batches: # M bacthes
            # Batches
            obs_batch = observations[idx_batch]
            actions_batch = actions[idx_batch]
            log_prob_old_batch = model.log_prob_old[idx_batch]
            advantage_function_batch = advantage_function[idx_batch]
            discounted_rewards_batch = discounted_rewards[idx_batch]
            # Run policy, value function
            logits_actor_batch, value_batch = model(obs_batch)
            log_prob_actor_batch = Categorical(
                logits=logits_actor_batch
            ).log_prob(actions_batch)

            # Calculate Loss
            loss_actor_batch = compute_actor_clip_loss(
                log_prob_actor_batch,
                log_prob_old_batch,
                advantage_function_batch
            )
            loss_value_batch = compute_value_loss(
                value_batch, 
                discounted_rewards_batch
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss_actor_batch.backward()
            loss_value_batch.backward()
            
            # Update / optimize
            optimizer.step()

            losses_actor_batches.append(loss_actor_batch.item())
            losses_value_batches.append(loss_value_batch.item())
        losses_actor.append(numpy.mean(losses_actor_batches))
        losses_value.append(numpy.mean(losses_value_batches))
    
    # Store old logits for next iteration
    # with torch.no_grad():
    #     logits_actor, _ = model(observations)
    #     model.logits_old = logits_actor.clone().detach()

    return numpy.mean(losses_actor), numpy.mean(losses_value)


def train_step(
        model, 
        env, 
        optimizer, 
        n_episodes,
        n_optim_epochs,
        n_optim_batch_size,
    ):
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
                discounted_reward = calculate_discounted_rewards(rewards)
                discounted_rewards += discounted_reward
                delta = calculate_delta(rewards, values)
                advantage_function += calculate_advantage_function(delta)
                
                rewards_total += sum(rewards)
                rewards = []
                values = []
                
                # End of the trajectories sampling
                if episodes >= n_episodes:
                    done = True
                obs, _ = env.reset()

    observations_tensor = torch.tensor(
        numpy.array(observations), 
        dtype=torch.float32
    )
    actions_tensor = torch.as_tensor(
        numpy.array(actions), 
        dtype=torch.int64
    )
    advantage_function_tensor = torch.as_tensor(
        numpy.array(advantage_function), 
        dtype=torch.float32
    )
    discounted_rewards_tensor = torch.as_tensor(
        numpy.array(discounted_rewards), 
        dtype=torch.float32
    )
    
    with torch.no_grad():
        logits_actor, _ = model(observations_tensor)
        model.log_prob_old = Categorical(logits=logits_actor).log_prob(actions_tensor)
    
    loss_actor, loss_value = train_micro_batch(
        model,
        optimizer,
        observations_tensor,
        actions_tensor,
        advantage_function_tensor,
        discounted_rewards_tensor,
        n_optim_epochs,
        n_optim_batch_size
    )

    rewards_mean = rewards_total/n_episodes
    
    return loss_actor, loss_value, rewards_mean


def train_loop(
        model, 
        env, 
        optimizer, 
        n_episodes=100, 
        n_epochs=100,
        n_optim_epochs=10,
        n_optim_batch_size=64,
    ):
    rewards = [] 
    losses_actor = []
    losses_value = []
    block = int(n_epochs//10)
    for i in range(n_epochs):
        loss_actor, loss_value, reward = train_step(
            model, 
            env, 
            optimizer,
            n_episodes,
            n_optim_epochs,
            n_optim_batch_size
        )
        losses_actor.append(loss_actor)
        losses_value.append(loss_value)
        rewards.append(reward)
        if i % block == 0:
            print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    print(f"Epoch ({i}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    env.close()
    return losses_actor, losses_value, rewards