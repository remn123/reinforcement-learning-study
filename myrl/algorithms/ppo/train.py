import torch
import numpy

from myrl.algorithms.ppo.utils import (
    calculate_discounted_rewards,
    calculate_delta,
    calculate_advantage_function,
    compute_actor_clip_loss,
    compute_value_loss,
    calculate_kl_divergence,
)
from typing import Dict


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
        writer,
        data,
        n_optim_epochs, 
        n_optim_batch_size
    ):
    losses_actor = []
    losses_value = []
    policy_update_indice = 1
    for _ in range(n_optim_epochs): # K epochs from PPO original paper
        losses_actor_batches = []
        losses_value_batches = []
        idx_batches = get_batches(data['observations'], n_optim_batch_size)
        for idx_batch in idx_batches: # M bacthes
            # Batches
            obs_batch = data['observations'][idx_batch]
            actions_batch = data['actions'][idx_batch]
            log_prob_old_batch = model.log_prob_old[idx_batch]
            advantage_function_batch = data['advantage'][idx_batch]
            discounted_rewards_batch = data['discounted_rewards'][idx_batch]
            # Run policy, value function
            logits_actor_batch, value_batch = model(obs_batch)
            log_prob_actor_batch = model.log_prob(
                logits=logits_actor_batch,
                actions=actions_batch,
            )
            # Compute Policy Update KL Divergence
            kl_divergence_policies = calculate_kl_divergence(
                log_prob_actor_batch,
                log_prob_old_batch,
            )
            
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
            
            with torch.no_grad():
                writer.add_scalar(
                    'KLDivergencePolicy/update',
                    kl_divergence_policies,
                    policy_update_indice,
                )
                writer.add_scalar(
                    'LossActorBatch/update',
                    loss_actor_batch,
                    policy_update_indice,
                )
                writer.add_scalar(
                    'LossValueBatch/update',
                    loss_value_batch,
                    policy_update_indice,
                )
            
            # Backpropagation
            optimizer.zero_grad()
            loss_actor_batch.backward()
            loss_value_batch.backward()
            
            # Update / optimize
            optimizer.step()

            losses_actor_batches.append(loss_actor_batch.item())
            losses_value_batches.append(loss_value_batch.item())
            policy_update_indice += 1
        losses_actor.append(numpy.mean(losses_actor_batches))
        losses_value.append(numpy.mean(losses_value_batches))
    return numpy.mean(losses_actor), numpy.mean(losses_value)


def data_collection(model, env, writer, n_episodes) -> Dict[str, torch.Tensor]:
    rewards = []
    discounted_rewards = []
    advantage_functions = []
    observations = []
    actions = []
    values = []
    total_rewards = 0.0
    episodes = 0
    
    done = False
    obs, _ = env.reset()
    # Run trajectories in the environment (N)
    while not done:
        with torch.no_grad():
            observations.append(obs.copy())
            obs_tensor = torch.from_numpy(obs).float()
            
            logits_actor, value = model(obs_tensor)
            action: torch.Tensor = model.sample_action(logits=logits_actor)
            
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
                advantage_function = calculate_advantage_function(delta)
                advantage_functions += advantage_function
                
                total_rewards += sum(rewards)
                writer.add_scalar(
                    "TotalRewardPerEpisode/collection", 
                    total_rewards, 
                    episodes
                )
                writer.add_scalar(
                    "AverageAdvantageArrayPerEpisode/collection",
                    numpy.mean(advantage_function),
                    episodes
                )
                writer.add_scalar(
                    "VarianceAdvantageArrayPerEpisode/collection",
                    numpy.std(advantage_function),
                    episodes
                )
                writer.add_scalar(
                    "AverageDiscountedRewardArrayPerEpisode/collection",
                    numpy.mean(discounted_reward),
                    episodes
                )
                writer.add_scalar(
                    "VarianceDiscountedRewardArrayPerEpisode/collection",
                    numpy.std(discounted_reward),
                    episodes
                )
                
                rewards = []
                values = []
                
                # End of the trajectories sampling
                if episodes >= n_episodes:
                    done = True
                obs, _ = env.reset()
    
    data = {
        'observations': torch.tensor(
            numpy.array(observations), 
            dtype=torch.float32
        ),
        'actions': torch.as_tensor(
            numpy.array(actions), 
            dtype=torch.int64
        ),
        'advantage': torch.as_tensor(
            numpy.array(advantage_functions), 
            dtype=torch.float32
        ),
        'discounted_rewards': torch.as_tensor(
            numpy.array(discounted_rewards), 
            dtype=torch.float32
        ),
        'total_rewards': torch.as_tensor(
            numpy.array(total_rewards), 
            dtype=torch.float32
        ),
    }
    
    return data

def train_step(
        model, 
        env, 
        optimizer,
        writer, 
        n_episodes,
        n_optim_epochs,
        n_optim_batch_size,
    ):
    
    data = data_collection(model, env, writer, n_episodes)
    
    model.save_log_prob_old(
        observations=data['observations'], 
        actions=data['actions']
    )
    
    loss_actor, loss_value = train_micro_batch(
        model,
        optimizer,
        writer,
        data,
        n_optim_epochs,
        n_optim_batch_size,
    )

    with torch.no_grad():
        rewards_mean = data['total_rewards'].mean().item()
    # I just noticed that I have been reporting the reward from the old policy
    # So I believe the best scenario would be to re-run the entire new policy
    # over the same observations, however, it seems that most of the RL 
    # libraries implement in this way too. So I'll keep it.
    
    return loss_actor, loss_value, rewards_mean


def train_loop(
        model, 
        env, 
        optimizer,
        writer=None,
        n_episodes=100, 
        n_epochs=100,
        n_optim_epochs=10,
        n_optim_batch_size=64,
    ):
    rewards = [] 
    losses_actor = []
    losses_value = []
    block = int(n_epochs//10)
    for epoch in range(n_epochs):
        loss_actor, loss_value, reward = train_step(
            model, 
            env, 
            optimizer,
            writer,
            n_episodes,
            n_optim_epochs,
            n_optim_batch_size
        )
        writer.add_scalar("LossActor/train", loss_actor, epoch)
        writer.add_scalar("LossValue/train", loss_value, epoch)
        writer.add_scalar("Reward/train", reward, epoch)
        losses_actor.append(loss_actor)
        losses_value.append(loss_value)
        rewards.append(reward)
        if epoch % block == 0:
            print(f"Epoch ({epoch}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    print(f"Epoch ({epoch}/{n_epochs}): Reward = {reward}, Loss Actor = {loss_actor}, Loss Value = {loss_value}")
    
    env.close()
    writer.flush()
    return losses_actor, losses_value, rewards