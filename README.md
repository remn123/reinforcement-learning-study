# myrl

A personal study repository for implementing classic and modern Reinforcement Learning algorithms **from scratch**, using PyTorch. The goal isn't to build a production-grade RL library, but to work through the math and mechanics of each algorithm by writing it end-to-end — including, where it makes sense, hand-rolled components like a custom gradient descent optimizer instead of relying on `torch.optim`.

## Package structure

```
myrl/
├── algorithms/
│   ├── mlp/         # Shared MLP backbone used by every policy/value network
│   ├── vpg/         # Vanilla Policy Gradient
│   ├── a2c/         # Advantage Actor-Critic with GAE
│   ├── ppo/         # Proximal Policy Optimization
│   └── trpo/        # Trust Region Policy Optimization (in progress)
└── optimizers/
    └── gradient_descent.py   # From-scratch policy-gradient optimizer
```

Each algorithm module follows the same shape:
- `model.py` — the network(s) (actor / actor-critic)
- `train.py` — `train_step` (one epoch of rollout + update) and `train_loop` (the outer training loop)
- `utils.py` — the math: returns, advantages, losses

## Implemented algorithms

### Vanilla Policy Gradient (VPG)
`myrl/algorithms/vpg`

The baseline REINFORCE-style policy gradient. Collects full episodes, computes **rewards-to-go** as the learning signal, and updates the policy by ascending the log-probability-weighted return.

### Advantage Actor-Critic with GAE (A2C)
`myrl/algorithms/a2c`

Adds a learned value function (critic) alongside the policy (actor), and replaces the raw return with **Generalized Advantage Estimation (GAE)** to trade off bias and variance in the advantage estimate. Trains actor and value losses jointly each step.

### Proximal Policy Optimization (PPO)
`myrl/algorithms/ppo`

Builds on the actor-critic + GAE setup and replaces the vanilla policy gradient objective with PPO's **clipped surrogate objective**, keeping the updated policy close to the policy that generated the data. Rollouts are collected once per epoch, then optimized over several inner epochs and mini-batches (`train_micro_batch`), following the original PPO paper's K-epochs/M-batches update scheme.

### Trust Region Policy Optimization (TRPO) — in progress
`myrl/algorithms/trpo`

Model scaffolding (actor-critic with GAE, mirroring A2C/PPO) is in place, but the constrained (KKT / conjugate-gradient + line search) update step itself is not implemented yet.

### Group Relative Policy Optimization (GRPO) — planned
`myrl/algorithms/grpo` *(not yet created)*

A PPO variant that drops the learned critic entirely, instead estimating the advantage by normalizing rewards within a group of sampled outputs/rollouts for the same input — reducing memory and compute versus PPO.

### Memory-R1 — planned
*(implementation reference, not a from-scratch algorithm)*

An RL framework (not a new policy-gradient algorithm on its own) that fine-tunes a **Memory Manager** (ADD/UPDATE/DELETE/NOOP operations) and an **Answer Agent** with outcome-driven RL — using PPO and GRPO as the underlying optimizers — to teach LLM agents to manage and use external memory.

## Custom optimizer

`myrl/optimizers/gradient_descent.py` implements `PolicyGradientOptimizer`, a minimal from-scratch gradient descent step (`zero_grad` / `step`) used in place of `torch.optim`, to keep the update mechanics fully explicit.

## Usage

Each algorithm exposes the same basic training entry point:

```python
from myrl.algorithms.vpg.model import VanillaPolicyGradient
from myrl.algorithms.vpg.train import train_loop
from myrl.optimizers.gradient_descent import PolicyGradientOptimizer
import gymnasium as gym

env = gym.make("CartPole-v1")
model = VanillaPolicyGradient(obs_dim=4, actions_dim=2)
optimizer = PolicyGradientOptimizer(model.parameters(), alpha=1e-2)

losses, rewards = train_loop(model, env, optimizer, n_episodes=100, n_epochs=100)
```

PPO's `train_loop` takes two extra arguments for the inner optimization loop:

```python
from myrl.algorithms.ppo.model import ProximalPolicyOptimization
from myrl.algorithms.ppo.train import train_loop

model = ProximalPolicyOptimization(obs_dim=4, actions_dim=2)
losses_actor, losses_value, rewards = train_loop(
    model, env, optimizer,
    n_episodes=100, n_epochs=100,
    n_optim_epochs=10, n_optim_batch_size=64,
)
```

## References

| Algorithm / Implementation | Paper | Authors | Year |
|---|---|---|---|
| VPG (REINFORCE) | [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) | Williams | 1992 |
| GAE | [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (arXiv:1506.02438) | Schulman, Moritz, Levine, Jordan, Abbeel | 2015 |
| TRPO | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (arXiv:1502.05477) | Schulman, Levine, Moritz, Jordan, Abbeel | 2015 |
| A2C | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (arXiv:1602.01783) | Mnih et al. | 2016 |
| PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (arXiv:1707.06347) | Schulman, Wolski, Dhariwal, Radford, Klimov | 2017 |
| GRPO | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (arXiv:2402.03300) | Shao et al. | 2024 |
| Memory-R1 | [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828) (arXiv:2508.19828) | Yan et al. | 2025 |

## Roadmap

- [x] Vanilla Policy Gradient
- [x] Generalized Advantage Estimation in Advantage Actor-Critic (A2C)
- [ ] Trust Region Policy Optimization (TRPO)
- [x] Proximal Policy Optimization (PPO)
- [ ] Group Relative Policy Optimization (GRPO)
- [ ] Memory-R1
- [ ] Continuous action spaces (Gaussian policy — mean, std)
- [ ] Suite of RL validation metrics in PyTorch TensorBoard

## Status

This is an active, ongoing study project — code and structure will keep evolving as new algorithms are added.
