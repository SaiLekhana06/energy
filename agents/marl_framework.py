"""Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and Decentralized Execution."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.slicing_agent import DynamicSlicingAgent
from agents.sleeping_agent import XAppSleepingAgent
from models.networks import CentralizedCriticNetwork


class RolloutBuffer:
    """Experience buffer for storing multi-agent on-policy rollouts."""

    def __init__(self):
        self.obs_slicing: List[np.ndarray] = []
        self.obs_sleeping: List[np.ndarray] = []
        self.states: List[np.ndarray] = []
        self.actions_slicing: List[int] = []
        self.actions_sleeping: List[int] = []
        self.logprobs_slicing: List[float] = []
        self.logprobs_sleeping: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.obs_slicing.clear()
        self.obs_sleeping.clear()
        self.states.clear()
        self.actions_slicing.clear()
        self.actions_sleeping.clear()
        self.logprobs_slicing.clear()
        self.logprobs_sleeping.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(
        self,
        obs_slicing: np.ndarray,
        obs_sleeping: np.ndarray,
        state: np.ndarray,
        act_slicing: int,
        act_sleeping: int,
        logp_slicing: float,
        logp_sleeping: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs_slicing.append(obs_slicing)
        self.obs_sleeping.append(obs_sleeping)
        self.states.append(state)
        self.actions_slicing.append(act_slicing)
        self.actions_sleeping.append(act_sleeping)
        self.logprobs_slicing.append(logp_slicing)
        self.logprobs_sleeping.append(logp_sleeping)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


class MAPPOTrainer:
    """Coordinates Centralized Training with Decentralized Execution (CTDE) for MAPPO.

    Features:
    - Decentralized actor policies (pi_slicing, pi_sleeping)
    - Centralized value network V(s) observing global state
    - Generalized Advantage Estimation (GAE)
    - PPO clipped surrogate objective with entropy regularization
    """

    def __init__(
        self,
        config: Dict[str, Any],
        obs_dim_slicing: int = 13,
        obs_dim_sleeping: int = 10,
        state_dim: int = 25,
        num_slicing_actions: int = 12,
        num_sleeping_actions: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        self.train_cfg = config.get("training", {})
        self.device = device if device is not None else torch.device("cpu")

        self.gamma = float(self.train_cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.train_cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(self.train_cfg.get("clip_eps", 0.2))
        self.entropy_coef = float(self.train_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 0.5))
        self.mini_epochs = int(self.train_cfg.get("mini_epochs", 5))
        self.batch_size = int(self.train_cfg.get("batch_size", 64))
        hidden_dim = int(self.train_cfg.get("hidden_dim", 128))

        # Instantiate Agents
        self.agent_slicing = DynamicSlicingAgent(
            obs_dim=obs_dim_slicing,
            action_dim=num_slicing_actions,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.agent_sleeping = XAppSleepingAgent(
            obs_dim=obs_dim_sleeping,
            action_dim=num_sleeping_actions,
            hidden_dim=hidden_dim,
            device=device,
        )

        # Centralized Critic
        self.critic = CentralizedCriticNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(device)

        # Optimizers
        lr_actor = float(self.train_cfg.get("lr_actor", 3e-4))
        lr_critic = float(self.train_cfg.get("lr_critic", 1e-3))

        self.optimizer_actor_slicing = optim.Adam(self.agent_slicing.actor.parameters(), lr=lr_actor)
        self.optimizer_actor_sleeping = optim.Adam(self.agent_sleeping.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def get_value(self, state: np.ndarray) -> float:
        """Estimate state value using Centralized Critic."""
        self.critic.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.critic(state_t)
        return float(val.item())

    def compute_gae(
        self,
        last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values[:-1], dtype=torch.float32, device=self.device)
        returns_t = advantages_t + values_t

        # Normalize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std(unbiased=False) if len(advantages_t) > 1 else torch.tensor(1.0, device=self.device)
        advantages_norm = (advantages_t - adv_mean) / (adv_std + 1e-8)

        return advantages_norm, returns_t

    def update(self, last_value: float) -> Dict[str, float]:
        """Perform PPO update over collected rollouts across multiple mini-epochs."""
        if len(self.buffer.rewards) == 0:
            return {"loss_actor_slicing": 0.0, "loss_actor_sleeping": 0.0, "loss_critic": 0.0}

        advantages, returns = self.compute_gae(last_value)

        # Convert buffer to PyTorch tensors
        obs_slicing_t = torch.as_tensor(np.array(self.buffer.obs_slicing), dtype=torch.float32, device=self.device)
        obs_sleeping_t = torch.as_tensor(np.array(self.buffer.obs_sleeping), dtype=torch.float32, device=self.device)
        states_t = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        acts_slicing_t = torch.as_tensor(self.buffer.actions_slicing, dtype=torch.int64, device=self.device)
        acts_sleeping_t = torch.as_tensor(self.buffer.actions_sleeping, dtype=torch.int64, device=self.device)
        old_logp_slicing_t = torch.as_tensor(self.buffer.logprobs_slicing, dtype=torch.float32, device=self.device)
        old_logp_sleeping_t = torch.as_tensor(self.buffer.logprobs_sleeping, dtype=torch.float32, device=self.device)

        total_samples = len(self.buffer.rewards)
        indices = np.arange(total_samples)

        epoch_actor_loss_slicing = 0.0
        epoch_actor_loss_sleeping = 0.0
        epoch_critic_loss = 0.0
        num_updates = 0

        self.agent_slicing.actor.train()
        self.agent_sleeping.actor.train()
        self.critic.train()

        for _ in range(self.mini_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, total_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch_idx = indices[start_idx:end_idx]

                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]
                b_state = states_t[batch_idx]

                # 1. Update Centralized Critic
                values_pred = self.critic(b_state)
                critic_loss = nn.functional.mse_loss(values_pred, b_ret)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # 2. Update Slicing Actor
                b_obs_s = obs_slicing_t[batch_idx]
                b_act_s = acts_slicing_t[batch_idx]
                b_old_logp_s = old_logp_slicing_t[batch_idx]

                new_logp_s, entropy_s = self.agent_slicing.actor.evaluate_action(b_obs_s, b_act_s)
                ratio_s = torch.exp(new_logp_s - b_old_logp_s)
                surr1_s = ratio_s * b_adv
                surr2_s = torch.clamp(ratio_s, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss_s = -torch.min(surr1_s, surr2_s).mean() - self.entropy_coef * entropy_s.mean()

                self.optimizer_actor_slicing.zero_grad()
                actor_loss_s.backward()
                nn.utils.clip_grad_norm_(self.agent_slicing.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor_slicing.step()

                # 3. Update Sleeping Actor
                b_obs_sleep = obs_sleeping_t[batch_idx]
                b_act_sleep = acts_sleeping_t[batch_idx]
                b_old_logp_sleep = old_logp_sleeping_t[batch_idx]

                new_logp_sleep, entropy_sleep = self.agent_sleeping.actor.evaluate_action(b_obs_sleep, b_act_sleep)
                ratio_sleep = torch.exp(new_logp_sleep - b_old_logp_sleep)
                surr1_sleep = ratio_sleep * b_adv
                surr2_sleep = torch.clamp(ratio_sleep, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss_sleep = -torch.min(surr1_sleep, surr2_sleep).mean() - self.entropy_coef * entropy_sleep.mean()

                self.optimizer_actor_sleeping.zero_grad()
                actor_loss_sleep.backward()
                nn.utils.clip_grad_norm_(self.agent_sleeping.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor_sleeping.step()

                epoch_actor_loss_slicing += actor_loss_s.item()
                epoch_actor_loss_sleeping += actor_loss_sleep.item()
                epoch_critic_loss += critic_loss.item()
                num_updates += 1

        self.buffer.clear()

        divisor = max(num_updates, 1)
        return {
            "loss_actor_slicing": float(np.round(epoch_actor_loss_slicing / divisor, 4)),
            "loss_actor_sleeping": float(np.round(epoch_actor_loss_sleeping / divisor, 4)),
            "loss_critic": float(np.round(epoch_critic_loss / divisor, 4)),
        }
