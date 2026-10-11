"""Baseline 4: Single-Agent Deep Reinforcement Learning (Joint PPO)."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import SingleAgentActorCriticNetwork


class SingleAgentDRLPolicy:
    """Single-Agent baseline that uses unified PPO without agent decomposition.

    Directly maps global state to both slicing and sleeping actions jointly.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        state_dim: int = 25,
        slicing_action_dim: int = 12,
        sleeping_action_dim: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.name = "Single_Agent_DRL"
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

        hidden_dim = int(config.get("training", {}).get("hidden_dim", 128))
        self.model = SingleAgentActorCriticNetwork(
            state_dim=state_dim,
            slicing_action_dim=slicing_action_dim,
            sleeping_action_dim=sleeping_action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        lr = float(config.get("training", {}).get("lr_actor", 3e-4))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(
        self,
        obs_dict: Dict[str, Any],
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Select joint action from global state vector."""
        state = obs_dict.get("state")
        if state is None:
            # Fallback if state not provided
            state = np.zeros(25, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            slicing_dist, sleeping_dist, _ = self.model(state_t)

            if deterministic:
                act_slicing = torch.argmax(slicing_dist.probs, dim=-1).item()
                act_sleeping = torch.argmax(sleeping_dist.probs, dim=-1).item()
            else:
                act_slicing = slicing_dist.sample().item()
                act_sleeping = sleeping_dist.sample().item()

        return {
            "slicing": int(act_slicing),
            "sleeping": int(act_sleeping),
        }

    def train_step(
        self,
        states: torch.Tensor,
        acts_slicing: torch.Tensor,
        acts_sleeping: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_logp_slicing: torch.Tensor,
        old_logp_sleeping: torch.Tensor,
        clip_eps: float = 0.2,
    ) -> float:
        """Perform one training update step."""
        self.model.train()
        slicing_dist, sleeping_dist, values = self.model(states)

        # Log probabilities
        new_logp_s = slicing_dist.log_prob(acts_slicing)
        new_logp_sleep = sleeping_dist.log_prob(acts_sleeping)

        ratio_s = torch.exp(new_logp_s - old_logp_slicing)
        surr1_s = ratio_s * advantages
        surr2_s = torch.clamp(ratio_s, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        loss_s = -torch.min(surr1_s, surr2_s).mean()

        ratio_sleep = torch.exp(new_logp_sleep - old_logp_sleeping)
        surr1_sleep = ratio_sleep * advantages
        surr2_sleep = torch.clamp(ratio_sleep, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        loss_sleep = -torch.min(surr1_sleep, surr2_sleep).mean()

        critic_loss = nn.functional.mse_loss(values, returns)
        total_loss = loss_s + loss_sleep + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return float(total_loss.item())
