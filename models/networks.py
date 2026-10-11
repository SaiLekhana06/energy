"""PyTorch neural network architectures for MAPPO and Single-Agent baseline."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_weights(m: nn.Module) -> None:
    """Orthogonal initialization for linear layers with standard gains."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class DecentralizedActorNetwork(nn.Module):
    """Decentralized policy network for an individual agent in CTDE.

    Maps private local observation to a categorical action distribution.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> Categorical:
        """Compute action distribution from local observation."""
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log probability.

        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim].
            deterministic: If True, select argmax action instead of sampling.

        Returns:
            action: Action tensor.
            log_prob: Log probability of the chosen action.
        """
        dist = self.forward(obs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for a given action batch."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class CentralizedCriticNetwork(nn.Module):
    """Centralized value network for MAPPO.

    Conditions on the joint global state during centralized training to estimate V(s).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state-value scalar V(s)."""
        return self.net(state).squeeze(-1)


class SingleAgentActorCriticNetwork(nn.Module):
    """Single-Agent baseline actor-critic network.

    Selects joint actions (slicing_action, sleeping_action) without agent decomposition.
    """

    def __init__(
        self,
        state_dim: int,
        slicing_action_dim: int = 12,
        sleeping_action_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.slicing_head = nn.Linear(hidden_dim, slicing_action_dim)
        self.sleeping_head = nn.Linear(hidden_dim, sleeping_action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.apply(_init_weights)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[Categorical, Categorical, torch.Tensor]:
        feat = self.backbone(state)
        slicing_dist = Categorical(logits=self.slicing_head(feat))
        sleeping_dist = Categorical(logits=self.sleeping_head(feat))
        value = self.critic_head(feat).squeeze(-1)
        return slicing_dist, sleeping_dist, value


def save_checkpoint(
    state_dict_payload: Dict[str, Any],
    filepath: str,
) -> Path:
    """Save model checkpoint to disk.

    Args:
        state_dict_payload: Dictionary containing model parameters, optimizers, and epoch metadata.
        filepath: Target file destination.

    Returns:
        Path object to saved checkpoint.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict_payload, path)
    return path


def load_checkpoint(
    filepath: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint from disk.

    Args:
        filepath: Path to saved checkpoint file.
        device: Target torch device.

    Returns:
        Loaded state dictionary payload.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if device is None:
        device = torch.device("cpu")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
