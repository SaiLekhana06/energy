"""Agent 1: Dynamic Network Slicing Agent for PRB Allocation."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from models.networks import DecentralizedActorNetwork


class DynamicSlicingAgent:
    """Agent 1 in O-RAN Near-RT RIC.

    Observes slice traffic demands, latencies, and SLA metrics to dynamically allocate
    Physical Resource Blocks (PRBs) among eMBB, URLLC, and mMTC slices.
    """

    def __init__(
        self,
        obs_dim: int = 13,
        action_dim: int = 12,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cpu")

        self.actor = DecentralizedActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Select action given local observation.

        Args:
            obs: Observation numpy array [obs_dim].
            deterministic: If True, greedy selection.

        Returns:
            action_idx: Integer action index.
            log_prob: Float log probability.
        """
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t, log_prob_t = self.actor.get_action(obs_t, deterministic=deterministic)
        return int(action_t.item()), float(log_prob_t.item())

    def get_state_dict(self) -> Dict[str, Any]:
        """Return actor model state dictionary."""
        return self.actor.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load actor model state dictionary."""
        self.actor.load_state_dict(state_dict)
