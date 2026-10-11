"""Baseline 1: Static PRB allocation with always-active components."""

from typing import Any, Dict


class StaticSlicingPolicy:
    """Fixed PRB partitioning policy.

    Allocates fixed PRB quotas (e.g., 50 eMBB, 30 URLLC, 20 mMTC)
    and maintains all xApps and cells in continuous Active state (no sleep).
    """

    def __init__(self, config: Dict[str, Any]):
        # Default fixed profile: index 3 -> (50, 30, 20)
        self.profile_index = 3
        self.name = "Static_Slicing"

    def select_action(self, obs_dict: Dict[str, Any]) -> Dict[str, int]:
        """Return fixed actions: static slicing profile, active sleep state."""
        return {
            "slicing": self.profile_index,
            "sleeping": 0,  # 0 = Active
        }
