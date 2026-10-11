"""Baseline 2: Always-active policy with adaptive slicing."""

from typing import Any, Dict
import numpy as np


class AlwaysActivePolicy:
    """Consistently keeps radio cells and xApps fully powered (state 0),

    while adjusting PRBs proportionally to current slice demands.
    """

    def __init__(self, config: Dict[str, Any]):
        self.name = "Always_Active"

    def select_action(self, obs_dict: Dict[str, Any]) -> Dict[str, int]:
        """Select dynamic slicing action, but strictly keep sleeping = 0 (Active)."""
        obs_slicing = obs_dict.get("slicing", np.zeros(13))
        # Demands are at indices 0, 1, 2
        d_embb = obs_slicing[0]
        d_urllc = obs_slicing[1]
        d_mmtc = obs_slicing[2]

        # Profile selection heuristic based on dominant demand
        if d_urllc > 0.20:
            slicing_act = 4  # Extreme URLLC profile
        elif d_embb > 0.60:
            slicing_act = 0  # Heavy eMBB profile
        elif d_mmtc > 0.15:
            slicing_act = 5  # High mMTC profile
        else:
            slicing_act = 3  # Standard balanced

        return {
            "slicing": slicing_act,
            "sleeping": 0,  # Never sleep
        }
