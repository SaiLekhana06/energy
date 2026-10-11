"""Baseline 3: Greedy / Threshold-based energy saving policy."""

from typing import Any, Dict
import numpy as np


class GreedyThresholdPolicy:
    """Heuristic rule-based policy.

    Decides sleep state based on aggregate network load threshold:
    - If total demand < low_threshold: Deep Sleep (2)
    - If low_threshold <= total demand < high_threshold: Light Sleep (1)
    - If total demand >= high_threshold or URLLC spike: Active (0)
    """

    def __init__(self, config: Dict[str, Any]):
        self.name = "Greedy_Threshold"
        g_cfg = config.get("baselines", {}).get("greedy_energy", {})
        self.low_threshold_mbps = float(g_cfg.get("traffic_threshold_mbps", 35.0))
        self.high_threshold_mbps = float(g_cfg.get("high_load_threshold_mbps", 70.0))

    def select_action(self, obs_dict: Dict[str, Any]) -> Dict[str, int]:
        obs_slicing = obs_dict.get("slicing", np.zeros(13))
        # Denormalize demands (scaled by 100 in environment)
        d_embb = obs_slicing[0] * 100.0
        d_urllc = obs_slicing[1] * 100.0
        d_mmtc = obs_slicing[2] * 100.0
        total_demand = d_embb + d_urllc + d_mmtc

        # Sleep state decision based on thresholds
        # If URLLC is high, avoid sleep to protect strict SLA
        if d_urllc > 15.0 or total_demand >= self.high_threshold_mbps:
            sleeping_act = 0  # Active
        elif total_demand < self.low_threshold_mbps:
            sleeping_act = 2  # Deep Sleep
        else:
            sleeping_act = 1  # Light Sleep

        # Slicing action
        if d_urllc > 15.0:
            slicing_act = 4  # URLLC heavy
        elif d_embb > 55.0:
            slicing_act = 0  # eMBB heavy
        elif d_mmtc > 12.0:
            slicing_act = 5  # mMTC heavy
        else:
            slicing_act = 3  # Balanced

        return {
            "slicing": slicing_act,
            "sleeping": sleeping_act,
        }
