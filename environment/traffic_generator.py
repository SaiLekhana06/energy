"""Synthetic time-varying multi-slice traffic generator for eMBB, URLLC, and mMTC."""

import math
from typing import Any, Dict
import numpy as np


class SyntheticTrafficGenerator:
    """Generates synthetic, time-varying traffic demand for 5G O-RAN slices.

    Slices:
    - eMBB (Enhanced Mobile Broadband): High throughput, continuous streaming & downloads,
      diurnal patterns with burst multipliers.
    - URLLC (Ultra-Reliable Low-Latency Communications): Critical periodic packets with
      Poisson burst arrivals representing tactile/emergency telemetry.
    - mMTC (Massive Machine-Type Communications): Periodic aggregate sensor payload
      from thousands of IoT nodes with active device ratio variation.
    """

    def __init__(self, config: Dict[str, Any], rng: np.random.Generator):
        self.cfg = config.get("traffic", {})
        self.rng = rng

        # eMBB parameters
        embb_cfg = self.cfg.get("embb", {})
        self.embb_mean = float(embb_cfg.get("mean_rate_mbps", 65.0))
        self.embb_burst_prob = float(embb_cfg.get("burst_probability", 0.15))
        self.embb_burst_mult = float(embb_cfg.get("burst_multiplier", 1.8))
        self.embb_phase = float(embb_cfg.get("diurnal_phase_offset", 0.0))

        # URLLC parameters
        urllc_cfg = self.cfg.get("urllc", {})
        self.urllc_mean = float(urllc_cfg.get("mean_rate_mbps", 12.0))
        self.urllc_burst_prob = float(urllc_cfg.get("burst_probability", 0.25))
        self.urllc_burst_mult = float(urllc_cfg.get("burst_multiplier", 2.5))

        # mMTC parameters
        mmtc_cfg = self.cfg.get("mmtc", {})
        self.mmtc_mean = float(mmtc_cfg.get("mean_rate_mbps", 8.0))
        self.mmtc_burst_prob = float(mmtc_cfg.get("burst_probability", 0.05))
        self.mmtc_burst_mult = float(mmtc_cfg.get("burst_multiplier", 1.5))

        self.current_step = 0

    def reset(self, rng: np.random.Generator) -> None:
        """Reset internal generator state and RNG."""
        self.rng = rng
        self.current_step = 0

    def generate(self, step: int) -> Dict[str, float]:
        """Generate traffic demand in Mbps for the current simulation step.

        Args:
            step: Discrete simulation step index.

        Returns:
            Dictionary mapping slice names ('embb', 'urllc', 'mmtc') to demand (Mbps).
        """
        self.current_step = step

        # 1. eMBB: Diurnal wave (period = 100 steps) + Gaussian noise + bursts
        diurnal_factor = 0.7 + 0.3 * math.sin(2 * math.pi * step / 100.0 + self.embb_phase)
        embb_base = self.embb_mean * diurnal_factor
        embb_noise = self.rng.normal(0.0, 0.08 * self.embb_mean)
        embb_rate = max(5.0, embb_base + embb_noise)
        if self.rng.random() < self.embb_burst_prob:
            embb_rate *= self.embb_burst_mult

        # 2. URLLC: Constant low rate baseline + Poisson burst spikes
        urllc_base = self.urllc_mean + self.rng.normal(0.0, 0.05 * self.urllc_mean)
        urllc_rate = max(1.0, urllc_base)
        if self.rng.random() < self.urllc_burst_prob:
            # Burst arrival: Poisson multiplier
            burst_intensity = self.rng.poisson(lam=self.urllc_burst_mult)
            urllc_rate += burst_intensity * 4.0

        # 3. mMTC: Active device fluctuation + periodic sensor reporting batch
        mmtc_fluctuation = 0.85 + 0.3 * self.rng.random()
        mmtc_rate = max(1.0, self.mmtc_mean * mmtc_fluctuation)
        if self.rng.random() < self.mmtc_burst_prob:
            mmtc_rate *= self.mmtc_burst_mult

        return {
            "embb": float(np.round(embb_rate, 3)),
            "urllc": float(np.round(urllc_rate, 3)),
            "mmtc": float(np.round(mmtc_rate, 3)),
        }
