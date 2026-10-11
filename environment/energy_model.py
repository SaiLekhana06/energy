"""O-RAN Base Station (gNB) and Near-RT RIC software xApp energy consumption model."""

from typing import Any, Dict, Tuple
import numpy as np


class ORanEnergyModel:
    """Calculates instantaneous power (Watts) and step energy consumption (Joules)

    for gNB radio cells and Near-RT RIC containerized microservice xApps.

    States:
        0: Active (Full performance, max power)
        1: Light Sleep (DTX / fast wake-up, reduced power)
        2: Deep Sleep (Standby / suspended, minimum power, wake-up penalty)
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("energy", {})
        self.p_base_watts = float(cfg.get("p_base_watts", 180.0))
        self.p_rf_active_watts = float(cfg.get("p_rf_active_watts", 120.0))
        self.p_rf_light_sleep_watts = float(cfg.get("p_rf_light_sleep_watts", 40.0))
        self.p_rf_deep_sleep_watts = float(cfg.get("p_rf_deep_sleep_watts", 8.0))

        self.p_xapp_active_watts = float(cfg.get("p_xapp_active_watts", 45.0))
        self.p_xapp_light_sleep_watts = float(cfg.get("p_xapp_light_sleep_watts", 15.0))
        self.p_xapp_deep_sleep_watts = float(cfg.get("p_xapp_deep_sleep_watts", 3.0))

        self.transition_wake_joules = float(cfg.get("transition_energy_wake_joules", 15.0))
        self.time_step_s = float(config.get("simulation", {}).get("time_step_duration_ms", 100)) / 1000.0

    def compute_energy(
        self,
        current_state: int,
        previous_state: int,
        num_cells: int = 1,
    ) -> Tuple[float, float, bool]:
        """Compute power, energy, and transition status for the current step.

        Args:
            current_state: 0=Active, 1=Light Sleep, 2=Deep Sleep.
            previous_state: 0=Active, 1=Light Sleep, 2=Deep Sleep from previous step.
            num_cells: Number of radio cells controlled.

        Returns:
            Tuple of:
            - power_watts: Instantaneous power in Watts.
            - energy_joules: Energy consumed during this time step in Joules.
            - is_waking_up: Boolean indicating wake-up transition from deep sleep.
        """
        # 1. Base station base power (always consumed for essential control channels)
        power_watts = self.p_base_watts

        # 2. RF power amplifier sub-system
        if current_state == 0:
            rf_power = self.p_rf_active_watts * num_cells
            xapp_power = self.p_xapp_active_watts
        elif current_state == 1:
            rf_power = self.p_rf_light_sleep_watts * num_cells
            xapp_power = self.p_xapp_light_sleep_watts
        else:  # current_state == 2
            rf_power = self.p_rf_deep_sleep_watts * num_cells
            xapp_power = self.p_xapp_deep_sleep_watts

        power_watts += rf_power + xapp_power

        # Base energy for this duration: E = P * dt
        energy_joules = power_watts * self.time_step_s

        # 3. Transition penalty: waking up from deep sleep (state 2 -> state 0 or 1)
        is_waking_up = (previous_state == 2 and current_state != 2)
        if is_waking_up:
            energy_joules += self.transition_wake_joules

        return float(np.round(power_watts, 2)), float(np.round(energy_joules, 2)), is_waking_up
