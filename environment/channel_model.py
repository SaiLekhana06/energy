"""5G NR wireless channel, SINR, spectral efficiency, and latency models."""

import math
from typing import Any, Dict
import numpy as np


class WirelessChannelModel:
    """Computes SINR, spectral efficiency (Shannon / 3GPP), throughput capacity,

    and slice queueing delays based on PRB allocation and xApp operating states.
    """

    def __init__(self, config: Dict[str, Any], rng: np.random.Generator):
        self.cfg = config.get("channel", {})
        self.rng = rng

        self.bw_mhz = float(self.cfg.get("bandwidth_mhz", 20.0))
        self.carrier_freq_ghz = float(self.cfg.get("carrier_frequency_ghz", 3.5))
        self.shadowing_std_db = float(self.cfg.get("shadowing_std_db", 4.0))

        # 30 kHz Subcarrier Spacing (SCS): 1 PRB = 12 * 30 kHz = 360 kHz
        self.prb_bandwidth_hz = 360e3

        # Base nominal spectral efficiency (bps/Hz) under good channel conditions
        self.nominal_sinr_db = 16.0

    def calculate_spectral_efficiency(self, xapp_state: int) -> float:
        """Calculate effective spectral efficiency (bps/Hz).

        If beamforming and modulation optimization xApps are asleep, effective
        beamforming gain drops slightly (e.g., -2.5 dB in Light Sleep, -4.0 dB in Deep Sleep).

        Args:
            xapp_state: 0=Active, 1=Light Sleep, 2=Deep Sleep.

        Returns:
            Spectral efficiency in bps/Hz.
        """
        gain_loss_db = 0.0
        if xapp_state == 1:
            gain_loss_db = 2.0
        elif xapp_state == 2:
            gain_loss_db = 4.0

        shadowing = self.rng.normal(0.0, self.shadowing_std_db)
        effective_sinr_db = self.nominal_sinr_db - gain_loss_db + shadowing
        # Bound SINR between -5 dB and 30 dB
        effective_sinr_db = np.clip(effective_sinr_db, -5.0, 30.0)

        sinr_linear = 10.0 ** (effective_sinr_db / 10.0)
        # Shannon capacity with 3GPP practical implementation loss (0.65 efficiency factor)
        se_bps_per_hz = 0.65 * math.log2(1.0 + sinr_linear)
        return float(np.clip(se_bps_per_hz, 0.5, 6.0))

    def compute_slice_metrics(
        self,
        slice_name: str,
        prbs_allocated: int,
        traffic_demand_mbps: float,
        xapp_state: int,
        is_waking_up: bool = False,
    ) -> Dict[str, float]:
        """Compute achieved throughput, average latency, and SLA violation.

        Args:
            slice_name: 'embb', 'urllc', or 'mmtc'.
            prbs_allocated: Integer number of PRBs assigned to this slice.
            traffic_demand_mbps: Incoming instantaneous load in Mbps.
            xapp_state: 0=Active, 1=Light Sleep, 2=Deep Sleep.
            is_waking_up: True if waking from deep sleep this step (incurs transient delay).

        Returns:
            Dictionary with throughput_mbps, latency_ms, prb_utilization, sla_violated.
        """
        se = self.calculate_spectral_efficiency(xapp_state)
        # Max capacity in Mbps = PRBs * PRB_BW * SE / 1e6
        capacity_mbps = (prbs_allocated * self.prb_bandwidth_hz * se) / 1e6

        # Achieved throughput cannot exceed demand or physical slice capacity
        achieved_throughput_mbps = min(traffic_demand_mbps, capacity_mbps)

        # PRB capacity utilization ratio
        prb_utilization = (achieved_throughput_mbps / (capacity_mbps + 1e-6))
        prb_utilization = float(np.clip(prb_utilization, 0.0, 1.0))

        # Latency calculation based on M/M/1 queuing model + transmission delay
        # D = D_trans + D_queue
        # When traffic demand approaches or exceeds capacity, queue delay escalates.
        load_factor = traffic_demand_mbps / (capacity_mbps + 1e-6)

        if slice_name == "urllc":
            # Baseline transmission & processing delay for URLLC is very small (1.0 ms)
            base_delay_ms = 1.2
            if load_factor < 0.85:
                queue_delay_ms = 0.5 / (1.0 - load_factor + 0.1)
            else:
                # Congestion penalty
                queue_delay_ms = 3.5 + 8.0 * (load_factor - 0.85)

            # Extra latency if waking up from deep sleep
            if is_waking_up:
                queue_delay_ms += 4.0

            latency_ms = base_delay_ms + queue_delay_ms
            # SLA violation: URLLC latency must be <= 5.0 ms and capacity must meet demand
            sla_violated = 1.0 if (latency_ms > 5.0 or achieved_throughput_mbps < 0.95 * traffic_demand_mbps) else 0.0

        elif slice_name == "embb":
            base_delay_ms = 8.0
            if load_factor < 0.90:
                queue_delay_ms = 3.0 / (1.0 - load_factor + 0.15)
            else:
                queue_delay_ms = 15.0 + 15.0 * (load_factor - 0.90)

            if is_waking_up:
                queue_delay_ms += 6.0

            latency_ms = base_delay_ms + queue_delay_ms
            # SLA violation: Throughput satisfaction ratio >= 90% and latency <= 25.0 ms
            sla_violated = 1.0 if (achieved_throughput_mbps < 0.90 * traffic_demand_mbps or latency_ms > 25.0) else 0.0

        else:  # mmtc
            base_delay_ms = 20.0
            if load_factor < 0.95:
                queue_delay_ms = 5.0 / (1.0 - load_factor + 0.1)
            else:
                queue_delay_ms = 30.0 + 40.0 * (load_factor - 0.95)

            latency_ms = base_delay_ms + queue_delay_ms
            # SLA violation: Throughput satisfaction ratio >= 85% and latency <= 80.0 ms
            sla_violated = 1.0 if (achieved_throughput_mbps < 0.85 * traffic_demand_mbps or latency_ms > 80.0) else 0.0

        return {
            "capacity_mbps": float(np.round(capacity_mbps, 3)),
            "achieved_throughput_mbps": float(np.round(achieved_throughput_mbps, 3)),
            "latency_ms": float(np.round(latency_ms, 3)),
            "prb_utilization": float(np.round(prb_utilization, 4)),
            "sla_violated": float(sla_violated),
        }
