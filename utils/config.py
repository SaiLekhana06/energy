"""Configuration loading, validation, and defaults management."""

from pathlib import Path
from typing import Any, Dict


def get_fallback_config() -> Dict[str, Any]:
    """Return fallback configuration dictionary if config.yaml cannot be parsed."""
    return {
        "project": {
            "name": "oran_energy_slicing_marl",
            "title": "Energy-Aware Multi-Agent DRL for Dynamic Network Slicing and xApp Sleeping in O-RAN",
            "version": "1.0.0",
            "random_seed": 42,
        },
        "simulation": {
            "time_step_duration_ms": 100,
            "episode_length_steps": 100,
            "total_prbs": 100,
            "min_prb_per_slice": 5,
            "num_cells": 3,
            "max_ue_per_slice": {"embb": 30, "urllc": 20, "mmtc": 150},
        },
        "traffic": {
            "embb": {"mean_rate_mbps": 65.0, "burst_probability": 0.15, "burst_multiplier": 1.8, "diurnal_phase_offset": 0.0},
            "urllc": {"mean_rate_mbps": 12.0, "burst_probability": 0.25, "burst_multiplier": 2.5, "arrival_mode": "poisson"},
            "mmtc": {"mean_rate_mbps": 8.0, "device_active_ratio": 0.4, "burst_probability": 0.05, "burst_multiplier": 1.5},
        },
        "sla": {
            "embb": {"min_throughput_mbps": 45.0, "max_latency_ms": 25.0},
            "urllc": {"min_throughput_mbps": 10.0, "max_latency_ms": 5.0, "target_reliability": 0.99999},
            "mmtc": {"min_throughput_mbps": 6.0, "max_latency_ms": 80.0},
        },
        "energy": {
            "p_base_watts": 180.0,
            "p_rf_active_watts": 120.0,
            "p_rf_light_sleep_watts": 40.0,
            "p_rf_deep_sleep_watts": 8.0,
            "p_xapp_active_watts": 15.0,
            "p_xapp_light_sleep_watts": 5.0,
            "p_xapp_deep_sleep_watts": 1.0,
            "wake_transition_penalty_joules": {"light_to_active": 6.0, "deep_to_active": 20.0},
        },
        "training": {
            "episodes": 250,
            "steps_per_episode": 100,
            "learning_rate_actor": 0.0003,
            "learning_rate_critic": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coeff": 0.01,
            "ppo_epochs": 4,
            "batch_size": 32,
            "checkpoint_interval": 50,
        },
        "baselines": {
            "eval_seeds": [42, 101, 2024, 777, 999],
            "static_prbs": {"embb": 50, "urllc": 30, "mmtc": 20},
            "greedy_thresholds": {"light_sleep_load_mbps": 70.0, "deep_sleep_load_mbps": 35.0},
        },
        "paths": {
            "checkpoints_dir": "./checkpoints",
            "results_dir": "./results",
            "plots_dir": "./results/plots",
        },
    }


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file with graceful fallback."""
    path = Path(config_path)
    if not path.is_file():
        return get_fallback_config()

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            validate_config(cfg)
            return cfg
    except ImportError:
        pass
    except Exception:
        pass

    return get_fallback_config()


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate critical parameters in the configuration dictionary."""
    if not isinstance(cfg, dict):
        return

    sim = cfg.get("simulation")
    if sim and isinstance(sim, dict):
        total = sim.get("total_prbs", 100)
        min_p = sim.get("min_prb_per_slice", 5)
        assert total > 0, "total_prbs must be positive"
        assert min_p * 3 <= total, "Sum of min_prb_per_slice cannot exceed total_prbs"
