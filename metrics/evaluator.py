"""Evaluation metrics computation for O-RAN slicing and energy performance."""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_energy_saving(baseline_energy: float, proposed_energy: float) -> float:
    """Calculate relative energy savings percentage safely.

    Formula:
        Energy saving (%) = ((Baseline energy - Proposed energy) / Baseline energy) * 100

    Args:
        baseline_energy: Energy consumed by baseline policy (Joules or kWh).
        proposed_energy: Energy consumed by evaluated policy.

    Returns:
        Energy saving percentage (float). Returns 0.0 if baseline is <= 0.
    """
    if baseline_energy <= 1e-9:
        return 0.0
    saving = ((baseline_energy - proposed_energy) / baseline_energy) * 100.0
    return float(np.round(saving, 2))


class PolicyEvaluator:
    """Runs standardized evaluation episodes across policies and computes comprehensive KPIs."""

    def __init__(self, env: Any, config: Dict[str, Any]):
        self.env = env
        self.config = config

    def evaluate_policy(
        self,
        policy_fn: Callable[[Dict[str, Any]], Dict[str, int]],
        episodes: int = 10,
        seeds: Optional[List[int]] = None,
        policy_name: str = "Policy",
    ) -> Dict[str, Any]:
        """Execute evaluation episodes for a given policy action selection function.

        Args:
            policy_fn: Callable mapping observation dictionary to actions {'slicing': int, 'sleeping': int}.
            episodes: Number of evaluation episodes to run.
            seeds: List of random seeds for evaluation reproducibility.
            policy_name: Name label of the policy.

        Returns:
            Dictionary containing aggregated mean metrics and raw per-episode data.
        """
        if seeds is None:
            seeds = [100 + i for i in range(episodes)]
        else:
            episodes = len(seeds)

        episode_records: List[Dict[str, Any]] = []

        for ep_idx in range(episodes):
            seed = seeds[ep_idx]
            obs_dict, _ = self.env.reset(seed=seed)
            done = False

            ep_energy = 0.0
            ep_reward = 0.0
            sla_violations: List[float] = []
            embb_latencies: List[float] = []
            urllc_latencies: List[float] = []
            mmtc_latencies: List[float] = []
            embb_throughputs: List[float] = []
            urllc_throughputs: List[float] = []
            mmtc_throughputs: List[float] = []
            prb_utils: List[float] = []
            states: List[int] = []
            transitions = 0
            prev_state = 0

            while not done:
                action = policy_fn(obs_dict)
                obs_dict, rewards, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                ep_energy += info["energy_joules"]
                ep_reward += rewards["global"]
                sla_violations.append(info["sla_violation_rate"])
                embb_latencies.append(info["latency_embb"])
                urllc_latencies.append(info["latency_urllc"])
                mmtc_latencies.append(info["latency_mmtc"])
                embb_throughputs.append(info["throughput_embb"])
                urllc_throughputs.append(info["throughput_urllc"])
                mmtc_throughputs.append(info["throughput_mmtc"])
                prb_utils.append(info["prb_utilization"])

                cur_state = info["xapp_state"]
                states.append(cur_state)
                if cur_state != prev_state:
                    transitions += 1
                prev_state = cur_state

            total_steps = max(len(states), 1)
            active_ratio = sum(1 for s in states if s == 0) / total_steps
            light_sleep_ratio = sum(1 for s in states if s == 1) / total_steps
            deep_sleep_ratio = sum(1 for s in states if s == 2) / total_steps

            episode_records.append({
                "policy": policy_name,
                "episode": ep_idx + 1,
                "seed": seed,
                "total_energy_joules": ep_energy,
                "cumulative_reward": ep_reward,
                "sla_violation_rate": float(np.mean(sla_violations)),
                "avg_urllc_latency_ms": float(np.mean(urllc_latencies)),
                "p99_urllc_latency_ms": float(np.percentile(urllc_latencies, 99)),
                "avg_embb_latency_ms": float(np.mean(embb_latencies)),
                "avg_mmtc_latency_ms": float(np.mean(mmtc_latencies)),
                "avg_embb_throughput_mbps": float(np.mean(embb_throughputs)),
                "avg_urllc_throughput_mbps": float(np.mean(urllc_throughputs)),
                "avg_mmtc_throughput_mbps": float(np.mean(mmtc_throughputs)),
                "avg_prb_utilization": float(np.mean(prb_utils)),
                "active_ratio": active_ratio,
                "light_sleep_ratio": light_sleep_ratio,
                "deep_sleep_ratio": deep_sleep_ratio,
                "sleep_transitions": transitions,
            })

        df = pd.DataFrame(episode_records)

        summary = {
            "policy": policy_name,
            "episodes_evaluated": episodes,
            "mean_energy_joules": float(df["total_energy_joules"].mean()),
            "std_energy_joules": float(df["total_energy_joules"].std()),
            "mean_cumulative_reward": float(df["cumulative_reward"].mean()),
            "mean_sla_violation_rate": float(df["sla_violation_rate"].mean()),
            "mean_urllc_latency_ms": float(df["avg_urllc_latency_ms"].mean()),
            "p99_urllc_latency_ms": float(df["p99_urllc_latency_ms"].mean()),
            "mean_embb_throughput_mbps": float(df["avg_embb_throughput_mbps"].mean()),
            "mean_prb_utilization": float(df["avg_prb_utilization"].mean()),
            "mean_active_ratio": float(df["active_ratio"].mean()),
            "mean_deep_sleep_ratio": float(df["deep_sleep_ratio"].mean()),
            "mean_transitions": float(df["sleep_transitions"].mean()),
            "raw_dataframe": df,
        }
        return summary
