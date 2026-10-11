"""Comparative evaluation of all baseline policies and proposed MARL."""

import argparse
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import torch

from agents.slicing_agent import DynamicSlicingAgent
from agents.sleeping_agent import XAppSleepingAgent
from baselines.always_active import AlwaysActivePolicy
from baselines.greedy_energy import GreedyThresholdPolicy
from baselines.single_agent_drl import SingleAgentDRLPolicy
from baselines.static_slicing import StaticSlicingPolicy
from environment.oran_env import ORanSlicingSleepingEnv
from metrics.evaluator import PolicyEvaluator, calculate_energy_saving
from models.networks import load_checkpoint
from utils.config import load_config
from utils.seeding import seed_everything


def run_baseline_comparison(
    config: Dict[str, Any],
    checkpoint_path: str = "checkpoints/mappo_final.pt",
    episodes: int = 10,
) -> pd.DataFrame:
    """Execute evaluation across all five benchmark policies."""
    seed_val = config.get("project", {}).get("random_seed", 42)
    seed_everything(seed_val)

    env = ORanSlicingSleepingEnv(config)
    evaluator = PolicyEvaluator(env, config)

    eval_seeds = config.get("baselines", {}).get("eval_seeds", [42, 101, 2024, 777, 999])
    if len(eval_seeds) < episodes:
        eval_seeds.extend([3000 + i for i in range(episodes - len(eval_seeds))])
    eval_seeds = eval_seeds[:episodes]

    summaries: List[Dict[str, Any]] = []
    all_ep_dfs: List[pd.DataFrame] = []

    # -------------------------------------------------------------
    # 1. Baseline 1: Static Slicing
    # -------------------------------------------------------------
    p_static = StaticSlicingPolicy(config)
    res_static = evaluator.evaluate_policy(
        policy_fn=p_static.select_action,
        episodes=episodes,
        seeds=eval_seeds,
        policy_name=p_static.name,
    )
    summaries.append(res_static)
    all_ep_dfs.append(res_static["raw_dataframe"])

    # -------------------------------------------------------------
    # 2. Baseline 2: Always Active (Primary Energy Benchmark)
    # -------------------------------------------------------------
    p_active = AlwaysActivePolicy(config)
    res_active = evaluator.evaluate_policy(
        policy_fn=p_active.select_action,
        episodes=episodes,
        seeds=eval_seeds,
        policy_name=p_active.name,
    )
    summaries.append(res_active)
    all_ep_dfs.append(res_active["raw_dataframe"])

    # -------------------------------------------------------------
    # 3. Baseline 3: Greedy / Threshold Heuristic
    # -------------------------------------------------------------
    p_greedy = GreedyThresholdPolicy(config)
    res_greedy = evaluator.evaluate_policy(
        policy_fn=p_greedy.select_action,
        episodes=episodes,
        seeds=eval_seeds,
        policy_name=p_greedy.name,
    )
    summaries.append(res_greedy)
    all_ep_dfs.append(res_greedy["raw_dataframe"])

    # -------------------------------------------------------------
    # 4. Baseline 4: Single-Agent DRL
    # -------------------------------------------------------------
    p_single = SingleAgentDRLPolicy(config)
    res_single = evaluator.evaluate_policy(
        policy_fn=p_single.select_action,
        episodes=episodes,
        seeds=eval_seeds,
        policy_name=p_single.name,
    )
    summaries.append(res_single)
    all_ep_dfs.append(res_single["raw_dataframe"])

    # -------------------------------------------------------------
    # 5. Proposed Method: MARL (MAPPO)
    # -------------------------------------------------------------
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.is_file():
        device = torch.device("cpu")
        ag_slicing = DynamicSlicingAgent(obs_dim=env.obs_dim_slicing, action_dim=env.num_slicing_actions, device=device)
        ag_sleeping = XAppSleepingAgent(obs_dim=env.obs_dim_sleeping, action_dim=env.num_sleeping_actions, device=device)
        ckpt = load_checkpoint(str(ckpt_file), device=device)
        ag_slicing.load_state_dict(ckpt["actor_slicing"])
        ag_sleeping.load_state_dict(ckpt["actor_sleeping"])

        def mappo_fn(obs_dict: Dict[str, Any]) -> Dict[str, int]:
            s_act, _ = ag_slicing.select_action(obs_dict["slicing"], deterministic=True)
            sleep_act, _ = ag_sleeping.select_action(obs_dict["sleeping"], deterministic=True)
            return {"slicing": s_act, "sleeping": sleep_act}

        res_mappo = evaluator.evaluate_policy(
            policy_fn=mappo_fn,
            episodes=episodes,
            seeds=eval_seeds,
            policy_name="Proposed_MAPPO",
        )
    else:
        print(f"[Notice] Checkpoint '{checkpoint_path}' not found. Using untrained cooperative agents for structure verification.")
        ag_slicing = DynamicSlicingAgent(obs_dim=env.obs_dim_slicing, action_dim=env.num_slicing_actions)
        ag_sleeping = XAppSleepingAgent(obs_dim=env.obs_dim_sleeping, action_dim=env.num_sleeping_actions)

        def untrained_fn(obs_dict: Dict[str, Any]) -> Dict[str, int]:
            s_act, _ = ag_slicing.select_action(obs_dict["slicing"], deterministic=True)
            sleep_act, _ = ag_sleeping.select_action(obs_dict["sleeping"], deterministic=True)
            return {"slicing": s_act, "sleeping": sleep_act}

        res_mappo = evaluator.evaluate_policy(
            policy_fn=untrained_fn,
            episodes=episodes,
            seeds=eval_seeds,
            policy_name="Proposed_MAPPO (Untrained)",
        )

    summaries.append(res_mappo)
    all_ep_dfs.append(res_mappo["raw_dataframe"])

    # -------------------------------------------------------------
    # Compute Relative Energy Savings (%) vs Always-Active
    # -------------------------------------------------------------
    always_active_energy = res_active["mean_energy_joules"]

    rows = []
    for s in summaries:
        saving = calculate_energy_saving(always_active_energy, s["mean_energy_joules"])
        rows.append({
            "Policy": s["policy"],
            "Energy (J)": s["mean_energy_joules"],
            "Energy Std": s["std_energy_joules"],
            "Energy Saving (%)": saving,
            "SLA Viol Rate (%)": s["mean_sla_violation_rate"] * 100.0,
            "URLLC Latency (ms)": s["mean_urllc_latency_ms"],
            "URLLC P99 Lat (ms)": s["p99_urllc_latency_ms"],
            "eMBB Thr (Mbps)": s["mean_embb_throughput_mbps"],
            "PRB Util (%)": s["mean_prb_utilization"] * 100.0,
            "Cumul Reward": s["mean_cumulative_reward"],
            "Deep Sleep (%)": s["mean_deep_sleep_ratio"] * 100.0,
            "Transitions": s["mean_transitions"],
        })

    summary_df = pd.DataFrame(rows)

    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "baseline_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    full_ep_df = pd.concat(all_ep_dfs, ignore_index=True)
    full_ep_path = results_dir / "baseline_comparison_episodes.csv"
    full_ep_df.to_csv(full_ep_path, index=False)

    print("\n" + "=" * 90)
    print("                    BASELINE COMPARISON TABLE")
    print("           (Energy Saving % calculated relative to 'Always_Active')")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print(f"\n[Saved] Summary saved to: {summary_path}")
    print(f"[Saved] Per-episode traces saved to: {full_ep_path}\n")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Baseline Policies and MARL")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mappo_final.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_baseline_comparison(cfg, checkpoint_path=args.checkpoint, episodes=args.episodes)
