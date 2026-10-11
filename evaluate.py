"""Evaluation script for trained MAPPO O-RAN slicing and sleeping models."""

import argparse
from pathlib import Path
from typing import Any, Dict
import torch

from agents.slicing_agent import DynamicSlicingAgent
from agents.sleeping_agent import XAppSleepingAgent
from environment.oran_env import ORanSlicingSleepingEnv
from metrics.evaluator import PolicyEvaluator
from models.networks import load_checkpoint
from utils.config import load_config
from utils.seeding import seed_everything


def evaluate_checkpoint(config: Dict[str, Any], checkpoint_path: str, episodes: int = 10) -> Dict[str, Any]:
    """Evaluate a trained MAPPO checkpoint on deterministic test episodes."""
    seed_everything(config.get("project", {}).get("random_seed", 42))
    device = torch.device("cpu")

    env = ORanSlicingSleepingEnv(config)
    evaluator = PolicyEvaluator(env, config)

    # Initialize agents
    agent_slicing = DynamicSlicingAgent(
        obs_dim=env.obs_dim_slicing,
        action_dim=env.num_slicing_actions,
        device=device,
    )
    agent_sleeping = XAppSleepingAgent(
        obs_dim=env.obs_dim_sleeping,
        action_dim=env.num_sleeping_actions,
        device=device,
    )

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        print(f"[Warning] Checkpoint '{checkpoint_path}' does not exist. Evaluation has not yet been performed.")
        print("[Notice] Please run 'python train.py' first to train the agents, or specify a valid checkpoint.")
        return {}

    checkpoint = load_checkpoint(str(ckpt_file), device=device)
    agent_slicing.load_state_dict(checkpoint["actor_slicing"])
    agent_sleeping.load_state_dict(checkpoint["actor_sleeping"])
    print(f"[Evaluation] Successfully loaded checkpoint from {checkpoint_path}")

    def mappo_policy(obs_dict: Dict[str, Any]) -> Dict[str, int]:
        act_s, _ = agent_slicing.select_action(obs_dict["slicing"], deterministic=True)
        act_sleep, _ = agent_sleeping.select_action(obs_dict["sleeping"], deterministic=True)
        return {"slicing": act_s, "sleeping": act_sleep}

    eval_seeds = config.get("baselines", {}).get("eval_seeds", [42, 101, 2024, 777, 999])
    if len(eval_seeds) < episodes:
        eval_seeds.extend([5000 + i for i in range(episodes - len(eval_seeds))])
    eval_seeds = eval_seeds[:episodes]

    results = evaluator.evaluate_policy(
        policy_fn=mappo_policy,
        episodes=episodes,
        seeds=eval_seeds,
        policy_name="Proposed_MAPPO",
    )

    print("\n" + "=" * 65)
    print("           MAPPO EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Episodes Evaluated      : {results['episodes_evaluated']}")
    print(f"Mean Total Energy       : {results['mean_energy_joules']:.2f} J (std: {results['std_energy_joules']:.2f})")
    print(f"Mean Cumulative Reward  : {results['mean_cumulative_reward']:.2f}")
    print(f"Mean SLA Violation Rate : {results['mean_sla_violation_rate'] * 100.0:.2f}%")
    print(f"Mean URLLC Latency      : {results['mean_urllc_latency_ms']:.2f} ms")
    print(f"99th % URLLC Latency    : {results['p99_urllc_latency_ms']:.2f} ms (Target <= 5.0 ms)")
    print(f"Mean eMBB Throughput    : {results['mean_embb_throughput_mbps']:.2f} Mbps")
    print(f"Mean PRB Utilization    : {results['mean_prb_utilization'] * 100.0:.2f}%")
    print(f"Active / Deep Sleep %   : {results['mean_active_ratio'] * 100.0:.1f}% / {results['mean_deep_sleep_ratio'] * 100.0:.1f}%")
    print(f"Mean Sleep Transitions  : {results['mean_transitions']:.1f}")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained MAPPO Checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mappo_final.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_checkpoint(cfg, args.checkpoint, episodes=args.episodes)
