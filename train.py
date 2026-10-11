"""Training pipeline for Energy-Aware Multi-Agent DRL in 5G O-RAN."""

import argparse
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

from agents.marl_framework import MAPPOTrainer
from environment.oran_env import ORanSlicingSleepingEnv
from models.networks import save_checkpoint
from utils.config import load_config
from utils.logger import ExperimentLogger
from utils.seeding import seed_everything


def train_mappo(config: Dict[str, Any], episodes_override: int = 0) -> str:
    """Run MAPPO centralized training with decentralized execution.

    Args:
        config: Master configuration dictionary.
        episodes_override: If > 0, overrides episodes from config.

    Returns:
        Path to the saved final checkpoint.
    """
    seed = int(config.get("project", {}).get("random_seed", 42))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device} | Seed: {seed}")

    env = ORanSlicingSleepingEnv(config, seed=seed)

    trainer = MAPPOTrainer(
        config=config,
        obs_dim_slicing=env.obs_dim_slicing,
        obs_dim_sleeping=env.obs_dim_sleeping,
        state_dim=env.state_dim,
        num_slicing_actions=env.num_slicing_actions,
        num_sleeping_actions=env.num_sleeping_actions,
        device=device,
    )

    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(log_dir=str(results_dir), experiment_name="mappo_training")

    train_cfg = config.get("training", {})
    total_episodes = episodes_override if episodes_override > 0 else int(train_cfg.get("episodes", 250))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 50))
    checkpoints_dir = Path(config.get("paths", {}).get("checkpoints_dir", "./checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Training] Starting MAPPO training for {total_episodes} episodes...")

    for ep in range(1, total_episodes + 1):
        obs_dict, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_energy = 0.0
        sla_violations = []
        urllc_latencies = []
        prb_utils = []

        done = False
        while not done:
            obs_slicing = obs_dict["slicing"]
            obs_sleeping = obs_dict["sleeping"]
            state = obs_dict["state"]

            # Decentralized action selection
            act_slicing, logp_slicing = trainer.agent_slicing.select_action(obs_slicing, deterministic=False)
            act_sleeping, logp_sleeping = trainer.agent_sleeping.select_action(obs_sleeping, deterministic=False)

            # Centralized Critic state-value estimation
            value = trainer.get_value(state)

            actions = {"slicing": act_slicing, "sleeping": act_sleeping}
            next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Store step in rollout buffer
            trainer.buffer.add(
                obs_slicing=obs_slicing,
                obs_sleeping=obs_sleeping,
                state=state,
                act_slicing=act_slicing,
                act_sleeping=act_sleeping,
                logp_slicing=logp_slicing,
                logp_sleeping=logp_sleeping,
                reward=rewards["global"],
                value=value,
                done=done,
            )

            obs_dict = next_obs_dict
            ep_reward += rewards["global"]
            ep_energy += info["energy_joules"]
            sla_violations.append(info["sla_violation_rate"])
            urllc_latencies.append(info["latency_urllc"])
            prb_utils.append(info["prb_utilization"])

        # Estimate final next state value for GAE bootstrap
        last_val = 0.0 if terminated else trainer.get_value(obs_dict["state"])
        loss_dict = trainer.update(last_val)

        # Log metrics
        ep_metrics = {
            "episode": ep,
            "reward": float(np.round(ep_reward, 3)),
            "energy_joules": float(np.round(ep_energy, 2)),
            "sla_violation_rate": float(np.round(np.mean(sla_violations), 4)),
            "urllc_latency_ms": float(np.round(np.mean(urllc_latencies), 3)),
            "prb_utilization": float(np.round(np.mean(prb_utils), 4)),
            **loss_dict,
        }
        logger.log_step(ep_metrics)

        if ep % 10 == 0 or ep == 1:
            logger.print_progress(ep, total_episodes, ep_metrics)

        # Checkpointing
        if ep % checkpoint_interval == 0 or ep == total_episodes:
            ckpt_path = checkpoints_dir / f"mappo_ep{ep}.pt"
            save_checkpoint(
                {
                    "episode": ep,
                    "actor_slicing": trainer.agent_slicing.get_state_dict(),
                    "actor_sleeping": trainer.agent_sleeping.get_state_dict(),
                    "critic": trainer.critic.state_dict(),
                    "config": config,
                },
                str(ckpt_path),
            )

    # Save final model
    final_path = checkpoints_dir / "mappo_final.pt"
    save_checkpoint(
        {
            "episode": total_episodes,
            "actor_slicing": trainer.agent_slicing.get_state_dict(),
            "actor_sleeping": trainer.agent_sleeping.get_state_dict(),
            "critic": trainer.critic.state_dict(),
            "config": config,
        },
        str(final_path),
    )

    csv_file = logger.save_csv("mappo_training_log.csv")
    print(f"[Training Complete] Checkpoint saved: {final_path}")
    print(f"[Training Complete] Logs saved: {csv_file}")
    return str(final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO for O-RAN Slicing & Sleeping")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=0, help="Override number of training episodes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_mappo(cfg, episodes_override=args.episodes)
