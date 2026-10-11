"""Generates publication-quality Matplotlib figures from experimental results."""

from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import load_config


def plot_all_results(config: Dict[str, Any]) -> None:
    """Generate and save all publication-style comparison plots."""
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "./results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    # -------------------------------------------------------------
    # 1. Training Convergence Curve
    # -------------------------------------------------------------
    train_log_path = results_dir / "mappo_training_log.csv"
    if train_log_path.is_file():
        df_train = pd.read_csv(train_log_path)
        fig, ax1 = plt.subplots(figsize=(8, 5))

        episodes = df_train["episode"]
        rewards = df_train["reward"]
        rolling_rew = rewards.rolling(window=max(len(rewards) // 10, 5), min_periods=1).mean()

        color = "tab:blue"
        ax1.set_xlabel("Training Episode", fontweight="bold")
        ax1.set_ylabel("Cumulative Episode Reward", color=color, fontweight="bold")
        ax1.plot(episodes, rewards, alpha=0.3, color=color, label="Raw Reward")
        ax1.plot(episodes, rolling_rew, color=color, linewidth=2.0, label="Moving Average")
        ax1.tick_params(axis="y", labelcolor=color)

        # Twin axis for SLA violation
        ax2 = ax1.twinx()
        color_sla = "tab:red"
        sla_pct = df_train["sla_violation_rate"] * 100.0
        rolling_sla = sla_pct.rolling(window=max(len(sla_pct) // 10, 5), min_periods=1).mean()
        ax2.set_ylabel("SLA Violation Rate (%)", color=color_sla, fontweight="bold")
        ax2.plot(episodes, rolling_sla, color=color_sla, linestyle="--", linewidth=1.8, label="SLA Viol % (MA)")
        ax2.tick_params(axis="y", labelcolor=color_sla)

        plt.title("MAPPO Multi-Agent Training Convergence & SLA Stabilization", fontweight="bold", pad=12)
        fig.tight_layout()
        out_path = plots_dir / "training_reward_curve.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Plot Generated] {out_path}")
    else:
        print(f"[Notice] {train_log_path} not found. Skipping training curve plot.")

    # -------------------------------------------------------------
    # 2. Baseline Comparison: Energy vs SLA Trade-off
    # -------------------------------------------------------------
    comp_path = results_dir / "baseline_comparison.csv"
    if comp_path.is_file():
        df_comp = pd.read_csv(comp_path)

        fig, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(12, 5))

        policies = df_comp["Policy"].str.replace("_", " ")
        energies = df_comp["Energy (J)"]
        sla_rates = df_comp["SLA Viol Rate (%)"]

        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"][: len(policies)]

        x_indices = np.arange(len(policies))

        # Energy consumption bar plot
        bars1 = ax_bar1.bar(x_indices, energies, color=colors, edgecolor="black", alpha=0.85)
        ax_bar1.set_ylabel("Total Energy Consumption (Joules)", fontweight="bold")
        ax_bar1.set_title("Energy Consumption across Policies", fontweight="bold")
        ax_bar1.set_xticks(x_indices)
        ax_bar1.set_xticklabels(policies, rotation=25, ha="right")
        for bar in bars1:
            yval = bar.get_height()
            ax_bar1.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval:.1f}J", ha="center", va="bottom", fontsize=9)

        # SLA violation rate bar plot
        bars2 = ax_bar2.bar(x_indices, sla_rates, color=colors, edgecolor="black", alpha=0.85)
        ax_bar2.set_ylabel("SLA Violation Rate (%)", fontweight="bold")
        ax_bar2.set_title("SLA Violations across Policies", fontweight="bold")
        ax_bar2.set_xticks(x_indices)
        ax_bar2.set_xticklabels(policies, rotation=25, ha="right")
        for bar in bars2:
            yval = bar.get_height()
            ax_bar2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out_tradeoff = plots_dir / "energy_vs_sla_tradeoff.png"
        fig.savefig(out_tradeoff, dpi=300)
        plt.close(fig)
        print(f"[Plot Generated] {out_tradeoff}")

        # -------------------------------------------------------------
        # 3. URLLC Latency vs 5ms SLA Limit
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        lat_mean = df_comp["URLLC Latency (ms)"]
        lat_p99 = df_comp["URLLC P99 Lat (ms)"]

        x = np.arange(len(policies))
        width = 0.35

        ax.bar(x - width / 2, lat_mean, width, label="Mean Latency", color="#4C72B0", edgecolor="black")
        ax.bar(x + width / 2, lat_p99, width, label="99th Percentile Latency", color="#C44E52", edgecolor="black")
        ax.axhline(5.0, color="red", linestyle="--", linewidth=1.5, label="URLLC SLA Limit (5.0 ms)")

        ax.set_ylabel("Latency (ms)", fontweight="bold")
        ax.set_title("URLLC Latency Performance vs SLA Requirement", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.legend()

        fig.tight_layout()
        out_lat = plots_dir / "latency_comparison.png"
        fig.savefig(out_lat, dpi=300)
        plt.close(fig)
        print(f"[Plot Generated] {out_lat}")

    else:
        print(f"[Notice] {comp_path} not found. Skipping baseline trade-off plots.")
        print("[Notice] Run 'python compare_baselines.py' to generate baseline data.")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    plot_all_results(cfg)
