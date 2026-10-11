"""Experiment logger for console output and CSV metric tracking."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ExperimentLogger:
    """Logs episode metrics, displays progress, and writes structured CSVs."""

    def __init__(self, log_dir: str = "./results", experiment_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []

    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Append a step or episode record to in-memory history."""
        self.history.append(step_data)

    def print_progress(self, episode: int, total_episodes: int, metrics: Dict[str, float]) -> None:
        """Print a formatted console message with key indicators."""
        reward = metrics.get("reward", 0.0)
        energy_j = metrics.get("energy_joules", 0.0)
        sla_viol = metrics.get("sla_violation_rate", 0.0) * 100.0
        urllc_lat = metrics.get("urllc_latency_ms", 0.0)
        prb_util = metrics.get("prb_utilization", 0.0) * 100.0

        print(
            f"Ep [{episode:4d}/{total_episodes:4d}] | "
            f"Rew: {reward:8.2f} | "
            f"Energy: {energy_j:7.1f} J | "
            f"SLA Viol: {sla_viol:5.2f}% | "
            f"URLLC Lat: {urllc_lat:5.2f} ms | "
            f"PRB Util: {prb_util:5.1f}%"
        )

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """Export logged history to a CSV file.

        Args:
            filename: Optional custom filename. Defaults to <experiment_name>_history.csv.

        Returns:
            Path to the saved CSV file.
        """
        if not filename:
            filename = f"{self.experiment_name}_history.csv"
        file_path = self.log_dir / filename

        df = pd.DataFrame(self.history)
        df.to_csv(file_path, index=False)
        return file_path
