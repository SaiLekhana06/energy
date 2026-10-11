"""Automated integration tests for checkpoint saving/loading and micro training pipeline."""

from pathlib import Path
import pytest
import torch

from models.networks import load_checkpoint, save_checkpoint
from train import train_mappo
from utils.config import load_config


@pytest.fixture
def config():
    return load_config("config.yaml")


def test_checkpoint_save_and_load(tmp_path):
    """Verify state dictionary saving and loading roundtrip."""
    dummy_payload = {
        "episode": 10,
        "dummy_weights": torch.tensor([1.0, 2.0, 3.0]),
        "info": "unit_test_checkpoint",
    }
    ckpt_file = tmp_path / "test_model.pt"

    save_checkpoint(dummy_payload, str(ckpt_file))
    assert ckpt_file.is_file()

    loaded = load_checkpoint(str(ckpt_file))
    assert loaded["episode"] == 10
    assert torch.equal(loaded["dummy_weights"], torch.tensor([1.0, 2.0, 3.0]))
    assert loaded["info"] == "unit_test_checkpoint"


def test_short_training_pipeline_execution(config, tmp_path):
    """Run a fast 2-episode training execution to verify end-to-end integration."""
    # Override directories to temporary test path
    config["paths"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    config["paths"]["results_dir"] = str(tmp_path / "results")

    final_ckpt = train_mappo(config, episodes_override=2)
    assert Path(final_ckpt).is_file()
    assert (tmp_path / "results" / "mappo_training_log.csv").is_file()
