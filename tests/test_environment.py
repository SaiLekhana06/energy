"""Automated tests for O-RAN simulation environment, channel, and energy models."""

import numpy as np
import pytest

from environment.oran_env import ORanSlicingSleepingEnv
from utils.config import load_config


@pytest.fixture
def config():
    return load_config("config.yaml")


@pytest.fixture
def env(config):
    return ORanSlicingSleepingEnv(config, seed=42)


def test_env_initialization(env):
    """Verify observation spaces, action spaces, and dimension consistency."""
    assert env.obs_dim_slicing == 13
    assert env.obs_dim_sleeping == 10
    assert env.state_dim == 25
    assert env.num_slicing_actions == 12
    assert env.num_sleeping_actions == 3


def test_prb_budget_and_starvation_constraints(env):
    """Ensure every discrete PRB profile sums to 100 and no slice starves."""
    for idx, (embb, urllc, mmtc) in enumerate(env.PRB_PROFILES):
        total = embb + urllc + mmtc
        assert total == 100, f"Profile {idx} sum {total} != 100"
        assert embb >= 5, f"eMBB allocation in profile {idx} < min_prb (5)"
        assert urllc >= 5, f"URLLC allocation in profile {idx} < min_prb (5)"
        assert mmtc >= 5, f"mMTC allocation in profile {idx} < min_prb (5)"


def test_env_reset(env):
    """Verify reset returns valid observation dictionaries with correct dimensions."""
    obs_dict, info = env.reset(seed=123)

    assert "slicing" in obs_dict
    assert "sleeping" in obs_dict
    assert "state" in obs_dict

    assert obs_dict["slicing"].shape == (13,)
    assert obs_dict["sleeping"].shape == (10,)
    assert obs_dict["state"].shape == (25,)

    assert not np.isnan(obs_dict["slicing"]).any()
    assert not np.isnan(obs_dict["sleeping"]).any()
    assert not np.isnan(obs_dict["state"]).any()


def test_env_step_execution(env):
    """Verify one simulation step executes correctly, returns finite reward and metrics."""
    env.reset(seed=42)
    actions = {"slicing": 3, "sleeping": 1}
    next_obs, rewards, term, trunc, info = env.step(actions)

    assert "slicing" in rewards
    assert "sleeping" in rewards
    assert "global" in rewards
    assert np.isfinite(rewards["global"])

    assert "energy_joules" in info
    assert info["energy_joules"] > 0
    assert "sla_violation_rate" in info
    assert 0.0 <= info["sla_violation_rate"] <= 1.0
    assert "prb_utilization" in info
    assert 0.0 <= info["prb_utilization"] <= 1.0


def test_sleep_wake_transition_penalty(env):
    """Verify waking up from deep sleep triggers is_waking_up flag and transient penalty."""
    env.reset(seed=42)
    # Step 1: Enter deep sleep (2)
    env.step({"slicing": 3, "sleeping": 2})
    assert env.xapp_state == 2

    # Step 2: Wake up to active (0)
    _, _, _, _, info = env.step({"slicing": 3, "sleeping": 0})
    assert env.xapp_state == 0
    assert info["is_waking_up"] is True
    assert info["transition_cost"] >= 1.0
