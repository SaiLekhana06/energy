"""Automated tests for all 4 baseline policies."""

import numpy as np
import pytest

from baselines.always_active import AlwaysActivePolicy
from baselines.greedy_energy import GreedyThresholdPolicy
from baselines.single_agent_drl import SingleAgentDRLPolicy
from baselines.static_slicing import StaticSlicingPolicy
from utils.config import load_config


@pytest.fixture
def config():
    return load_config("config.yaml")


@pytest.fixture
def dummy_obs():
    return {
        "slicing": np.random.uniform(0.1, 0.9, size=(13,)).astype(np.float32),
        "sleeping": np.random.uniform(0.1, 0.9, size=(10,)).astype(np.float32),
        "state": np.random.uniform(0.1, 0.9, size=(25,)).astype(np.float32),
    }


def test_static_slicing_baseline(config, dummy_obs):
    policy = StaticSlicingPolicy(config)
    act = policy.select_action(dummy_obs)
    assert act["slicing"] == 3
    assert act["sleeping"] == 0


def test_always_active_baseline(config, dummy_obs):
    policy = AlwaysActivePolicy(config)
    act = policy.select_action(dummy_obs)
    assert act["sleeping"] == 0
    assert 0 <= act["slicing"] < 12


def test_greedy_threshold_baseline(config, dummy_obs):
    policy = GreedyThresholdPolicy(config)
    act = policy.select_action(dummy_obs)
    assert act["sleeping"] in [0, 1, 2]
    assert 0 <= act["slicing"] < 12


def test_single_agent_baseline(config, dummy_obs):
    policy = SingleAgentDRLPolicy(config)
    act = policy.select_action(dummy_obs)
    assert act["sleeping"] in [0, 1, 2]
    assert 0 <= act["slicing"] < 12
