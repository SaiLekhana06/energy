"""Automated tests for MARL agents, neural network architectures, and MAPPO trainer."""

import numpy as np
import pytest
import torch

from agents.marl_framework import MAPPOTrainer, RolloutBuffer
from agents.slicing_agent import DynamicSlicingAgent
from agents.sleeping_agent import XAppSleepingAgent
from models.networks import CentralizedCriticNetwork
from utils.config import load_config


@pytest.fixture
def config():
    return load_config("config.yaml")


def test_slicing_agent_forward():
    """Verify slicing agent outputs valid discrete action and finite log probability."""
    agent = DynamicSlicingAgent(obs_dim=13, action_dim=12, hidden_dim=64)
    obs = np.random.uniform(0.0, 1.0, size=(13,)).astype(np.float32)

    act, logp = agent.select_action(obs, deterministic=False)
    assert isinstance(act, int)
    assert 0 <= act < 12
    assert np.isfinite(logp)


def test_sleeping_agent_forward():
    """Verify sleeping agent outputs valid discrete state action {0, 1, 2}."""
    agent = XAppSleepingAgent(obs_dim=10, action_dim=3, hidden_dim=64)
    obs = np.random.uniform(0.0, 1.0, size=(10,)).astype(np.float32)

    act, logp = agent.select_action(obs, deterministic=False)
    assert isinstance(act, int)
    assert act in [0, 1, 2]
    assert np.isfinite(logp)


def test_centralized_critic():
    """Verify centralized critic outputs a finite scalar value V(s)."""
    critic = CentralizedCriticNetwork(state_dim=25, hidden_dim=64)
    state = torch.randn(4, 25)
    val = critic(state)

    assert val.shape == (4,)
    assert torch.isfinite(val).all()


def test_mappo_buffer_and_update(config):
    """Verify rollout collection and GAE update loop in MAPPO trainer."""
    trainer = MAPPOTrainer(
        config=config,
        obs_dim_slicing=13,
        obs_dim_sleeping=10,
        state_dim=25,
        num_slicing_actions=12,
        num_sleeping_actions=3,
    )

    # Populate buffer with dummy transitions
    for i in range(3):
        trainer.buffer.add(
            obs_slicing=np.zeros(13, dtype=np.float32),
            obs_sleeping=np.zeros(10, dtype=np.float32),
            state=np.zeros(25, dtype=np.float32),
            act_slicing=2,
            act_sleeping=1,
            logp_slicing=-1.5,
            logp_sleeping=-0.9,
            reward=1.2 + 0.1 * i,
            value=0.5,
            done=(i == 2),
        )

    losses = trainer.update(last_value=0.0)
    assert "loss_actor_slicing" in losses
    assert "loss_actor_sleeping" in losses
    assert "loss_critic" in losses
    assert len(trainer.buffer.rewards) == 0  # Buffer cleared
