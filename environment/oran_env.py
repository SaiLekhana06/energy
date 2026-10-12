"""Gymnasium-compatible multi-agent O-RAN environment for dynamic slicing and xApp sleeping."""

from typing import Any, Dict, List, Optional, Tuple
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _DummyGymEnv:
        """Fallback Env class if gymnasium is not installed."""
        pass
    class _DummySpaces:
        """Fallback spaces if gymnasium is not installed."""
        class Box:
            def __init__(self, low=0.0, high=1.0, shape=None, dtype=None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        class Discrete:
            def __init__(self, n=0):
                self.n = n
    gym = type("gym", (), {"Env": _DummyGymEnv})()
    spaces = _DummySpaces()
import numpy as np

from environment.channel_model import WirelessChannelModel
from environment.energy_model import ORanEnergyModel
from environment.traffic_generator import SyntheticTrafficGenerator


class ORanSlicingSleepingEnv(gym.Env):
    """Multi-Agent 5G O-RAN Environment for Joint Dynamic PRB Slicing and xApp Sleeping.

    Academic Context: Minor Project-II, ECE Dept, Vel Tech R&D Institute.

    Agents:
    - Agent 1 (Slicing Agent): Allocates 100 PRBs dynamically across eMBB, URLLC, and mMTC slices.
    - Agent 2 (Sleeping Agent): Controls Near-RT RIC xApp instances and RF chains (Active, Light Sleep, Deep Sleep).
    """

    metadata = {"render_modes": ["human"]}

    # Pre-defined discrete PRB allocation profiles guaranteeing total_prbs = 100
    # and min_prb_per_slice >= 5 for all slices (eMBB, URLLC, mMTC)
    PRB_PROFILES: List[Tuple[int, int, int]] = [
        (70, 15, 15),  # Profile 0: Heavy eMBB download burst
        (60, 25, 15),  # Profile 1: Moderate eMBB + elevated URLLC
        (50, 35, 15),  # Profile 2: High URLLC priority
        (50, 30, 20),  # Profile 3: Standard balanced
        (40, 45, 15),  # Profile 4: Extreme URLLC burst
        (40, 30, 30),  # Profile 5: High mMTC IoT batch
        (35, 35, 30),  # Profile 6: Low eMBB, high URLLC + mMTC
        (55, 20, 25),  # Profile 7: eMBB + mMTC
        (45, 25, 30),  # Profile 8: Distributed
        (30, 50, 20),  # Profile 9: Critical URLLC priority
        (65, 20, 15),  # Profile 10: Video streaming focus
        (30, 30, 40),  # Profile 11: Sensor event focus
    ]

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        super().__init__()
        self.cfg = config
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Simulation settings
        sim_cfg = config.get("simulation", {})
        self.total_prbs = int(sim_cfg.get("total_prbs", 100))
        self.min_prb = int(sim_cfg.get("min_prb_per_slice", 5))
        self.num_cells = int(sim_cfg.get("num_cells", 3))
        self.max_steps = int(sim_cfg.get("episode_length_steps", 100))

        # Reward weights
        rew_cfg = config.get("rewards", {})
        self.w_qos = float(rew_cfg.get("w_qos", 1.0))
        self.w_energy = float(rew_cfg.get("w_energy", 1.2))
        self.w_sla = float(rew_cfg.get("w_sla", 2.5))
        self.w_trans = float(rew_cfg.get("w_trans", 0.3))
        self.w_util = float(rew_cfg.get("w_util", 0.5))

        # Max theoretical power for normalization: (base + 3*active_rf + active_xapp)
        self.max_power_watts = (
            config.get("energy", {}).get("p_base_watts", 180.0)
            + self.num_cells * config.get("energy", {}).get("p_rf_active_watts", 120.0)
            + config.get("energy", {}).get("p_xapp_active_watts", 45.0)
        )

        # Sub-modules
        self.traffic_gen = SyntheticTrafficGenerator(config, self.rng)
        self.channel_model = WirelessChannelModel(config, self.rng)
        self.energy_model = ORanEnergyModel(config)

        # Define Observation Spaces
        # Agent 1 (Slicing): dim 13
        # [demands (3), prb_alloc (3), latencies (3), xapp_state (1), sla_violations (3)]
        self.obs_dim_slicing = 13
        self.observation_space_slicing = spaces.Box(
            low=0.0, high=10.0, shape=(self.obs_dim_slicing,), dtype=np.float32
        )

        # Agent 2 (Sleeping): dim 10
        # [demands (3), xapp_state (1), steps_in_state (1), sla_violations (3), power_ratio (1), urllc_pressure (1)]
        self.obs_dim_sleeping = 10
        self.observation_space_sleeping = spaces.Box(
            low=0.0, high=10.0, shape=(self.obs_dim_sleeping,), dtype=np.float32
        )

        # Centralized Critic Global State: dim 25
        self.state_dim = self.obs_dim_slicing + self.obs_dim_sleeping + 2

        # Define Action Spaces
        # Agent 1: Selection among discrete PRB distribution profiles
        self.num_slicing_actions = len(self.PRB_PROFILES)
        self.action_space_slicing = spaces.Discrete(self.num_slicing_actions)

        # Agent 2: 0=Active, 1=Light Sleep, 2=Deep Sleep
        self.num_sleeping_actions = 3
        self.action_space_sleeping = spaces.Discrete(self.num_sleeping_actions)

        # Internal state tracking
        self.current_step = 0
        self.xapp_state = 0  # 0=Active
        self.previous_xapp_state = 0
        self.steps_in_current_state = 0
        self.current_prbs = (50, 30, 20)
        self.last_demands = {"embb": 50.0, "urllc": 12.0, "mmtc": 8.0}
        self.last_metrics: Dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: Any = None,
        options: Any = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)

        self.traffic_gen.reset(self.rng)
        self.current_step = 0
        self.xapp_state = 0
        self.previous_xapp_state = 0
        self.steps_in_current_state = 0
        self.current_prbs = self.PRB_PROFILES[3]  # Standard balanced (50, 30, 20)

        # Initial traffic sample
        self.last_demands = self.traffic_gen.generate(0)

        obs_slicing = self._get_obs_slicing()
        obs_sleeping = self._get_obs_sleeping()
        global_state = self._get_global_state(obs_slicing, obs_sleeping)

        obs_dict = {
            "slicing": obs_slicing,
            "sleeping": obs_sleeping,
            "state": global_state,
        }
        info = {"step": 0, "initial_prbs": self.current_prbs}
        return obs_dict, info

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        """Execute one simulation step with joint actions from both agents.

        Args:
            actions: Dict with keys 'slicing' (int in [0, 11]) and 'sleeping' (int in [0, 2]).

        Returns:
            Tuple of (observations_dict, rewards_dict, terminated, truncated, info_dict).
        """
        self.current_step += 1

        # 1. Parse actions
        slicing_act = int(actions.get("slicing", 3))
        sleeping_act = int(actions.get("sleeping", 0))

        # Enforce valid action ranges
        slicing_act = max(0, min(slicing_act, self.num_slicing_actions - 1))
        sleeping_act = max(0, min(sleeping_act, self.num_sleeping_actions - 1))

        # Apply PRB allocation
        self.current_prbs = self.PRB_PROFILES[slicing_act]
        assert sum(self.current_prbs) == self.total_prbs, "PRB allocation must sum to total_prbs"
        assert all(p >= self.min_prb for p in self.current_prbs), "Each slice must receive at least min_prb"

        # Apply Sleep/Wake state transition
        self.previous_xapp_state = self.xapp_state
        self.xapp_state = sleeping_act
        if self.xapp_state == self.previous_xapp_state:
            self.steps_in_current_state += 1
        else:
            self.steps_in_current_state = 1

        # 2. Generate time-varying traffic demand
        demands = self.traffic_gen.generate(self.current_step)
        self.last_demands = demands

        # 3. Energy computation
        power_w, energy_j, is_waking_up = self.energy_model.compute_energy(
            current_state=self.xapp_state,
            previous_state=self.previous_xapp_state,
            num_cells=self.num_cells,
        )

        # 4. Radio Channel, Throughput, and Latency Evaluation
        embb_m = self.channel_model.compute_slice_metrics(
            "embb", self.current_prbs[0], demands["embb"], self.xapp_state, is_waking_up
        )
        urllc_m = self.channel_model.compute_slice_metrics(
            "urllc", self.current_prbs[1], demands["urllc"], self.xapp_state, is_waking_up
        )
        mmtc_m = self.channel_model.compute_slice_metrics(
            "mmtc", self.current_prbs[2], demands["mmtc"], self.xapp_state, is_waking_up
        )

        # Aggregate SLA violations
        sla_violations = [embb_m["sla_violated"], urllc_m["sla_violated"], mmtc_m["sla_violated"]]
        sla_violation_rate = float(np.mean(sla_violations))

        # QoS satisfaction ratio: achieved / demanded
        qos_ratios = [
            embb_m["achieved_throughput_mbps"] / (demands["embb"] + 1e-6),
            urllc_m["achieved_throughput_mbps"] / (demands["urllc"] + 1e-6),
            mmtc_m["achieved_throughput_mbps"] / (demands["mmtc"] + 1e-6),
        ]
        qos_satisfaction = float(np.clip(np.mean(qos_ratios), 0.0, 1.0))

        # Overall PRB utilization
        avg_prb_util = float(
            (embb_m["prb_utilization"] + urllc_m["prb_utilization"] + mmtc_m["prb_utilization"]) / 3.0
        )

        # State transition cost
        state_transition_cost = 1.0 if (self.xapp_state != self.previous_xapp_state) else 0.0
        if is_waking_up:
            state_transition_cost += 1.0  # Extra penalty for waking up delay shock

        # Normalized power ratio
        norm_power = float(power_w / self.max_power_watts)

        # 5. Reward Calculation
        # R = w_qos * QoS - w_energy * Energy - w_sla * SLA_Violations - w_trans * TransCost + w_util * Util
        reward = (
            self.w_qos * qos_satisfaction
            - self.w_energy * norm_power
            - self.w_sla * sla_violation_rate
            - self.w_trans * state_transition_cost
            + self.w_util * avg_prb_util
        )

        # Shared cooperative reward in cooperative CTDE framework
        rewards = {
            "slicing": float(np.round(reward, 4)),
            "sleeping": float(np.round(reward, 4)),
            "global": float(np.round(reward, 4)),
        }

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Gather metrics
        self.last_metrics = {
            "step": self.current_step,
            "power_watts": power_w,
            "energy_joules": energy_j,
            "is_waking_up": is_waking_up,
            "xapp_state": self.xapp_state,
            "prb_embb": self.current_prbs[0],
            "prb_urllc": self.current_prbs[1],
            "prb_mmtc": self.current_prbs[2],
            "demand_embb": demands["embb"],
            "demand_urllc": demands["urllc"],
            "demand_mmtc": demands["mmtc"],
            "throughput_embb": embb_m["achieved_throughput_mbps"],
            "throughput_urllc": urllc_m["achieved_throughput_mbps"],
            "throughput_mmtc": mmtc_m["achieved_throughput_mbps"],
            "latency_embb": embb_m["latency_ms"],
            "latency_urllc": urllc_m["latency_ms"],
            "latency_mmtc": mmtc_m["latency_ms"],
            "sla_violation_rate": sla_violation_rate,
            "prb_utilization": avg_prb_util,
            "qos_satisfaction": qos_satisfaction,
            "transition_cost": state_transition_cost,
            "reward": reward,
        }

        # Observations
        obs_slicing = self._get_obs_slicing()
        obs_sleeping = self._get_obs_sleeping()
        global_state = self._get_global_state(obs_slicing, obs_sleeping)

        obs_dict = {
            "slicing": obs_slicing,
            "sleeping": obs_sleeping,
            "state": global_state,
        }

        return obs_dict, rewards, terminated, truncated, self.last_metrics

    def _get_obs_slicing(self) -> np.ndarray:
        """Construct Agent 1 (Slicing) observation vector (dim 13)."""
        # Normalized demands (scaled by 100 Mbps)
        demands = np.array([
            self.last_demands.get("embb", 50.0) / 100.0,
            self.last_demands.get("urllc", 12.0) / 100.0,
            self.last_demands.get("mmtc", 8.0) / 100.0,
        ], dtype=np.float32)

        # Current PRB allocation ratios
        prbs = np.array([
            self.current_prbs[0] / self.total_prbs,
            self.current_prbs[1] / self.total_prbs,
            self.current_prbs[2] / self.total_prbs,
        ], dtype=np.float32)

        # Latencies normalized (scaled by 100 ms)
        lat = np.array([
            self.last_metrics.get("latency_embb", 15.0) / 100.0,
            self.last_metrics.get("latency_urllc", 3.0) / 10.0,  # URLLC scaled by 10ms for sensitivity
            self.last_metrics.get("latency_mmtc", 30.0) / 100.0,
        ], dtype=np.float32)

        # xApp state normalized
        state_val = np.array([self.xapp_state / 2.0], dtype=np.float32)

        # SLA violation flags
        sla = np.array([
            float(self.last_metrics.get("latency_embb", 15.0) > 25.0),
            float(self.last_metrics.get("latency_urllc", 3.0) > 5.0),
            float(self.last_metrics.get("latency_mmtc", 30.0) > 80.0),
        ], dtype=np.float32)

        obs = np.concatenate([demands, prbs, lat, state_val, sla]).astype(np.float32)
        return obs

    def _get_obs_sleeping(self) -> np.ndarray:
        """Construct Agent 2 (Sleeping) observation vector (dim 10)."""
        demands = np.array([
            self.last_demands.get("embb", 50.0) / 100.0,
            self.last_demands.get("urllc", 12.0) / 100.0,
            self.last_demands.get("mmtc", 8.0) / 100.0,
        ], dtype=np.float32)

        state_val = np.array([self.xapp_state / 2.0], dtype=np.float32)
        steps_norm = np.array([min(self.steps_in_current_state, 50) / 50.0], dtype=np.float32)

        sla = np.array([
            float(self.last_metrics.get("latency_embb", 15.0) > 25.0),
            float(self.last_metrics.get("latency_urllc", 3.0) > 5.0),
            float(self.last_metrics.get("latency_mmtc", 30.0) > 80.0),
        ], dtype=np.float32)

        power_ratio = np.array([
            self.last_metrics.get("power_watts", self.max_power_watts) / self.max_power_watts
        ], dtype=np.float32)

        # URLLC traffic load pressure
        urllc_pressure = np.array([
            self.last_demands.get("urllc", 12.0) / (self.current_prbs[1] * 0.36 * 3.0 + 1e-6)
        ], dtype=np.float32)

        obs = np.concatenate([demands, state_val, steps_norm, sla, power_ratio, urllc_pressure]).astype(np.float32)
        return obs

    def _get_global_state(self, obs_slicing: np.ndarray, obs_sleeping: np.ndarray) -> np.ndarray:
        """Construct Centralized State for Centralized Critic (dim 25)."""
        step_progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        total_traffic = np.array([sum(self.last_demands.values()) / 200.0], dtype=np.float32)
        return np.concatenate([obs_slicing, obs_sleeping, step_progress, total_traffic]).astype(np.float32)
