from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - project officially requires 3.11+
    from pip._vendor import tomli as tomllib


@dataclass(slots=True)
class EnvConfig:
    width: int = 16
    height: int = 12
    max_agvs: int = 20
    max_pickers: int = 10
    max_tasks: int = 64
    initial_agvs_min: int = 8
    initial_agvs_max: int = 14
    initial_pickers_min: int = 4
    initial_pickers_max: int = 7
    initial_tasks_min: int = 12
    initial_tasks_max: int = 24
    episode_steps: int = 256
    order_arrival_rate: float = 0.20
    order_cancel_probability: float = 0.005
    agent_join_probability: float = 0.015
    agent_leave_probability: float = 0.010
    deadline_min: int = 45
    deadline_max: int = 110
    step_penalty: float = 0.002
    contention_penalty: float = 0.02
    deadline_penalty: float = 0.25
    pickup_reward: float = 0.10
    delivery_reward: float = 1.0
    num_zones: int = 4
    task_hotspot_probability: float = 0.0
    arrival_profile: str = "stationary"
    agent_spawn_mode: str = "uniform_depot"
    picker_service_delay_min: int = 0
    picker_service_delay_max: int = 0
    scheduled_breakdown_step: int = -1
    scheduled_breakdown_count: int = 0
    scheduled_repair_step: int = -1
    scheduled_repair_count: int = 0

    @property
    def max_agents(self) -> int:
        return self.max_agvs + self.max_pickers

    def validate(self) -> None:
        if not (1 <= self.initial_agvs_min <= self.initial_agvs_max <= self.max_agvs):
            raise ValueError("invalid AGV population bounds")
        if not (1 <= self.initial_pickers_min <= self.initial_pickers_max <= self.max_pickers):
            raise ValueError("invalid picker population bounds")
        if not (0 <= self.initial_tasks_min <= self.initial_tasks_max <= self.max_tasks):
            raise ValueError("invalid task population bounds")
        if self.width < 4 or self.height < 4 or self.num_zones < 1:
            raise ValueError("warehouse dimensions and zones must be positive")
        if not 0.0 <= self.task_hotspot_probability <= 1.0:
            raise ValueError("task_hotspot_probability must be in [0, 1]")
        if self.arrival_profile not in {"stationary", "mid_shift_peak", "two_waves"}:
            raise ValueError("unknown arrival_profile")
        if self.agent_spawn_mode not in {"uniform_depot", "left_depot", "split_depot"}:
            raise ValueError("unknown agent_spawn_mode")
        if not 0 <= self.picker_service_delay_min <= self.picker_service_delay_max:
            raise ValueError("invalid picker service delay bounds")
        if self.scheduled_breakdown_count < 0 or self.scheduled_breakdown_count > self.max_agvs:
            raise ValueError("invalid scheduled breakdown count")
        if self.scheduled_repair_count < 0 or self.scheduled_repair_count > self.max_agvs:
            raise ValueError("invalid scheduled repair count")
        if self.scheduled_repair_count and self.scheduled_repair_step <= self.scheduled_breakdown_step:
            raise ValueError("scheduled repair must occur after the scheduled breakdown")


@dataclass(slots=True)
class TrainConfig:
    total_steps: int = 2_000_000
    num_envs: int = 8
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001
    entropy_anneal_fraction: float = 0.60
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    encoder_layers: int = 2
    checkpoint_interval: int = 100_000
    device: str = "auto"
    # 0 keeps the original in-process simulator. Setting this equal to
    # num_envs creates one persistent Windows-safe process per environment.
    parallel_env_workers: int = 0


@dataclass(slots=True)
class ModelConfig:
    coalition_top_k: int = 3
    # CAHR selects a conflict-free set of service coalitions autoregressively.
    # Zero uses the physical upper bound min(max_agvs, max_pickers).
    reservation_max_rounds: int = 0
    num_groups: int = 4
    attention_heads: int = 4
    # MOHITO-only settings follow the official UAI 2025 implementation.  They
    # are intentionally separated from the shared environment, seed, horizon,
    # discount and interaction-budget controls used for fairness.
    mohito_actor_lr: float = 0.009
    mohito_critic_lr: float = 0.01
    mohito_hidden_dim: int = 24
    mohito_layers: int = 3
    mohito_heads: int = 2
    mohito_tau: float = 0.005
    mohito_replay_size: int = 1_000
    mohito_batch_size: int = 16
    mohito_warmup_steps: int = 1_000
    # The official implementation exposes the same replay update cadence as
    # experiences_before_update/backups_per_update.  Updating every warehouse
    # transition is prohibitively expensive at the larger EAAI scale.
    mohito_experiences_before_update: int = 32
    mohito_backups_per_update: int = 1
    mohito_initial_epsilon: float = 0.9
    mohito_epsilon_decay: float = 0.999
    mohito_min_epsilon: float = 0.001


@dataclass(slots=True)
class EvaluationConfig:
    episodes: int = 100
    bootstrap_samples: int = 5_000
    deterministic: bool = True
    scenario_names: list[str] = field(default_factory=lambda: ["in_distribution"])


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.env.validate()
        if self.train.total_steps <= 0 or self.train.num_envs <= 0:
            raise ValueError("training steps and number of environments must be positive")
        if self.train.total_steps % self.train.num_envs:
            raise ValueError("total_steps must be divisible by num_envs for an exact fair budget")
        if self.train.parallel_env_workers not in {0, self.train.num_envs}:
            raise ValueError("parallel_env_workers must be 0 or equal num_envs")
        if not 0.0 < self.train.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if self.train.entropy_coef < 0.0 or self.train.entropy_coef_final < 0.0:
            raise ValueError("entropy coefficients must be non-negative")
        if not 0.0 < self.train.entropy_anneal_fraction <= 1.0:
            raise ValueError("entropy_anneal_fraction must be in (0, 1]")
        if self.model.coalition_top_k < 1 or self.model.reservation_max_rounds < 0:
            raise ValueError("invalid coalition candidate/reservation settings")
        if self.model.coalition_top_k > self.env.max_agents:
            raise ValueError("coalition_top_k cannot exceed padded agent capacity")
        if self.model.mohito_experiences_before_update < 1:
            raise ValueError("mohito_experiences_before_update must be positive")
        if self.model.mohito_backups_per_update < 1:
            raise ValueError("mohito_backups_per_update must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CORE6_SCENARIOS = {
    "in_distribution",
    "unseen_small",
    "unseen_large",
    "agv_heavy_ratio",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
}


def _construct(
    cls: type,
    values: dict[str, Any] | None,
    *,
    archive_compatibility: bool = False,
):
    values = dict(values or {})
    if archive_compatibility and cls is EvaluationConfig and "scenario_names" in values:
        values["scenario_names"] = [
            name
            for name in values["scenario_names"]
            if name in _CORE6_SCENARIOS
        ]
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown and not archive_compatibility:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**{key: value for key, value in values.items() if key in known})


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    cfg = ExperimentConfig(
        env=_construct(EnvConfig, raw.get("env")),
        train=_construct(TrainConfig, raw.get("train")),
        model=_construct(ModelConfig, raw.get("model")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
    )
    cfg.validate()
    return cfg


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    # Checkpoints are immutable experiment artifacts. Their metadata may
    # contain retired no-op fields, but the executable configuration is
    # canonicalized to the current schema and locked Core-6 scenario set.
    cfg = ExperimentConfig(
        env=_construct(EnvConfig, raw.get("env"), archive_compatibility=True),
        train=_construct(TrainConfig, raw.get("train"), archive_compatibility=True),
        model=_construct(ModelConfig, raw.get("model"), archive_compatibility=True),
        evaluation=_construct(
            EvaluationConfig,
            raw.get("evaluation"),
            archive_compatibility=True,
        ),
    )
    cfg.validate()
    return cfg
