from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EaaiBlock:
    name: str
    scenarios: tuple[str, ...]
    engineering_question: str


# Locked six-scenario protocol used by the paper and publication-scale runs.
EAAI_BLOCKS = (
    EaaiBlock(
        "nominal_and_cardinality_transfer",
        ("in_distribution", "unseen_small", "unseen_large"),
        "Can one released controller operate at unseen team and backlog sizes?",
    ),
    EaaiBlock(
        "typed_resource_imbalance",
        ("agv_heavy_ratio",),
        "Can dispatch account for a picker-scarce team composition?",
    ),
    EaaiBlock(
        "task_set_dynamics",
        ("fixed_team_dynamic_orders",),
        "Can dispatch handle a changing task set under a fixed team?",
    ),
    EaaiBlock(
        "coalition_reconfiguration",
        ("robot_breakdown",),
        "Does the service coalition reconfigure after interruption and restoration?",
    ),
)

# Deliberately excluded from the primary paper matrix.  They remain available
# in scenarios.py for optional sensitivity work, but cannot silently enter an
# EAAI main/ablation table or checkpoint-selection loop.
EAAI_EXCLUDED_DIAGNOSTICS = (
    "picker_heavy_ratio",
    "high_churn",
    "picker_shift",
    "peak_demand",
    "high_cancellation",
    "combined_disruption",
    "slow_pickers",
    "spatial_skew",
    "large_layout",
)

# The first three entries are the explicitly labelled pilot subset.  Seed 8191
# was locked as the sixth formal seed before the remaining publication runs;
# it was not selected by looking at method performance.
EAAI_PRELIMINARY_SEEDS = (1, 45, 789)
EAAI_FORMAL_SEEDS = (1, 45, 789, 2027, 4096, 8191)


# Values are simulator regimes, not claims that one step equals a universal
# physical unit.  A paper must cite the source used to calibrate each range.
REALISM_ASSUMPTIONS = {
    "order_process": {
        "implementation": "stochastic arrivals and cancellations within the selected regimes",
        "paper_obligation": "report the chosen step-to-minute mapping and cite the order-log source",
    },
    "robot_availability": {
        "implementation": "stochastic joins/leaves plus scheduled AGV failure and later repair",
        "paper_obligation": "map probabilities and repair duration to maintenance/charging records",
    },
    "human_service": {
        "implementation": "bounded stochastic pick delay",
        "paper_obligation": "justify delay ranges with time-study data or literature",
    },
    "order_cancellation": {
        "implementation": "per-step cancellation hazard before pickup",
        "paper_obligation": "report sensitivity and the implied episode-level cancellation rate",
    },
    "layout": {
        "implementation": "the configured warehouse grid used by the six evaluation regimes",
        "paper_obligation": "publish the map dimensions and depot rules",
    },
}


PRIMARY_METRICS = (
    "service_level",
    "throughput_per_100_steps",
    "deadline_miss_rate",
    "mean_flow_time",
)

SECONDARY_METRICS = (
    "travel_per_delivery",
    "utilization",
    "contention_per_delivery",
    "unserved_orders",
    "agent_joins",
    "agent_leaves",
    "forced_failures",
    "repairs",
    "candidate_coalitions_per_step",
    "selected_coalitions_per_step",
    "masked_coalitions_per_step",
    "reservation_rounds_per_step",
    "decoder_latency_ms_per_step",
)


def all_eaai_scenarios() -> list[str]:
    return [scenario for block in EAAI_BLOCKS for scenario in block.scenarios]


def core6_eaai_scenarios() -> list[str]:
    return all_eaai_scenarios()


def protocol_scenarios(name: str) -> list[str]:
    if name == "core6":
        return core6_eaai_scenarios()
    raise KeyError(f"unknown EAAI scenario set {name!r}")


def protocol_blocks(name: str) -> tuple[EaaiBlock, ...]:
    if name == "core6":
        return EAAI_BLOCKS
    raise KeyError(f"unknown EAAI scenario set {name!r}")


def infer_protocol(scenarios: list[str]) -> str:
    if scenarios == core6_eaai_scenarios():
        return "core6"
    raise ValueError("evaluation scenarios must exactly match the locked Core-6 order")


def validate_eaai_protocol() -> None:
    scenarios = all_eaai_scenarios()
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("EAAI scenario blocks must be disjoint")
    required = {
        "in_distribution",
        "unseen_small",
        "unseen_large",
        "agv_heavy_ratio",
        "fixed_team_dynamic_orders",
        "robot_breakdown",
    }
    if len(scenarios) != 6 or set(scenarios) != required:
        raise ValueError("EAAI Core-6 protocol drifted from the locked scenario set")
    if set(scenarios) & set(EAAI_EXCLUDED_DIAGNOSTICS):
        raise ValueError("excluded scenarios leaked into the EAAI Core-6 protocol")
