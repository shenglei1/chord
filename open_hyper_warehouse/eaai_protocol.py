from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EaaiBlock:
    name: str
    scenarios: tuple[str, ...]
    engineering_question: str


# Locked before publication-scale runs.  These blocks deliberately distinguish
# capacity generalization, staffing changes, order-process uncertainty and
# physical/layout disruptions so that one favorable stress case cannot stand
# in for the complete applied-robotics claim.
EAAI_BLOCKS = (
    EaaiBlock(
        "nominal_and_cardinality_transfer",
        ("in_distribution", "unseen_small", "unseen_large"),
        "Can one released controller operate at unseen fleet, workforce and backlog sizes?",
    ),
    EaaiBlock(
        "fleet_and_workforce_reconfiguration",
        ("agv_heavy_ratio", "picker_heavy_ratio", "high_churn", "picker_shift"),
        "Can dispatch continue during fleet imbalance, charging/maintenance and shift turnover?",
    ),
    EaaiBlock(
        "online_demand_and_service_variability",
        ("fixed_team_dynamic_orders", "peak_demand", "high_cancellation"),
        "Can the controller absorb open order streams, cancellations and demand peaks?",
    ),
    EaaiBlock(
        "robot_reconfiguration_and_compound_disruption",
        ("robot_breakdown", "combined_disruption"),
        "Does coordination reconfigure after a repairable robot failure and combined stress?",
    ),
)

# Deliberately excluded from the primary paper matrix.  They remain available
# in scenarios.py for optional sensitivity work, but cannot silently enter an
# EAAI main/ablation table or checkpoint-selection loop.
EAAI_EXCLUDED_DIAGNOSTICS = ("slow_pickers", "spatial_skew", "large_layout")

# Minimal primary matrix for the six-seed EAAI paper. Each scenario isolates
# one part of the central claim; correlated demand/process stresses remain in
# the Full-12 supplementary robustness matrix.
EAAI_CORE8_BLOCKS = (
    EaaiBlock(
        "nominal_and_cardinality_transfer",
        ("in_distribution", "unseen_small", "unseen_large"),
        "Can one released controller operate at unseen team and backlog sizes?",
    ),
    EaaiBlock(
        "typed_workforce_imbalance",
        ("agv_heavy_ratio", "picker_heavy_ratio"),
        "Does explicit typed coalition modelling identify the scarce role?",
    ),
    EaaiBlock(
        "agent_and_task_openness",
        ("picker_shift", "fixed_team_dynamic_orders"),
        "Can scheduled workforce turnover and task openness be isolated under a common policy?",
    ),
    EaaiBlock(
        "coalition_reconfiguration",
        ("robot_breakdown",),
        "Does the service coalition dissolve and reform after failure and repair?",
    ),
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
        "implementation": "Poisson arrivals with stationary, mid-shift peak and two-wave profiles",
        "paper_obligation": "report the chosen step-to-minute mapping and cite the order-log source",
    },
    "robot_availability": {
        "implementation": "stochastic joins/leaves plus scheduled AGV failure and later repair",
        "paper_obligation": "map probabilities and repair duration to maintenance/charging records",
    },
    "human_service": {
        "implementation": "picker shift replacement and bounded stochastic pick delay",
        "paper_obligation": "justify delay and shift ranges with time-study data or literature",
    },
    "order_cancellation": {
        "implementation": "per-step cancellation hazard before pickup",
        "paper_obligation": "report sensitivity and the implied episode-level cancellation rate",
    },
    "layout": {
        "implementation": "nominal, enlarged, left-depot hotspot and split-depot layouts",
        "paper_obligation": "publish every map dimension and depot/hotspot rule",
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


def core8_eaai_scenarios() -> list[str]:
    return [scenario for block in EAAI_CORE8_BLOCKS for scenario in block.scenarios]


def protocol_scenarios(name: str) -> list[str]:
    if name == "core8":
        return core8_eaai_scenarios()
    if name == "full12":
        return all_eaai_scenarios()
    raise KeyError(f"unknown EAAI scenario set {name!r}")


def protocol_blocks(name: str) -> tuple[EaaiBlock, ...]:
    if name == "core8":
        return EAAI_CORE8_BLOCKS
    if name == "full12":
        return EAAI_BLOCKS
    raise KeyError(f"unknown EAAI scenario set {name!r}")


def infer_protocol(scenarios: list[str]) -> str:
    for name in ("core8", "full12"):
        if scenarios == protocol_scenarios(name):
            return name
    raise ValueError(
        "evaluation scenarios must exactly match the locked Core-8 or Full-12 order"
    )


def validate_eaai_protocol() -> None:
    scenarios = all_eaai_scenarios()
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("EAAI scenario blocks must be disjoint")
    if "in_distribution" not in scenarios:
        raise ValueError("EAAI protocol requires a nominal operating point")
    if not {"unseen_small", "unseen_large", "robot_breakdown", "picker_shift"} <= set(scenarios):
        raise ValueError("EAAI protocol is missing a required deployment stress")
    if len(scenarios) != 12:
        raise ValueError("EAAI Full-12 supplementary protocol must contain 12 scenarios")
    if set(scenarios) & set(EAAI_EXCLUDED_DIAGNOSTICS):
        raise ValueError("diagnostic-only scenarios leaked into the EAAI primary protocol")


def validate_core8_protocol() -> None:
    scenarios = core8_eaai_scenarios()
    if len(scenarios) != 8 or len(scenarios) != len(set(scenarios)):
        raise ValueError("EAAI Core-8 protocol must contain eight distinct scenarios")
    required = {
        "in_distribution",
        "unseen_small",
        "unseen_large",
        "agv_heavy_ratio",
        "picker_heavy_ratio",
        "picker_shift",
        "fixed_team_dynamic_orders",
        "robot_breakdown",
    }
    if set(scenarios) != required:
        raise ValueError("EAAI Core-8 protocol drifted from the preregistered set")
    if not set(scenarios) <= set(all_eaai_scenarios()):
        raise ValueError("Core-8 must be a strict subset of Full-12")
