"""Locked EAAI main-comparison registry and reproducibility metadata.

The registry prevents display names, implementation claims and learner
families from drifting between runners, manifests and the paper tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    key: str
    paper_label: str
    method_class: str
    learner_family: str
    implementation_level: str
    adaptation_boundary: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


EAAI_MAIN_METHODS = (
    BaselineSpec(
        key="chord_mappo",
        paper_label="CHORD-CAHR-MAPPO (ours)",
        method_class="proposed",
        learner_family="joint conditional MAPPO",
        implementation_level="native implementation",
        adaptation_boundary=(
            "capacity-aware AGV-picker-task coalition scoring, autoregressive "
            "hyperedge reservation and exact conditional joint-action PPO"
        ),
        source="this work",
    ),
    BaselineSpec(
        key="plato_faithful",
        paper_label="PLATO-faithful",
        method_class="published learning baseline",
        learner_family="MAPPO",
        implementation_level="paper-faithful clean-room implementation",
        adaptation_boundary=(
            "warehouse observation factorization only; pointer actor, four-statistic "
            "team query, bipartite two-layer GCN critic and MAPPO are retained"
        ),
        source="PLATO arXiv:2607.25082v2, Algorithm 1",
    ),
    BaselineSpec(
        key="mohito_official",
        paper_label="MOHITO-official + inactive-slot wrapper",
        method_class="published learning baseline",
        learner_family="MOHITO deterministic actor-critic",
        implementation_level="official network code with domain adapter",
        adaptation_boundary=(
            "warehouse Levi graph and inactive-agent wrapper; official Actor, "
            "Critic and GNN modules, replay and target networks are retained"
        ),
        source="oasys-mas/mohito-public@105ddccbacc92953a7aa6bd0d19f8ec299f7476d",
    ),
    BaselineSpec(
        key="entity_mappo",
        paper_label="Entity-MAPPO",
        method_class="controlled representation baseline",
        learner_family="MAPPO",
        implementation_level="native controlled implementation",
        adaptation_boundary="DeepSets-style masked means; no explicit relation edges",
        source="controlled baseline in this work",
    ),
    BaselineSpec(
        key="graph_mappo",
        paper_label="Graph-MAPPO",
        method_class="controlled representation baseline",
        learner_family="MAPPO",
        implementation_level="native controlled implementation",
        adaptation_boundary=(
            "binary pairwise 2-section of exactly the proposed task/zone incidence; "
            "matched encoder depth and hidden width, with the common factorized decoder"
        ),
        source="controlled baseline in this work",
    ),
    BaselineSpec(
        key="rolling_horizon",
        paper_label="Rolling-Horizon MILP",
        method_class="operational optimization baseline",
        learner_family="receding-horizon MILP",
        implementation_level="native optimization implementation",
        adaptation_boundary=(
            "same observable state, four planning periods, deterministic candidate "
            "screening, one-second solve limit and period-zero execution"
        ),
        source="formulation documented in docs/BASELINES.md",
    ),
)

EAAI_MAIN_BY_KEY = {spec.key: spec for spec in EAAI_MAIN_METHODS}
EAAI_MAIN_LEARNED_KEYS = tuple(
    spec.key for spec in EAAI_MAIN_METHODS if spec.key != "rolling_horizon"
)
EAAI_MAIN_OPERATIONAL_KEYS = ("rolling_horizon",)


def validate_main_registry() -> None:
    keys = [spec.key for spec in EAAI_MAIN_METHODS]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate method in EAAI main-comparison registry")
    required = {
        "chord_mappo",
        "plato_faithful",
        "mohito_official",
        "entity_mappo",
        "graph_mappo",
        "rolling_horizon",
    }
    if set(keys) != required:
        raise ValueError(f"EAAI main comparison must be exactly {sorted(required)}")


def baseline_metadata(key: str) -> dict[str, str]:
    try:
        return EAAI_MAIN_BY_KEY[key].to_dict()
    except KeyError:
        return {
            "key": key,
            "paper_label": key,
            "method_class": "appendix or ablation",
            "learner_family": "see experiment manifest",
            "implementation_level": "not a main-table method",
            "adaptation_boundary": "see method documentation",
            "source": "see docs/BASELINES.md",
        }
