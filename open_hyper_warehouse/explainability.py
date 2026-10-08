"""Post-hoc semantic hyperedge attribution for CHORD-CAHR.

This is a model-internal masking analysis, not a claim of causal explanation of
the physical warehouse. It requires no retraining and never changes a policy.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .buffer import to_torch
from .config import ExperimentConfig
from .env import OpenWarehouseEnv
from .evaluation import load_policy
from .hypergraph import IncidenceBatch
from .models import OpenHyperPolicy
from .scenarios import make_scenario


EXPLANATION_SCENARIO = "robot_breakdown"
EXPLANATION_EVENTS = ("pre_breakdown", "post_breakdown", "post_repair")


def _repeat_observation(obs: dict[str, Tensor], count: int) -> dict[str, Tensor]:
    return {key: value.expand(count, *value.shape[1:]).clone() for key, value in obs.items()}


def _mean_over_decisions(values: Tensor, decision_mask: Tensor) -> Tensor:
    weights = decision_mask.to(values.dtype)
    return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)


def _edge_semantics(
    edge: int, incidence: IncidenceBatch, obs: dict[str, np.ndarray], max_agents: int, max_tasks: int
) -> dict[str, Any]:
    members = torch.nonzero(incidence.matrix[0, :, edge] > 0, as_tuple=False).flatten().tolist()
    agent_slots = [int(node) for node in members if node < max_agents]
    task_slots = [int(node - max_agents) for node in members if node >= max_agents]
    agent_features = obs["agent_features"]
    return {
        "edge_index": int(edge),
        "edge_type": "task_coalition" if edge < max_tasks else "shared_zone",
        "semantic_id": int(edge if edge < max_tasks else edge - max_tasks),
        "agent_slots": agent_slots,
        "agv_slots": [slot for slot in agent_slots if agent_features[slot, 0] > 0.5],
        "picker_slots": [slot for slot in agent_slots if agent_features[slot, 1] > 0.5],
        "task_slots": task_slots,
        "cardinality": len(members),
    }


def _masked_outputs(
    policy: OpenHyperPolicy,
    obs: dict[str, Tensor],
    incidence: IncidenceBatch,
    edge_groups: list[list[int]],
):
    count = len(edge_groups)
    repeated_obs = _repeat_observation(obs, count)
    matrix = incidence.matrix.expand(count, -1, -1).clone()
    edge_mask = incidence.edge_mask.expand(count, -1).clone()
    for row, edges in enumerate(edge_groups):
        if edges:
            matrix[row, :, edges] = 0.0
            edge_mask[row, edges] = False
    intervention = IncidenceBatch(matrix, edge_mask)
    if getattr(policy, "uses_coalition_reservation", False):
        return policy.reservation_outputs_with_incidence(repeated_obs, intervention)
    return policy.forward_with_incidence(repeated_obs, intervention)


def _conditional_distributions(
    scores: Tensor, stop: Tensor, conditional_masks: Tensor
) -> Tensor:
    base = torch.cat([stop.unsqueeze(-1), scores], dim=-1)
    rows = []
    for decision_round in range(conditional_masks.shape[1]):
        logits = base.masked_fill(~conditional_masks[:, decision_round].bool(), -1e9)
        rows.append(torch.softmax(logits, dim=-1))
    return torch.stack(rows, dim=1)


@torch.no_grad()
def semantic_edge_attribution(
    policy: OpenHyperPolicy,
    obs_numpy: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Measure prediction changes caused by masking each semantic hyperedge."""

    obs = to_torch({key: value[None] for key, value in obs_numpy.items()}, device)
    incidence = policy.build_incidence(obs)
    active_edges = torch.nonzero(incidence.edge_mask[0], as_tuple=False).flatten().tolist()
    if getattr(policy, "uses_coalition_reservation", False):
        actions, _, _, _, trace = policy.act(
            obs, deterministic=True, return_trace=True
        )
        base_scores, base_stop, _, base_value = policy.reservation_outputs_with_incidence(
            obs, incidence
        )
        base_prob = _conditional_distributions(
            base_scores, base_stop, trace["conditional_masks"]
        )
        selected = trace["sequence"]
        selected_prob = base_prob.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        decision_mask = trace["round_mask"].bool()

        intervention_scores, intervention_stop, _, intervention_value = _masked_outputs(
            policy, obs, incidence, [[edge] for edge in active_edges]
        )
        intervention_masks = trace["conditional_masks"].expand(
            len(active_edges), -1, -1
        )
        intervention_prob = _conditional_distributions(
            intervention_scores, intervention_stop, intervention_masks
        )
        selected_for_rows = selected.expand(len(active_edges), -1)
        intervened_selected_prob = intervention_prob.gather(
            -1, selected_for_rows.unsqueeze(-1)
        ).squeeze(-1)
        probability_drop = selected_prob.expand_as(intervened_selected_prob) - intervened_selected_prob
        p = base_prob.expand_as(intervention_prob).clamp_min(1e-12)
        q = intervention_prob.clamp_min(1e-12)
        mixture = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.log() - mixture.log())).sum(-1)
            + (q * (q.log() - mixture.log())).sum(-1)
        )
        row_mask = decision_mask.expand(len(active_edges), -1)
        mean_drop = _mean_over_decisions(probability_drop, row_mask)
        mean_abs_drop = _mean_over_decisions(probability_drop.abs(), row_mask)
        mean_js = _mean_over_decisions(js, row_mask)
        value_drop = base_value.expand_as(intervention_value) - intervention_value
        selected_agent_actions = actions
    else:
        # Compatibility for pilot_factorized_decoder checkpoints only.
        base = policy.forward_with_incidence(obs, incidence)
        base_prob = torch.softmax(base.logits, dim=-1)
        selected = base.logits.argmax(dim=-1)
        selected_prob = base_prob.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        decision_mask = base.decision_mask

        interventions = _masked_outputs(policy, obs, incidence, [[edge] for edge in active_edges])
        intervention_prob = torch.softmax(interventions.logits, dim=-1)
        selected_for_rows = selected.expand(len(active_edges), -1)
        intervened_selected_prob = intervention_prob.gather(
            -1, selected_for_rows.unsqueeze(-1)
        ).squeeze(-1)
        probability_drop = selected_prob.expand_as(intervened_selected_prob) - intervened_selected_prob
        p = base_prob.expand_as(intervention_prob).clamp_min(1e-12)
        q = intervention_prob.clamp_min(1e-12)
        mixture = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.log() - mixture.log())).sum(-1)
            + (q * (q.log() - mixture.log())).sum(-1)
        )
        row_mask = decision_mask.expand(len(active_edges), -1)
        mean_drop = _mean_over_decisions(probability_drop, row_mask)
        mean_abs_drop = _mean_over_decisions(probability_drop.abs(), row_mask)
        mean_js = _mean_over_decisions(js, row_mask)
        value_drop = base.value.expand_as(interventions.value) - interventions.value
        base_value = base.value
        selected_agent_actions = selected

    rows = []
    for row, edge in enumerate(active_edges):
        rows.append(
            {
                **_edge_semantics(
                    edge,
                    incidence,
                    obs_numpy,
                    policy.env_cfg.max_agents,
                    policy.env_cfg.max_tasks,
                ),
                "selected_probability_drop": float(mean_drop[row]),
                "absolute_probability_change": float(mean_abs_drop[row]),
                "policy_js_divergence": float(mean_js[row]),
                "critic_value_drop": float(value_drop[row]),
            }
        )

    task_edges = [edge for edge in active_edges if edge < policy.env_cfg.max_tasks]
    zone_edges = [edge for edge in active_edges if edge >= policy.env_cfg.max_tasks]
    grouped = {}
    for label, edges in (("all_task_coalitions", task_edges), ("all_shared_zones", zone_edges)):
        if not edges:
            continue
        output = _masked_outputs(policy, obs, incidence, [edges])
        if getattr(policy, "uses_coalition_reservation", False):
            scores, stop, _, grouped_value = output
            probability = _conditional_distributions(
                scores, stop, trace["conditional_masks"]
            )
            after = probability.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
            critic_drop = float((base_value - grouped_value)[0])
        else:
            probability = torch.softmax(output.logits, dim=-1)
            after = probability.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
            critic_drop = float((base_value - output.value)[0])
        grouped[label] = {
            "edge_count": len(edges),
            "selected_probability_drop": float(
                _mean_over_decisions(selected_prob - after, decision_mask)[0]
            ),
            "critic_value_drop": critic_drop,
        }

    active_agents = torch.nonzero(obs["agent_mask"][0], as_tuple=False).flatten()
    return {
        "active_agent_count": int(active_agents.numel()),
        "active_task_count": int(obs["task_mask"][0].sum()),
        "base_value": float(base_value[0]),
        "selected_actions": {
            str(int(slot)): int(selected_agent_actions[0, slot])
            for slot in active_agents.tolist()
        },
        "edge_attributions": sorted(
            rows, key=lambda item: item["absolute_probability_change"], reverse=True
        ),
        "group_attributions": grouped,
    }


def _choose_action(policy, obs_numpy, device) -> np.ndarray:
    obs = to_torch({key: value[None] for key, value in obs_numpy.items()}, device)
    with torch.no_grad():
        actions = policy.act(obs, deterministic=True)[0]
    return actions[0].cpu().numpy()


def capture_breakdown_episodes(
    policy,
    config: ExperimentConfig,
    device: torch.device,
    episodes: int,
    seed_offset: int = 4_000_000,
) -> list[dict[str, Any]]:
    """Capture fixed event-relative states and select no episode by performance."""

    scenario_cfg = make_scenario(config.env, EXPLANATION_SCENARIO)
    captured = []
    for episode in range(episodes):
        seed = seed_offset + episode
        env = OpenWarehouseEnv(scenario_cfg, seed=seed)
        obs = env.reset(seed=seed)
        snapshots: dict[str, dict[str, np.ndarray]] = {}
        scene_states: dict[str, dict[str, Any]] = {}
        timeline: list[dict[str, Any]] = []

        def capture(event: str) -> None:
            snapshots[event] = {k: v.copy() for k, v in obs.items()}
            scene_states[event] = {
                "time": env.time,
                "width": scenario_cfg.width,
                "height": scenario_cfg.height,
                "agents": [{"slot": i, **asdict(a)} for i, a in enumerate(env.agents)],
                "tasks": [
                    {"slot": i, **asdict(task)}
                    for i, task in enumerate(env.tasks) if task.active
                ],
                "metrics": dict(env.episode_metrics),
            }
        done = False
        info: dict[str, Any] = {}
        while not done:
            if env.time == scenario_cfg.scheduled_breakdown_step:
                capture("pre_breakdown")
            actions = _choose_action(policy, obs, device)
            obs, _, done, info = env.step(actions)
            timeline.append({
                "time": env.time,
                **{key: float(info[key]) for key in (
                    "service_level", "deliveries", "deadline_miss_rate",
                    "active_agents", "active_tasks", "contentions",
                    "forced_failures", "repairs",
                )},
            })
            if env.time == scenario_cfg.scheduled_breakdown_step + 1:
                capture("post_breakdown")
            if env.time == scenario_cfg.scheduled_repair_step + 1:
                capture("post_repair")
        missing = set(EXPLANATION_EVENTS) - set(snapshots)
        if missing:
            raise RuntimeError(f"breakdown episode missing snapshots: {sorted(missing)}")
        captured.append(
            {
                "episode": episode,
                "seed": seed,
                "service_level": float(info["service_level"]),
                "throughput_per_100_steps": float(info["throughput_per_100_steps"]),
                "snapshots": snapshots,
                "scene_states": scene_states,
                "timeline": timeline,
            }
        )
    return captured


def _median_episode_index(episodes: list[dict[str, Any]]) -> int:
    values = np.asarray([episode["service_level"] for episode in episodes])
    median = float(np.median(values))
    return min(
        range(len(episodes)),
        key=lambda index: (abs(episodes[index]["service_level"] - median), episodes[index]["seed"]),
    )


def _event_summary(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate one value per execution episode before summarizing.

    Hyperedges within a state are not treated as independent observations.
    """

    rows: list[dict[str, Any]] = []
    for event in EXPLANATION_EVENTS:
        reports = [episode["events"][event] for episode in detailed]
        for edge_type in ("task_coalition", "shared_zone"):
            episode_means, episode_maxima = [], []
            for report in reports:
                values = [
                    edge["absolute_probability_change"]
                    for edge in report["edge_attributions"]
                    if edge["edge_type"] == edge_type
                ]
                if values:
                    episode_means.append(float(np.mean(values)))
                    episode_maxima.append(float(np.max(values)))
            if episode_means:
                rows.append(
                    {
                        "event": event,
                        "attribution_scope": edge_type,
                        "n_execution_episodes": len(episode_means),
                        "mean_edge_abs_probability_change": float(np.mean(episode_means)),
                        "std_edge_abs_probability_change": (
                            float(np.std(episode_means, ddof=1))
                            if len(episode_means) > 1
                            else 0.0
                        ),
                        "mean_top_edge_abs_probability_change": float(np.mean(episode_maxima)),
                        "mean_abs_group_probability_drop": None,
                        "mean_abs_group_value_drop": None,
                        "mean_active_agents": float(
                            np.mean([report["active_agent_count"] for report in reports])
                        ),
                    }
                )
        for group in ("all_task_coalitions", "all_shared_zones"):
            available = [report["group_attributions"][group] for report in reports if group in report["group_attributions"]]
            if available:
                rows.append(
                    {
                        "event": event,
                        "attribution_scope": group,
                        "n_execution_episodes": len(available),
                        "mean_edge_abs_probability_change": None,
                        "std_edge_abs_probability_change": None,
                        "mean_top_edge_abs_probability_change": None,
                        "mean_abs_group_probability_drop": float(
                            np.mean([abs(item["selected_probability_drop"]) for item in available])
                        ),
                        "mean_abs_group_value_drop": float(
                            np.mean([abs(item["critic_value_drop"]) for item in available])
                        ),
                        "mean_active_agents": float(
                            np.mean([report["active_agent_count"] for report in reports])
                        ),
                    }
                )
    return rows


def explain_checkpoint(
    checkpoint: str | Path,
    episodes: int = 20,
    device: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy, config, metadata, resolved = load_policy(checkpoint, device)
    if not isinstance(policy, OpenHyperPolicy) or metadata["algorithm"] not in {
        "chord_mappo",
    }:
        raise ValueError("semantic attribution requires a CHORD-CAHR checkpoint")
    captured = capture_breakdown_episodes(policy, config, resolved, episodes)
    chosen_index = _median_episode_index(captured)
    raw_rows: list[dict[str, Any]] = []
    detailed = []
    for episode_index, episode in enumerate(captured):
        event_reports = {}
        for event in EXPLANATION_EVENTS:
            attribution = semantic_edge_attribution(policy, episode["snapshots"][event], resolved)
            scene = episode["scene_states"][event]
            order_ids = {task["slot"]: task["order_id"] for task in scene["tasks"]}
            for edge in attribution["edge_attributions"]:
                edge["order_ids"] = [order_ids[slot] for slot in edge["task_slots"]]
            attribution["scene"] = scene
            attribution["selected_assignments"] = [
                {"agent_slot": int(slot), "task_slot": int(action) - 1,
                 "order_id": order_ids.get(int(action) - 1)}
                for slot, action in attribution["selected_actions"].items() if action > 0
            ]
            event_reports[event] = attribution
            for rank, edge in enumerate(attribution["edge_attributions"], start=1):
                raw_rows.append(
                    {
                        "training_seed": metadata["seed"],
                        "episode": episode["episode"],
                        "execution_seed": episode["seed"],
                        "is_median_case": episode_index == chosen_index,
                        "event": event,
                        "rank": rank,
                        **{
                            key: json.dumps(value) if isinstance(value, list) else value
                            for key, value in edge.items()
                        },
                    }
                )
        detailed.append(
            {
                "episode": episode["episode"],
                "execution_seed": episode["seed"],
                "service_level": episode["service_level"],
                "throughput_per_100_steps": episode["throughput_per_100_steps"],
                "is_median_case": episode_index == chosen_index,
                "events": event_reports,
                "timeline": episode["timeline"],
            }
        )
    summary = {
        "algorithm": "chord_mappo",
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_seed": metadata["seed"],
        "learner_revision": metadata.get("learner_revision"),
        "action_decoder": metadata.get("action_decoder"),
        "scenario": EXPLANATION_SCENARIO,
        "selection_rule": "service-level median episode; execution seed breaks ties",
        "candidate_episode_count": episodes,
        "scenario_config": asdict(make_scenario(config.env, EXPLANATION_SCENARIO)),
        "event_boundary": (
            "Snapshots straddle complete environment steps, not isolated causal interventions. "
            "Background arrivals/joins remain enabled. Repair restores available AGV capacity; "
            "the environment does not guarantee reactivation of the same failed identity."
        ),
        "median_case_execution_seed": captured[chosen_index]["seed"],
        "claim_boundary": (
            "model-internal semantic hyperedge attribution; not a causal claim about "
            "physical warehouse dynamics"
        ),
        "event_summary": _event_summary(detailed),
        "episodes": detailed,
    }
    return summary, raw_rows


def write_explanation(
    summaries: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: str | Path
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "semantic_attribution.json").write_text(
        json.dumps({"checkpoints": summaries}, indent=2), encoding="utf-8"
    )
    if rows:
        with (target / "edge_attributions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    event_rows = [
        {"training_seed": summary["training_seed"], **row}
        for summary in summaries
        for row in summary["event_summary"]
    ]
    if event_rows:
        with (target / "semantic_event_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)
    lines = [
        "# CHORD-CAHR semantic coalition case study",
        "",
        "Scenario: AGV breakdown and service-capacity restoration. Cases use a fixed median-service-level selection rule.",
        "Attributions are model-internal edge-masking effects, not physical causal effects.",
        "Task slots may be reused: compare persistent order_ids across snapshots, not slot numbers alone.",
        "Repair restores capacity and need not reactivate the same failed robot. Background events remain enabled.",
        "",
    ]
    for summary in summaries:
        median = next(item for item in summary["episodes"] if item["is_median_case"])
        lines.extend(
            [
                f"## Training seed {summary['training_seed']}",
                "",
                f"Execution seed: {median['execution_seed']}; service level: {median['service_level']:.4f}.",
                "",
            ]
        )
        for event in EXPLANATION_EVENTS:
            top = median["events"][event]["edge_attributions"][:3]
            labels = ", ".join(
                f"{item['edge_type']}:{item['semantic_id']} orders={item.get('order_ids', [])} "
                f"(|Δp|={item['absolute_probability_change']:.4f}, JS={item['policy_js_divergence']:.4f})"
                for item in top
            )
            lines.append(f"- {event}: {labels}")
        lines.extend(["", "Across the fixed candidate episodes:", ""])
        for event in EXPLANATION_EVENTS:
            event_rows_for_seed = [
                row
                for row in summary["event_summary"]
                if row["event"] == event
                and row["attribution_scope"] in {"task_coalition", "shared_zone"}
            ]
            values = ", ".join(
                f"{row['attribution_scope']} mean |Δp|={row['mean_edge_abs_probability_change']:.4f}"
                for row in event_rows_for_seed
            )
            lines.append(f"- {event}: {values}")
        lines.append("")
    (target / "semantic_case_study.md").write_text("\n".join(lines), encoding="utf-8")
    timeline_rows = [
        {"training_seed": summary["training_seed"],
         "execution_seed": episode["execution_seed"],
         "is_median_case": episode["is_median_case"], **point}
        for summary in summaries for episode in summary["episodes"]
        for point in episode.get("timeline", [])
    ]
    if timeline_rows:
        with (target / "case_timelines.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timeline_rows[0]))
            writer.writeheader()
            writer.writerows(timeline_rows)
