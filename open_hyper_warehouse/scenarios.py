from __future__ import annotations

from dataclasses import replace

from .config import EnvConfig


SCENARIOS = (
    "in_distribution",
    "fixed_team_dynamic_orders",
    "unseen_small",
    "unseen_large",
    "agv_heavy_ratio",
    "robot_breakdown",
    "demand_050",
    "demand_100",
    "demand_200",
    "demand_350",
    "breakdown_000",
    "breakdown_010",
    "breakdown_025",
    "breakdown_040",
)


def make_scenario(base: EnvConfig, name: str) -> EnvConfig:
    """Create deterministic evaluation regimes without changing capacity."""

    if name == "in_distribution":
        return replace(base)
    if name == "fixed_team_dynamic_orders":
        return replace(base, agent_join_probability=0.0, agent_leave_probability=0.0)
    if name == "unseen_small":
        return replace(
            base,
            initial_agvs_min=min(3, base.max_agvs),
            initial_agvs_max=min(3, base.max_agvs),
            initial_pickers_min=min(2, base.max_pickers),
            initial_pickers_max=min(2, base.max_pickers),
            initial_tasks_min=min(4, base.max_tasks),
            initial_tasks_max=min(6, base.max_tasks),
            agent_join_probability=0.0,
            agent_leave_probability=0.0,
        )
    if name == "unseen_large":
        # Moderate out-of-distribution scale for the EAAI main table. The
        # previous near-capacity 23--24/11--12/60--80 regime is retained by
        # the separate stress protocols rather than creating a floor effect.
        agv_min = min(
            base.max_agvs,
            max(base.initial_agvs_max + 2, int(round(base.initial_agvs_max * 1.25))),
        )
        agv_max = min(base.max_agvs, max(agv_min, int(round(base.initial_agvs_max * 1.40))))
        picker_min = min(
            base.max_pickers,
            max(base.initial_pickers_max + 1, int(round(base.initial_pickers_max * 1.25))),
        )
        picker_max = min(
            base.max_pickers,
            max(picker_min, int(round(base.initial_pickers_max * 1.40))),
        )
        task_min = min(
            base.max_tasks,
            max(base.initial_tasks_max + 4, int(round(base.initial_tasks_max * 1.50))),
        )
        task_max = min(
            base.max_tasks,
            max(task_min, int(round(base.initial_tasks_max * 2.00))),
        )
        return replace(
            base,
            initial_agvs_min=agv_min,
            initial_agvs_max=agv_max,
            initial_pickers_min=picker_min,
            initial_pickers_max=picker_max,
            initial_tasks_min=task_min,
            initial_tasks_max=task_max,
            agent_join_probability=0.0,
            agent_leave_probability=0.0,
        )
    if name == "agv_heavy_ratio":
        agvs = min(
            base.max_agvs,
            max(base.initial_agvs_max + 2, int(round(base.max_agvs * 0.70))),
        )
        pickers = min(base.max_pickers, max(2, base.initial_pickers_min - 1))
        return replace(
            base,
            initial_agvs_min=agvs,
            initial_agvs_max=agvs,
            initial_pickers_min=pickers,
            initial_pickers_max=pickers,
            agent_join_probability=0.0,
            agent_leave_probability=0.0,
        )
    if name == "robot_breakdown":
        breakdown_step = max(1, base.episode_steps // 3)
        expected_initial_agvs = 0.5 * (base.initial_agvs_min + base.initial_agvs_max)
        breakdown_count = max(1, int(round(0.20 * expected_initial_agvs)))
        return replace(
            base,
            scheduled_breakdown_step=breakdown_step,
            scheduled_breakdown_count=breakdown_count,
            scheduled_repair_step=max(breakdown_step + 1, 2 * base.episode_steps // 3),
            scheduled_repair_count=breakdown_count,
            agent_leave_probability=0.0,
        )
    if name.startswith("demand_"):
        multiplier = {
            "demand_050": 0.5,
            "demand_100": 1.0,
            "demand_200": 2.0,
            "demand_350": 3.5,
        }.get(name)
        if multiplier is not None:
            return replace(
                base,
                order_arrival_rate=min(1.0, base.order_arrival_rate * multiplier),
                agent_join_probability=0.0,
                agent_leave_probability=0.0,
            )
    if name.startswith("breakdown_"):
        fraction = {
            "breakdown_000": 0.0,
            "breakdown_010": 0.10,
            "breakdown_025": 0.25,
            "breakdown_040": 0.40,
        }.get(name)
        if fraction is not None:
            count = min(base.max_agvs, int(round(base.initial_agvs_max * fraction)))
            return replace(
                base,
                scheduled_breakdown_step=(-1 if count == 0 else max(1, base.episode_steps // 3)),
                scheduled_breakdown_count=count,
                scheduled_repair_step=(-1 if count == 0 else max(2, 2 * base.episode_steps // 3)),
                scheduled_repair_count=count,
                agent_join_probability=0.0,
                agent_leave_probability=0.0,
            )
    raise KeyError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
