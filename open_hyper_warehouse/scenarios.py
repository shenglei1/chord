from __future__ import annotations

from dataclasses import replace

from .config import EnvConfig


SCENARIOS = (
    "in_distribution",
    "fixed_team_dynamic_orders",
    "unseen_small",
    "unseen_large",
    "ratio_shift",
    "agv_heavy_ratio",
    "picker_heavy_ratio",
    "burst_orders",
    "high_churn",
    "hotspot",
    "robot_breakdown",
    "picker_shift",
    "peak_demand",
    "tight_deadlines",
    "slow_pickers",
    "high_cancellation",
    "spatial_skew",
    "large_layout",
    "combined_disruption",
    "demand_050",
    "demand_100",
    "demand_200",
    "demand_350",
    "breakdown_000",
    "breakdown_010",
    "breakdown_025",
    "breakdown_040",
    "picker_delay_000",
    "picker_delay_002",
    "picker_delay_004",
    "picker_delay_006",
    "cancellation_000",
    "cancellation_010",
    "cancellation_040",
    "cancellation_080",
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
    if name == "ratio_shift":
        return replace(
            base,
            initial_agvs_min=max(1, int(base.max_agvs * 0.8)),
            initial_agvs_max=max(1, int(base.max_agvs * 0.8)),
            initial_pickers_min=min(2, base.max_pickers),
            initial_pickers_max=min(2, base.max_pickers),
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
    if name == "picker_heavy_ratio":
        agvs = min(base.max_agvs, max(3, base.initial_agvs_min - 2))
        pickers = min(
            base.max_pickers,
            max(base.initial_pickers_max + 2, int(round(base.max_pickers * 0.75))),
        )
        return replace(
            base,
            initial_agvs_min=agvs,
            initial_agvs_max=agvs,
            initial_pickers_min=pickers,
            initial_pickers_max=pickers,
            agent_join_probability=0.0,
            agent_leave_probability=0.0,
        )
    if name == "burst_orders":
        return replace(
            base,
            order_arrival_rate=max(base.order_arrival_rate * 3.0, 0.5),
            initial_tasks_min=max(base.initial_tasks_min, min(base.max_tasks, base.initial_tasks_max)),
            initial_tasks_max=min(base.max_tasks, max(base.initial_tasks_max, int(base.max_tasks * 0.75))),
        )
    if name == "high_churn":
        return replace(base, agent_join_probability=0.03, agent_leave_probability=0.02)
    if name == "hotspot":
        return replace(base, task_hotspot_probability=0.90)
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
    if name == "picker_shift":
        return replace(
            base,
            picker_shift_step=max(1, base.episode_steps // 2),
            picker_shift_leave_count=max(1, base.initial_pickers_min // 2),
            picker_shift_join_count=max(1, base.initial_pickers_min // 2),
            agent_leave_probability=0.0,
        )
    if name == "peak_demand":
        return replace(base, arrival_profile="mid_shift_peak")
    if name == "tight_deadlines":
        return replace(
            base,
            deadline_min=max(4, base.deadline_min // 2),
            deadline_max=max(8, base.deadline_max // 2),
        )
    if name == "slow_pickers":
        return replace(base, picker_service_delay_min=2, picker_service_delay_max=6)
    if name == "high_cancellation":
        return replace(base, order_cancel_probability=max(0.04, base.order_cancel_probability * 8))
    if name == "spatial_skew":
        return replace(base, agent_spawn_mode="left_depot", task_hotspot_probability=0.95)
    if name == "large_layout":
        return replace(base, width=max(base.width + 8, int(base.width * 1.5)), height=max(base.height + 6, int(base.height * 1.5)))
    if name == "combined_disruption":
        breakdown_step = max(1, base.episode_steps // 3)
        breakdown_count = max(1, base.max_agvs // 5)
        return replace(
            base,
            arrival_profile="two_waves",
            agent_spawn_mode="split_depot",
            task_hotspot_probability=0.80,
            agent_join_probability=max(0.06, base.agent_join_probability),
            agent_leave_probability=max(0.04, base.agent_leave_probability),
            scheduled_breakdown_step=breakdown_step,
            scheduled_breakdown_count=breakdown_count,
            scheduled_repair_step=max(breakdown_step + 1, 3 * base.episode_steps // 4),
            scheduled_repair_count=breakdown_count,
            picker_shift_step=max(1, base.episode_steps // 2),
            picker_shift_leave_count=max(1, base.initial_pickers_min // 3),
            picker_shift_join_count=max(1, base.initial_pickers_min // 3),
            picker_service_delay_min=1,
            picker_service_delay_max=4,
            deadline_min=max(6, int(base.deadline_min * 0.65)),
            deadline_max=max(12, int(base.deadline_max * 0.65)),
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
    if name.startswith("picker_delay_"):
        upper = {
            "picker_delay_000": 0,
            "picker_delay_002": 2,
            "picker_delay_004": 4,
            "picker_delay_006": 6,
        }.get(name)
        if upper is not None:
            return replace(
                base,
                picker_service_delay_min=0,
                picker_service_delay_max=upper,
                agent_join_probability=0.0,
                agent_leave_probability=0.0,
            )
    if name.startswith("cancellation_"):
        probability = {
            "cancellation_000": 0.0,
            "cancellation_010": 0.01,
            "cancellation_040": 0.04,
            "cancellation_080": 0.08,
        }.get(name)
        if probability is not None:
            return replace(
                base,
                order_cancel_probability=probability,
                agent_join_probability=0.0,
                agent_leave_probability=0.0,
            )
    raise KeyError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
