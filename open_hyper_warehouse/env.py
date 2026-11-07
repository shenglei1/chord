from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from .config import EnvConfig


AGENT_FEATURE_DIM = 12
TASK_FEATURE_DIM = 10


class AgentKind(IntEnum):
    AGV = 0
    PICKER = 1


class TaskStage(IntEnum):
    WAITING = 0
    DELIVERY = 1


@dataclass(slots=True)
class AgentState:
    kind: AgentKind
    active: bool = False
    x: int = 0
    y: int = 0
    assigned_task: int = -1
    carrying: bool = False
    joined_at: int = 0

    @property
    def busy(self) -> bool:
        return self.assigned_task >= 0


@dataclass(slots=True)
class TaskState:
    active: bool = False
    x: int = 0
    y: int = 0
    goal_x: int = 0
    goal_y: int = 0
    priority: float = 1.0
    deadline: int = 0
    created_at: int = 0
    stage: TaskStage = TaskStage.WAITING
    agv_slot: int = -1
    picker_slot: int = -1
    service_remaining: int = -1
    order_id: int = -1


class OpenWarehouseEnv:
    """Synchronous, open-population warehouse task-assignment environment.

    The simulator keeps fixed *capacity* arrays for batching, but entity masks
    and stable type-shared semantics define the actual random finite sets. An
    algorithm must never treat inactive padding slots as warehouse entities.
    """

    def __init__(self, config: EnvConfig, seed: int = 0):
        self.cfg = config
        self.cfg.validate()
        self.seed_value = seed
        self.next_order_id = 0
        self.agents = [AgentState(AgentKind.AGV) for _ in range(config.max_agvs)]
        self.agents += [AgentState(AgentKind.PICKER) for _ in range(config.max_pickers)]
        self.tasks = [TaskState() for _ in range(config.max_tasks)]
        self.time = 0
        self.episode_metrics: dict[str, float] = {}

    @property
    def max_agents(self) -> int:
        return self.cfg.max_agents

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed_value = seed
        self.time = 0
        self.next_order_id = 0
        self.agents = [AgentState(AgentKind.AGV) for _ in range(self.cfg.max_agvs)]
        self.agents += [AgentState(AgentKind.PICKER) for _ in range(self.cfg.max_pickers)]
        self.tasks = [TaskState() for _ in range(self.cfg.max_tasks)]
        self.episode_metrics = {
            "deliveries": 0.0,
            "pickups": 0.0,
            "deadline_misses": 0.0,
            "cancellations": 0.0,
            "contentions": 0.0,
            "travel": 0.0,
            "agent_joins": 0.0,
            "agent_leaves": 0.0,
            "active_agent_steps": 0.0,
            "busy_agent_steps": 0.0,
            "orders_created": 0.0,
            "orders_offered": 0.0,
            "task_flow_time_sum": 0.0,
            "max_backlog": 0.0,
            "forced_failures": 0.0,
            "repairs": 0.0,
            "return": 0.0,
        }
        initial_rng = self._event_rng(1, 0)
        nagv = int(initial_rng.integers(self.cfg.initial_agvs_min, self.cfg.initial_agvs_max + 1))
        npicker = int(initial_rng.integers(self.cfg.initial_pickers_min, self.cfg.initial_pickers_max + 1))
        for slot in range(nagv):
            self._activate_agent(slot, count_join=False)
        for offset in range(npicker):
            self._activate_agent(self.cfg.max_agvs + offset, count_join=False)
        ntasks = int(initial_rng.integers(self.cfg.initial_tasks_min, self.cfg.initial_tasks_max + 1))
        for _ in range(ntasks):
            self._spawn_task()
        return self.observe()

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.max_agents,):
            raise ValueError(f"actions must have shape {(self.max_agents,)}, got {actions.shape}")
        action_mask = self._action_mask()
        invalid = [i for i, a in enumerate(actions) if a < 0 or a > self.cfg.max_tasks or not action_mask[i, a]]
        if invalid:
            raise ValueError(f"invalid actions for slots {invalid[:8]}")

        reward = -self.cfg.step_penalty * sum(a.active for a in self.agents)
        reward += self._accept_assignments(actions)
        reward += self._advance_agents_and_tasks()
        reward += self._expire_and_cancel_tasks()
        self._apply_open_events()
        self.time += 1

        active = sum(a.active for a in self.agents)
        busy = sum(a.active and a.busy for a in self.agents)
        self.episode_metrics["active_agent_steps"] += active
        self.episode_metrics["busy_agent_steps"] += busy
        self.episode_metrics["max_backlog"] = max(
            self.episode_metrics["max_backlog"],
            float(sum(task.active for task in self.tasks)),
        )
        self.episode_metrics["return"] += reward
        done = self.time >= self.cfg.episode_steps
        info = dict(self.episode_metrics)
        eligible_orders = max(
            self.episode_metrics["orders_created"] - self.episode_metrics["cancellations"],
            1.0,
        )
        info.update(
            active_agents=float(active),
            active_tasks=float(sum(t.active for t in self.tasks)),
            utilization=float(
                self.episode_metrics["busy_agent_steps"]
                / max(self.episode_metrics["active_agent_steps"], 1.0)
            ),
            completion_ratio=float(
                self.episode_metrics["deliveries"] / max(self.episode_metrics["orders_created"], 1.0)
            ),
            service_level=float(self.episode_metrics["deliveries"] / eligible_orders),
            deadline_miss_rate=float(self.episode_metrics["deadline_misses"] / eligible_orders),
            throughput_per_100_steps=float(
                100.0 * self.episode_metrics["deliveries"] / max(self.time + 1, 1)
            ),
            mean_active_agents=float(
                self.episode_metrics["active_agent_steps"] / max(self.time + 1, 1)
            ),
            contention_per_delivery=float(
                self.episode_metrics["contentions"]
                / max(self.episode_metrics["deliveries"], 1.0)
            ),
            unserved_orders=float(sum(t.active for t in self.tasks)),
            mean_flow_time=float(
                self.episode_metrics["task_flow_time_sum"]
                / max(self.episode_metrics["deliveries"], 1.0)
            ),
            travel_per_delivery=float(
                self.episode_metrics["travel"]
                / max(self.episode_metrics["deliveries"], 1.0)
            ),
        )
        return self.observe(), float(reward), done, info

    def observe(self) -> dict[str, np.ndarray]:
        agent_features = np.zeros((self.max_agents, AGENT_FEATURE_DIM), dtype=np.float32)
        agent_mask = np.zeros(self.max_agents, dtype=np.bool_)
        for i, agent in enumerate(self.agents):
            if not agent.active:
                continue
            agent_mask[i] = True
            assigned_norm = 0.0 if agent.assigned_task < 0 else (agent.assigned_task + 1) / self.cfg.max_tasks
            goal_dist = 0.0
            if agent.assigned_task >= 0 and self.tasks[agent.assigned_task].active:
                task = self.tasks[agent.assigned_task]
                tx, ty = (task.goal_x, task.goal_y) if task.stage == TaskStage.DELIVERY else (task.x, task.y)
                goal_dist = self._distance(agent.x, agent.y, tx, ty) / (self.cfg.width + self.cfg.height)
            age = (self.time - agent.joined_at) / max(self.cfg.episode_steps, 1)
            agent_features[i] = np.array(
                [
                    float(agent.kind == AgentKind.AGV),
                    float(agent.kind == AgentKind.PICKER),
                    agent.x / max(self.cfg.width - 1, 1),
                    agent.y / max(self.cfg.height - 1, 1),
                    1.0,
                    float(agent.busy),
                    float(agent.carrying),
                    assigned_norm,
                    goal_dist,
                    min(age, 1.0),
                    self.time / max(self.cfg.episode_steps, 1),
                    1.0,
                ],
                dtype=np.float32,
            )

        task_features = np.zeros((self.cfg.max_tasks, TASK_FEATURE_DIM), dtype=np.float32)
        task_mask = np.zeros(self.cfg.max_tasks, dtype=np.bool_)
        for i, task in enumerate(self.tasks):
            if not task.active:
                continue
            task_mask[i] = True
            slack = (task.deadline - self.time) / max(self.cfg.deadline_max, 1)
            age = (self.time - task.created_at) / max(self.cfg.deadline_max, 1)
            task_features[i] = np.array(
                [
                    task.x / max(self.cfg.width - 1, 1),
                    task.y / max(self.cfg.height - 1, 1),
                    task.goal_x / max(self.cfg.width - 1, 1),
                    task.goal_y / max(self.cfg.height - 1, 1),
                    task.priority / 2.0,
                    np.clip(slack, -1.0, 1.0),
                    min(age, 1.0),
                    float(task.stage == TaskStage.WAITING),
                    float(task.stage == TaskStage.DELIVERY),
                    1.0,
                ],
                dtype=np.float32,
            )
        return {
            "agent_features": agent_features,
            "task_features": task_features,
            "agent_mask": agent_mask,
            "task_mask": task_mask,
            "action_mask": self._action_mask(),
        }

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros((self.max_agents, self.cfg.max_tasks + 1), dtype=np.bool_)
        mask[:, 0] = True
        for i, agent in enumerate(self.agents):
            if not agent.active or agent.busy:
                continue
            for j, task in enumerate(self.tasks):
                if not task.active or task.stage != TaskStage.WAITING:
                    continue
                if agent.kind == AgentKind.AGV and task.agv_slot < 0:
                    mask[i, j + 1] = True
                if agent.kind == AgentKind.PICKER and task.picker_slot < 0:
                    mask[i, j + 1] = True
        return mask

    def _accept_assignments(self, actions: np.ndarray) -> float:
        reward = 0.0
        proposals: dict[tuple[int, AgentKind], list[int]] = {}
        for slot, raw_action in enumerate(actions):
            if raw_action == 0:
                continue
            task_slot = int(raw_action - 1)
            proposals.setdefault((task_slot, self.agents[slot].kind), []).append(slot)
        for (task_slot, kind), candidates in proposals.items():
            task = self.tasks[task_slot]
            ranked = sorted(
                candidates,
                key=lambda s: (self._distance(self.agents[s].x, self.agents[s].y, task.x, task.y), s),
            )
            winner = ranked[0]
            if len(ranked) > 1:
                conflicts = len(ranked) - 1
                self.episode_metrics["contentions"] += conflicts
                reward -= self.cfg.contention_penalty * conflicts
            agent = self.agents[winner]
            agent.assigned_task = task_slot
            if kind == AgentKind.AGV:
                task.agv_slot = winner
            else:
                task.picker_slot = winner
        return reward

    def _advance_agents_and_tasks(self) -> float:
        reward = 0.0
        for agent in self.agents:
            if not agent.active or not agent.busy:
                continue
            task = self.tasks[agent.assigned_task]
            if not task.active:
                self._release_agent(agent)
                continue
            target = (task.goal_x, task.goal_y) if task.stage == TaskStage.DELIVERY else (task.x, task.y)
            moved = self._move_one(agent, *target)
            self.episode_metrics["travel"] += float(moved)

        for task_slot, task in enumerate(self.tasks):
            if not task.active:
                continue
            if task.stage == TaskStage.WAITING and task.agv_slot >= 0 and task.picker_slot >= 0:
                agv, picker = self.agents[task.agv_slot], self.agents[task.picker_slot]
                if (agv.x, agv.y) == (task.x, task.y) and (picker.x, picker.y) == (task.x, task.y):
                    if task.service_remaining < 0:
                        task.service_remaining = int(
                            self._event_rng(401, task.order_id).integers(
                                self.cfg.picker_service_delay_min,
                                self.cfg.picker_service_delay_max + 1,
                            )
                        )
                    if task.service_remaining > 0:
                        task.service_remaining -= 1
                    else:
                        task.stage = TaskStage.DELIVERY
                        agv.carrying = True
                        self._release_agent(picker)
                        task.picker_slot = -1
                        reward += self.cfg.pickup_reward * task.priority
                        self.episode_metrics["pickups"] += 1
            if task.stage == TaskStage.DELIVERY and task.agv_slot >= 0:
                agv = self.agents[task.agv_slot]
                if (agv.x, agv.y) == (task.goal_x, task.goal_y):
                    reward += self.cfg.delivery_reward * task.priority
                    self.episode_metrics["deliveries"] += 1
                    self.episode_metrics["task_flow_time_sum"] += self.time - task.created_at + 1
                    self._release_agent(agv)
                    self.tasks[task_slot] = TaskState()
        return reward

    def _expire_and_cancel_tasks(self) -> float:
        reward = 0.0
        for slot, task in enumerate(self.tasks):
            if not task.active:
                continue
            expired = self.time >= task.deadline
            cancel_draw = self._event_rng(301, self.time, task.order_id).random()
            cancelled = task.stage == TaskStage.WAITING and cancel_draw < self.cfg.order_cancel_probability
            if expired or cancelled:
                if expired:
                    reward -= self.cfg.deadline_penalty * task.priority
                    self.episode_metrics["deadline_misses"] += 1
                else:
                    self.episode_metrics["cancellations"] += 1
                self._release_task_agents(task)
                self.tasks[slot] = TaskState()
        return reward

    def _apply_open_events(self) -> None:
        rate = self._current_arrival_rate()
        arrivals = int(self._event_rng(101, self.time).poisson(rate))
        for _ in range(arrivals):
            self._spawn_task()
        if self.time == self.cfg.scheduled_breakdown_step:
            self._apply_scheduled_breakdown()
        if self.time == self.cfg.scheduled_repair_step:
            self._apply_scheduled_repair()
        availability_rng = self._event_rng(501, self.time)
        join_event, leave_event, join_selector, leave_selector = availability_rng.random(4)
        if join_event < self.cfg.agent_join_probability:
            self._join_random_agent(join_selector)
        if leave_event < self.cfg.agent_leave_probability:
            self._leave_random_idle_agent(leave_selector)

    def _current_arrival_rate(self) -> float:
        phase = self.time / max(self.cfg.episode_steps - 1, 1)
        if self.cfg.arrival_profile == "mid_shift_peak":
            return self.cfg.order_arrival_rate * (3.5 if 0.35 <= phase <= 0.65 else 0.5)
        if self.cfg.arrival_profile == "two_waves":
            in_wave = 0.15 <= phase <= 0.30 or 0.65 <= phase <= 0.80
            return self.cfg.order_arrival_rate * (3.0 if in_wave else 0.35)
        return self.cfg.order_arrival_rate

    def _apply_scheduled_breakdown(self) -> None:
        candidates = [
            i for i in range(self.cfg.max_agvs) if self.agents[i].active
        ]
        # Busy robots fail first to test recovery, with slot id as deterministic tie-break.
        candidates.sort(key=lambda i: (not self.agents[i].busy, i))
        active = len(candidates)
        count = min(self.cfg.scheduled_breakdown_count, max(active - 1, 0))
        for slot in candidates[:count]:
            self._deactivate_agent(slot, forced=True)

    def _apply_scheduled_repair(self) -> None:
        inactive = [i for i in range(self.cfg.max_agvs) if not self.agents[i].active]
        for slot in inactive[: self.cfg.scheduled_repair_count]:
            self._activate_agent(slot, count_join=True)
            self.episode_metrics["repairs"] += 1

    def _spawn_task(self) -> bool:
        order_id = self.next_order_id
        self.next_order_id += 1
        self.episode_metrics["orders_offered"] += 1
        task_rng = self._event_rng(201, order_id)
        hotspot_draw = task_rng.random()
        hotspot_dx = int(task_rng.integers(-1, 2))
        hotspot_dy = int(task_rng.integers(-1, 2))
        regular_x = int(task_rng.integers(1, max(self.cfg.width - 1, 2)))
        regular_y = int(task_rng.integers(1, max(self.cfg.height - 2, 2)))
        goal_x = int(task_rng.integers(0, self.cfg.width))
        priority = float(task_rng.choice([1.0, 1.5, 2.0], p=[0.70, 0.20, 0.10]))
        deadline = self.time + int(task_rng.integers(self.cfg.deadline_min, self.cfg.deadline_max + 1))
        free = [i for i, task in enumerate(self.tasks) if not task.active]
        if not free:
            return False
        slot = free[0]
        if hotspot_draw < self.cfg.task_hotspot_probability:
            center_x = self.cfg.width // 2
            center_y = self.cfg.height // 2
            x = int(np.clip(center_x + hotspot_dx, 1, self.cfg.width - 2))
            y = int(np.clip(center_y + hotspot_dy, 1, self.cfg.height - 3))
        else:
            x = regular_x
            y = regular_y
        goal_y = self.cfg.height - 1
        self.tasks[slot] = TaskState(
            active=True,
            x=x,
            y=y,
            goal_x=goal_x,
            goal_y=goal_y,
            priority=priority,
            deadline=deadline,
            created_at=self.time,
            order_id=order_id,
        )
        self.episode_metrics["orders_created"] += 1
        return True

    def _join_random_agent(self, selector: float) -> bool:
        free_agvs = [i for i in range(self.cfg.max_agvs) if not self.agents[i].active]
        free_pickers = [
            i for i in range(self.cfg.max_agvs, self.max_agents) if not self.agents[i].active
        ]
        choices = []
        if free_agvs:
            choices.append(free_agvs[0])
        if free_pickers:
            choices.append(free_pickers[0])
        if not choices:
            return False
        index = min(int(selector * len(choices)), len(choices) - 1)
        return self._activate_agent(choices[index], count_join=True)

    def _leave_random_idle_agent(self, selector: float) -> bool:
        candidates = [i for i, a in enumerate(self.agents) if a.active and not a.busy]
        active_agvs = sum(a.active and a.kind == AgentKind.AGV for a in self.agents)
        active_pickers = sum(a.active and a.kind == AgentKind.PICKER for a in self.agents)
        candidates = [
            i
            for i in candidates
            if not (
                (self.agents[i].kind == AgentKind.AGV and active_agvs <= 1)
                or (self.agents[i].kind == AgentKind.PICKER and active_pickers <= 1)
            )
        ]
        if not candidates:
            return False
        index = min(int(selector * len(candidates)), len(candidates) - 1)
        slot = candidates[index]
        self._deactivate_agent(slot, forced=False)
        return True

    def _activate_agent(self, slot: int, count_join: bool) -> bool:
        agent = self.agents[slot]
        agent.active = True
        position_rng = self._event_rng(601, self.time, slot)
        if self.cfg.agent_spawn_mode == "left_depot":
            agent.x = int(position_rng.integers(0, max(1, self.cfg.width // 4)))
        elif self.cfg.agent_spawn_mode == "split_depot":
            left = slot % 2 == 0
            lo = 0 if left else max(0, 3 * self.cfg.width // 4)
            hi = max(lo + 1, self.cfg.width // 4 if left else self.cfg.width)
            agent.x = int(position_rng.integers(lo, hi))
        else:
            agent.x = int(position_rng.integers(0, self.cfg.width))
        agent.y = self.cfg.height - 1
        agent.assigned_task = -1
        agent.carrying = False
        agent.joined_at = self.time
        if count_join:
            self.episode_metrics["agent_joins"] += 1
        return True

    def _deactivate_agent(self, slot: int, forced: bool) -> None:
        agent = self.agents[slot]
        if not agent.active:
            return
        if agent.assigned_task >= 0 and self.tasks[agent.assigned_task].active:
            task = self.tasks[agent.assigned_task]
            if agent.kind == AgentKind.AGV:
                task.agv_slot = -1
                if task.stage == TaskStage.DELIVERY:
                    task.stage = TaskStage.WAITING
                    task.service_remaining = -1
            else:
                task.picker_slot = -1
                task.service_remaining = -1
        self._release_agent(agent)
        agent.active = False
        self.episode_metrics["agent_leaves"] += 1
        if forced:
            self.episode_metrics["forced_failures"] += 1

    def _release_task_agents(self, task: TaskState) -> None:
        for slot in (task.agv_slot, task.picker_slot):
            if slot >= 0:
                self._release_agent(self.agents[slot])

    @staticmethod
    def _release_agent(agent: AgentState) -> None:
        agent.assigned_task = -1
        agent.carrying = False

    @staticmethod
    def _distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _event_rng(self, stream: int, *indices: int) -> np.random.Generator:
        """Policy-independent random stream keyed by event type and identity."""
        entropy = [int(self.seed_value), int(stream), *(int(index) for index in indices)]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    @staticmethod
    def _move_one(agent: AgentState, tx: int, ty: int) -> bool:
        old = (agent.x, agent.y)
        if agent.x < tx:
            agent.x += 1
        elif agent.x > tx:
            agent.x -= 1
        elif agent.y < ty:
            agent.y += 1
        elif agent.y > ty:
            agent.y -= 1
        return old != (agent.x, agent.y)
