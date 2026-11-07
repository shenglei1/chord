from __future__ import annotations

import numpy as np


def _priority_order(obs: dict[str, np.ndarray], rule: str) -> list[int]:
    tasks = obs["task_features"]
    active = np.flatnonzero(obs["task_mask"]).tolist()
    if rule == "fcfs":
        return sorted(active, key=lambda t: (-tasks[t, 6], t))
    if rule == "edd":
        return sorted(active, key=lambda t: (tasks[t, 5], -tasks[t, 4], t))
    if rule == "priority":
        return sorted(active, key=lambda t: (-tasks[t, 4], tasks[t, 5], t))
    raise KeyError(rule)


def greedy_feasible_actions(obs: dict[str, np.ndarray], rule: str) -> np.ndarray:
    """Dispatch tasks by a warehouse rule, then select the nearest free worker."""

    agents = obs["agent_features"]
    tasks = obs["task_features"]
    action_mask = obs["action_mask"]
    actions = np.zeros(agents.shape[0], dtype=np.int64)
    used: set[int] = set()
    priorities = _priority_order(obs, rule)
    for task in priorities:
        for role in (0, 1):
            candidates = [
                a
                for a in np.flatnonzero(obs["agent_mask"])
                if a not in used and agents[a, role] > 0.5 and action_mask[a, task + 1]
            ]
            if not candidates:
                continue
            winner = min(
                candidates,
                key=lambda a: (np.abs(agents[a, 2:4] - tasks[task, :2]).sum(), a),
            )
            actions[winner] = task + 1
            used.add(winner)
    return actions


def closest_feasible_actions(obs: dict[str, np.ndarray]) -> np.ndarray:
    return greedy_feasible_actions(obs, "priority")


def fcfs_actions(obs: dict[str, np.ndarray]) -> np.ndarray:
    return greedy_feasible_actions(obs, "fcfs")


def earliest_deadline_actions(obs: dict[str, np.ndarray]) -> np.ndarray:
    return greedy_feasible_actions(obs, "edd")


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost rectangular assignment; returns the column for every row."""
    n, m = cost.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n > m:
        raise ValueError("Hungarian implementation requires rows <= columns")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=np.bool_)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def joint_hungarian_actions(obs: dict[str, np.ndarray]) -> np.ndarray:
    """One-step joint task screening followed by exact bipartite assignments.

    It first selects tasks that can receive both roles using joint travel,
    priority and deadline slack, then solves one rectangular assignment for
    AGVs and one for pickers. Routing remains part of the simulator.
    """
    agents = obs["agent_features"]
    tasks = obs["task_features"]
    mask = obs["action_mask"]
    actions = np.zeros(agents.shape[0], dtype=np.int64)
    available_by_role: list[list[int]] = []
    for role in (0, 1):
        available_by_role.append(
            [
                a
                for a in np.flatnonzero(obs["agent_mask"])
                if agents[a, role] > 0.5 and mask[a, 1:].any()
            ]
        )
    agvs, pickers = available_by_role
    if not agvs or not pickers:
        return actions
    candidates = [
        t
        for t in np.flatnonzero(obs["task_mask"])
        if any(mask[a, t + 1] for a in agvs) and any(mask[p, t + 1] for p in pickers)
    ]
    if not candidates:
        return actions
    joint_scores = []
    for task in candidates:
        agv_distance = min(np.abs(agents[a, 2:4] - tasks[task, :2]).sum() for a in agvs)
        picker_distance = min(np.abs(agents[p, 2:4] - tasks[task, :2]).sum() for p in pickers)
        urgency = 1.0 - np.clip(tasks[task, 5], -1.0, 1.0)
        joint_scores.append((agv_distance + picker_distance - 0.4 * urgency - 0.2 * tasks[task, 4], task))
    coalition_count = min(len(agvs), len(pickers), len(candidates))
    selected = [task for _, task in sorted(joint_scores)[:coalition_count]]

    for workers in (agvs, pickers):
        n = len(workers)
        # One dummy column per worker guarantees a feasible rectangular solve.
        cost = np.full((n, len(selected) + n), 4.0, dtype=np.float64)
        for row, worker in enumerate(workers):
            for column, task in enumerate(selected):
                if mask[worker, task + 1]:
                    distance = np.abs(agents[worker, 2:4] - tasks[task, :2]).sum()
                    urgency = 1.0 - np.clip(tasks[task, 5], -1.0, 1.0)
                    cost[row, column] = distance - 0.25 * urgency - 0.1 * tasks[task, 4]
                else:
                    cost[row, column] = 1e6
        assignment = _hungarian(cost)
        for row, column in enumerate(assignment):
            if 0 <= column < len(selected) and cost[row, column] < 1e5:
                actions[workers[row]] = selected[column] + 1
    return actions


def rolling_horizon_actions(
    obs: dict[str, np.ndarray], horizon: int = 4, candidate_limit: int = 12
) -> np.ndarray:
    """Receding-horizon coalition scheduling solved as a binary MILP.

    The model jointly schedules one AGV and one picker for every selected order,
    enforces worker occupancy over future periods, penalizes deferral and
    deadline risk, executes only period zero, and resolves after the next event.
    Candidate screening is deterministic and reported because solving all
    64 orders at every simulator tick is not operationally meaningful.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("rolling_horizon requires scipy>=1.11") from exc

    agents = obs["agent_features"]
    tasks = obs["task_features"]
    mask = obs["action_mask"]
    active_agents = np.flatnonzero(obs["agent_mask"]).tolist()
    agvs = [a for a in active_agents if agents[a, 0] > 0.5 and mask[a, 1:].any()]
    pickers = [a for a in active_agents if agents[a, 1] > 0.5 and mask[a, 1:].any()]
    actions = np.zeros(agents.shape[0], dtype=np.int64)
    if not agvs or not pickers:
        return actions
    feasible_tasks = [
        t
        for t in np.flatnonzero(obs["task_mask"])
        if any(mask[a, t + 1] for a in agvs) and any(mask[p, t + 1] for p in pickers)
    ]
    if not feasible_tasks:
        return actions
    # EDD/priority screening is independent of the evaluated algorithm and
    # deterministic under ties, so every seed sees the same optimization set.
    feasible_tasks.sort(key=lambda t: (tasks[t, 5], -tasks[t, 4], -tasks[t, 6], t))
    jobs = feasible_tasks[:candidate_limit]

    index: dict[tuple[str, int, int, int], int] = {}
    cursor = 0
    for tau in range(horizon):
        for j, task in enumerate(jobs):
            index[("z", j, tau, 0)] = cursor
            cursor += 1
            for ai, _ in enumerate(agvs):
                index[("x", ai, j, tau)] = cursor
                cursor += 1
            for pi, _ in enumerate(pickers):
                index[("y", pi, j, tau)] = cursor
                cursor += 1
    nvars = cursor
    objective = np.zeros(nvars, dtype=np.float64)
    variable_upper = np.ones(nvars, dtype=np.float64)
    travel_a = np.zeros((len(agvs), len(jobs)), dtype=np.float64)
    travel_p = np.zeros((len(pickers), len(jobs)), dtype=np.float64)
    for ai, agent in enumerate(agvs):
        travel_a[ai] = [np.abs(agents[agent, 2:4] - tasks[t, :2]).sum() for t in jobs]
    for pi, agent in enumerate(pickers):
        travel_p[pi] = [np.abs(agents[agent, 2:4] - tasks[t, :2]).sum() for t in jobs]
    for tau in range(horizon):
        for j, task in enumerate(jobs):
            slack = float(tasks[task, 5])
            priority = float(tasks[task, 4])
            lateness = max(0.0, 0.22 * tau - slack)
            objective[index[("z", j, tau, 0)]] = 0.12 * tau + 1.5 * lateness - 0.35 * priority
            for ai, agent in enumerate(agvs):
                key = index[("x", ai, j, tau)]
                objective[key] = 0.5 * travel_a[ai, j]
                if not mask[agent, task + 1]:
                    variable_upper[key] = 0.0
            for pi, agent in enumerate(pickers):
                key = index[("y", pi, j, tau)]
                objective[key] = 0.5 * travel_p[pi, j]
                if not mask[agent, task + 1]:
                    variable_upper[key] = 0.0

    rows: list[dict[int, float]] = []
    lower, upper = [], []

    def add(coefficients: dict[int, float], lo: float, hi: float) -> None:
        rows.append(coefficients)
        lower.append(lo)
        upper.append(hi)

    # At most one scheduled coalition per order.
    for j in range(len(jobs)):
        add({index[("z", j, tau, 0)]: 1.0 for tau in range(horizon)}, 0.0, 1.0)
    # A coalition exists iff exactly one AGV and one picker are scheduled.
    for tau in range(horizon):
        for j in range(len(jobs)):
            z = index[("z", j, tau, 0)]
            add({z: -1.0, **{index[("x", ai, j, tau)]: 1.0 for ai in range(len(agvs))}}, 0.0, 0.0)
            add({z: -1.0, **{index[("y", pi, j, tau)]: 1.0 for pi in range(len(pickers))}}, 0.0, 0.0)
    # Occupancy constraints approximate travel/service duration in planning
    # periods; this is the critical distinction from the old one-step matcher.
    for ai in range(len(agvs)):
        duration = np.maximum(1, np.ceil(travel_a[ai] / 0.35).astype(int))
        for period in range(horizon):
            coeff = {}
            for tau in range(period + 1):
                for j in range(len(jobs)):
                    if tau + duration[j] > period:
                        coeff[index[("x", ai, j, tau)]] = 1.0
            add(coeff, 0.0, 1.0)
    for pi in range(len(pickers)):
        duration = np.maximum(1, np.ceil(travel_p[pi] / 0.35).astype(int) + 1)
        for period in range(horizon):
            coeff = {}
            for tau in range(period + 1):
                for j in range(len(jobs)):
                    if tau + duration[j] > period:
                        coeff[index[("y", pi, j, tau)]] = 1.0
            add(coeff, 0.0, 1.0)
    # Work conserving at the execution boundary; future periods remain free to
    # reserve scarce labor for more urgent orders.
    add(
        {index[("z", j, 0, 0)]: 1.0 for j in range(len(jobs))},
        1.0,
        float(min(len(agvs), len(pickers), len(jobs))),
    )

    matrix = lil_matrix((len(rows), nvars), dtype=np.float64)
    for row, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row, column] = value
    result = milp(
        c=objective,
        integrality=np.ones(nvars),
        bounds=Bounds(np.zeros(nvars), variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 1.0, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None:
        return joint_hungarian_actions(obs)
    for j, task in enumerate(jobs):
        if result.x[index[("z", j, 0, 0)]] < 0.5:
            continue
        for ai, agent in enumerate(agvs):
            if result.x[index[("x", ai, j, 0)]] > 0.5:
                actions[agent] = task + 1
        for pi, agent in enumerate(pickers):
            if result.x[index[("y", pi, j, 0)]] > 0.5:
                actions[agent] = task + 1
    return actions
