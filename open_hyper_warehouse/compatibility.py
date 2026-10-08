"""End-to-end compatibility audit for the locked EAAI main comparison.

This is an integration check, not a performance experiment: untrained policies
are deliberately used so that checkpoint quality cannot hide an interface or
action-mask failure in an open-population scenario.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .baselines import EAAI_MAIN_LEARNED_KEYS, EAAI_MAIN_OPERATIONAL_KEYS
from .buffer import to_torch
from .config import ExperimentConfig
from .eaai_protocol import all_eaai_scenarios
from .env import OpenWarehouseEnv
from .heuristics import rolling_horizon_actions
from .models import build_policy
from .scenarios import make_scenario


def _assert_valid_actions(obs: dict[str, np.ndarray], actions: np.ndarray) -> None:
    if actions.shape != (obs["agent_mask"].shape[0],):
        raise AssertionError(f"invalid action shape {actions.shape}")
    slots = np.arange(actions.shape[0])
    if not np.all(obs["action_mask"][slots, actions]):
        raise AssertionError("method selected an action excluded by the shared feasibility mask")
    if not np.all(actions[~obs["agent_mask"]] == 0):
        raise AssertionError("inactive agent slot selected a non-NOOP action")


@torch.no_grad()
def audit_main_baseline_compatibility(
    config: ExperimentConfig,
    execution_seed: int = 73_000,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one complete fixed-seed episode per main method and scenario."""

    learned = {}
    for index, name in enumerate(EAAI_MAIN_LEARNED_KEYS):
        torch.manual_seed(index)
        learned[name] = build_policy(name, config.env, config.train, config.model).eval()

    scenario_names = scenarios or all_eaai_scenarios()
    rows: list[dict[str, Any]] = []
    methods = (*EAAI_MAIN_LEARNED_KEYS, *EAAI_MAIN_OPERATIONAL_KEYS)
    for scenario_index, scenario in enumerate(scenario_names):
        scenario_cfg = make_scenario(config.env, scenario)
        for method_index, method in enumerate(methods):
            seed = execution_seed + scenario_index * 100 + method_index
            env = OpenWarehouseEnv(scenario_cfg, seed=seed)
            obs = env.reset(seed=seed)
            done = False
            steps = 0
            info: dict[str, Any] = {}
            while not done:
                if method == "rolling_horizon":
                    actions = rolling_horizon_actions(obs)
                else:
                    batch = to_torch(
                        {key: value[None] for key, value in obs.items()},
                        torch.device("cpu"),
                    )
                    actions = learned[method].act(batch, deterministic=True)[0][0].cpu().numpy()
                _assert_valid_actions(obs, actions)
                obs, _, done, info = env.step(actions)
                steps += 1
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "execution_seed": seed,
                    "steps": steps,
                    "valid_actions": True,
                    "agent_joins": int(info.get("agent_joins", 0)),
                    "agent_leaves": int(info.get("agent_leaves", 0)),
                    "forced_failures": int(info.get("forced_failures", 0)),
                    "repairs": int(info.get("repairs", 0)),
                    "shift_changes": int(info.get("shift_changes", 0)),
                }
            )
    return {
        "purpose": "interface/action-feasibility audit; not a performance result",
        "scenario_count": len(scenario_names),
        "method_count": len(methods),
        "episode_count": len(rows),
        "all_valid": all(row["valid_actions"] for row in rows),
        "rows": rows,
    }
