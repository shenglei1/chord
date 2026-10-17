from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from open_hyper_warehouse.baselines import EAAI_MAIN_LEARNED_KEYS
from open_hyper_warehouse.buffer import to_torch
from open_hyper_warehouse.config import load_config
from open_hyper_warehouse.env import OpenWarehouseEnv
from open_hyper_warehouse.models import build_policy


ALGORITHMS = list(EAAI_MAIN_LEARNED_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eaai_core6_cuda.toml")
    parser.add_argument("--output", help="optional JSON artifact path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # The audit inspects interfaces and parameter budgets only; forcing its
    # temporary models to CPU makes it portable and never mutates the TOML or
    # the device used by actual training runs.
    cfg.train.device = "cpu"
    obs = OpenWarehouseEnv(cfg.env, 0).reset(seed=0)
    batch = {key: value[None] for key, value in obs.items()}
    rows = []
    shared_protocol = {
        "environment_interactions": cfg.train.total_steps,
        "gamma": cfg.train.gamma,
        "reward_and_action_mask": "identical",
        "training_and_evaluation_seed_schedule": "identical",
        "checkpoint_rule": "final interaction budget",
    }
    common_ppo = {
        key: getattr(cfg.train, key)
        for key in (
            "learning_rate", "gae_lambda", "ppo_clip", "entropy_coef",
            "entropy_coef_final", "entropy_anneal_fraction",
            "value_coef", "max_grad_norm", "update_epochs", "minibatch_size"
        )
    }
    for name in ALGORITHMS:
        torch.manual_seed(0)
        policy = build_policy(name, cfg.env, cfg.train, cfg.model).eval()
        torch_batch = to_torch(batch, torch.device("cpu"))
        with torch.no_grad():
            if name == "mohito_official":
                actions, _, _, value = policy.act(torch_batch, deterministic=True)
                finite_logits = bool(torch.isfinite(actions.float()).all())
                value_finite = bool(torch.isfinite(value).all())
            elif getattr(policy, "uses_coalition_reservation", False):
                actions, log_prob, entropy, value, trace = policy.act(
                    torch_batch, deterministic=True, return_trace=True
                )
                valid = torch_batch["action_mask"].gather(
                    -1, actions.unsqueeze(-1)
                ).all()
                finite_logits = bool(
                    valid
                    and torch.isfinite(log_prob).all()
                    and torch.isfinite(entropy).all()
                    and trace["conditional_masks"].any(-1).all()
                )
                value_finite = bool(torch.isfinite(value).all())
            else:
                output = policy(torch_batch)
                finite_logits = bool(torch.isfinite(output.logits[batch["action_mask"]]).all())
                value_finite = bool(torch.isfinite(output.value).all())
        rows.append({
            "algorithm": name,
            "parameters": policy.parameter_count(),
            "finite_logits": finite_logits,
            "value_finite": value_finite,
            "shared_protocol": shared_protocol,
            "learner_specific": (
                {
                    "family": "official MOHITO deterministic actor-critic",
                    "actor_lr": cfg.model.mohito_actor_lr,
                    "critic_lr": cfg.model.mohito_critic_lr,
                    "replay_size": cfg.model.mohito_replay_size,
                    "batch_size": cfg.model.mohito_batch_size,
                    "target_tau": cfg.model.mohito_tau,
                }
                if name == "mohito_official"
                else {
                    "family": (
                        "joint conditional MAPPO"
                        if getattr(policy, "uses_coalition_reservation", False)
                        else "factorized agent-wise MAPPO"
                    ),
                    **common_ppo,
                }
            ),
        })
    # The publication's parameter-matched encoder control is Graph+CAHR. The
    # plain Graph-MAPPO main baseline intentionally retains its own factorized
    # action head and therefore need not have the CAHR decoder parameters.
    torch.manual_seed(0)
    graph_cahr = build_policy("graph_cahr", cfg.env, cfg.train, cfg.model)
    chord_count = next(row["parameters"] for row in rows if row["algorithm"] == "chord_mappo")
    if chord_count != graph_cahr.parameter_count():
        raise AssertionError("Graph+CAHR must match the proposed parameter budget")
    rendered = json.dumps(rows, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
