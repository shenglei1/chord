"""Capacity-aware candidate service coalitions for reservation policies.

Candidate ordering is deterministic for a fixed padded observation.  A block
of ``top_k ** 2`` entries is reserved for every task so rollout traces can
store conditional masks without duplicating the candidate identities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import EnvConfig


COALITION_CAPACITY_DIM = 10


@dataclass(slots=True)
class CoalitionCandidates:
    agv: Tensor
    picker: Tensor
    task: Tensor
    mask: Tensor
    capacity_features: Tensor

    @property
    def size(self) -> int:
        return int(self.mask.shape[-1])


def build_coalition_candidates(
    obs: dict[str, Tensor],
    env_cfg: EnvConfig,
    top_k: int,
) -> CoalitionCandidates:
    """Return feasible complete or repair coalitions and load descriptors.

    Complete waiting jobs create ``(AGV, picker, task)`` triples.  If only one
    role is feasible (for example after a repairable failure releases one side
    of an existing coalition), a one-sided repair candidate is emitted with
    ``-1`` in the already occupied role.
    """

    top_k = max(int(top_k), 1)
    batch = int(obs["agent_features"].shape[0])
    tasks = env_cfg.max_tasks
    block = top_k * top_k
    device = obs["agent_features"].device
    dtype = obs["agent_features"].dtype
    agent_features = obs["agent_features"]
    task_features = obs["task_features"]
    agent_mask = obs["agent_mask"].bool()
    task_mask = obs["task_mask"].bool()
    feasible = obs["action_mask"][..., 1:].bool()
    feasible &= agent_mask.unsqueeze(-1) & task_mask.unsqueeze(1)
    agv_feasible = feasible & (agent_features[..., 0] > 0.5).unsqueeze(-1)
    picker_feasible = feasible & (agent_features[..., 1] > 0.5).unsqueeze(-1)

    # All operations below are batched. This path is called from every PPO
    # minibatch, so avoiding per-observation Python loops is essential for the
    # publication-scale GPU run.
    distance = (
        agent_features[:, :, None, 2:4] - task_features[:, None, :, 0:2]
    ).abs().sum(-1)
    inf = torch.finfo(dtype).max
    agv_distance = distance.masked_fill(~agv_feasible, inf)
    picker_distance = distance.masked_fill(~picker_feasible, inf)
    _, agv_top = torch.topk(agv_distance, top_k, dim=1, largest=False, sorted=True)
    _, picker_top = torch.topk(
        picker_distance, top_k, dim=1, largest=False, sorted=True
    )
    agv_top = agv_top.transpose(1, 2)       # [B, T, K]
    picker_top = picker_top.transpose(1, 2) # [B, T, K]
    agv_valid = torch.gather(
        agv_feasible.transpose(1, 2), 2, agv_top
    )
    picker_valid = torch.gather(
        picker_feasible.transpose(1, 2), 2, picker_top
    )
    agv_distance_top = torch.gather(
        distance.transpose(1, 2), 2, agv_top
    )
    picker_distance_top = torch.gather(
        distance.transpose(1, 2), 2, picker_top
    )

    agv_grid = agv_top.unsqueeze(-1).expand(-1, -1, -1, top_k)
    picker_grid = picker_top.unsqueeze(-2).expand(-1, -1, top_k, -1)
    agv_grid_valid = agv_valid.unsqueeze(-1).expand_as(agv_grid)
    picker_grid_valid = picker_valid.unsqueeze(-2).expand_as(picker_grid)
    has_agv = agv_valid.any(-1, keepdim=True).unsqueeze(-1)
    has_picker = picker_valid.any(-1, keepdim=True).unsqueeze(-1)
    complete = agv_grid_valid & picker_grid_valid & has_agv & has_picker
    first_picker_column = (
        torch.arange(top_k, device=device).view(1, 1, 1, top_k) == 0
    )
    first_agv_row = (
        torch.arange(top_k, device=device).view(1, 1, top_k, 1) == 0
    )
    agv_repair = agv_grid_valid & ~has_picker & first_picker_column
    picker_repair = picker_grid_valid & ~has_agv & first_agv_row
    mask_4d = complete | agv_repair | picker_repair
    agv_grid = torch.where(complete | agv_repair, agv_grid, -torch.ones_like(agv_grid))
    picker_grid = torch.where(
        complete | picker_repair, picker_grid, -torch.ones_like(picker_grid)
    )

    task_grid = torch.arange(tasks, device=device).view(1, tasks, 1, 1)
    task_grid = task_grid.expand(batch, -1, top_k, top_k)
    task_grid = torch.where(mask_4d, task_grid, -torch.ones_like(task_grid))
    da = agv_distance_top.unsqueeze(-1).expand(-1, -1, -1, top_k)
    dp = picker_distance_top.unsqueeze(-2).expand(-1, -1, top_k, -1)
    da = torch.where(complete | agv_repair, da, torch.zeros_like(da))
    dp = torch.where(complete | picker_repair, dp, torch.zeros_like(dp))

    agv_count = agv_feasible.sum(1).to(dtype)
    picker_count = picker_feasible.sum(1).to(dtype)
    feasible_count = feasible.sum(1).to(dtype)
    agent_zone = torch.clamp(
        (agent_features[..., 2] * env_cfg.num_zones).long(),
        0,
        env_cfg.num_zones - 1,
    )
    task_zone = torch.clamp(
        (task_features[..., 0] * env_cfg.num_zones).long(),
        0,
        env_cfg.num_zones - 1,
    )
    zone_agent_count = (
        (agent_zone.unsqueeze(1) == task_zone.unsqueeze(-1))
        & agent_mask.unsqueeze(1)
    ).sum(-1).to(dtype)
    zone_task_count = (
        (task_zone.unsqueeze(1) == task_zone.unsqueeze(-1))
        & task_mask.unsqueeze(1)
    ).sum(-1).to(dtype)

    def expand_task(values: Tensor) -> Tensor:
        return values[:, :, None, None].expand(-1, -1, top_k, top_k)

    features = torch.stack(
        [
            expand_task(agv_count / max(env_cfg.max_agvs, 1)),
            expand_task(picker_count / max(env_cfg.max_pickers, 1)),
            expand_task(feasible_count / max(env_cfg.max_agents, 1)),
            expand_task(task_features[..., 5]),
            expand_task(task_features[..., 4]),
            da,
            dp,
            expand_task(zone_agent_count / max(env_cfg.max_agents, 1)),
            expand_task(zone_task_count / max(env_cfg.max_tasks, 1)),
            complete.to(dtype),
        ],
        dim=-1,
    )
    count = tasks * block
    return CoalitionCandidates(
        agv_grid.reshape(batch, count),
        picker_grid.reshape(batch, count),
        task_grid.reshape(batch, count),
        mask_4d.reshape(batch, count),
        features.reshape(batch, count, COALITION_CAPACITY_DIM),
    )


def task_capacity_features(
    obs: dict[str, Tensor], env_cfg: EnvConfig
) -> Tensor:
    """Permutation-invariant per-task competition features for an ablation."""

    batch, tasks = obs["task_mask"].shape
    output = obs["task_features"].new_zeros((batch, tasks, COALITION_CAPACITY_DIM))
    action_mask = obs["action_mask"][..., 1:].bool()
    agents = obs["agent_features"]
    agent_mask = obs["agent_mask"].bool()
    task_mask = obs["task_mask"].bool()
    feasible = action_mask & agent_mask.unsqueeze(-1) & task_mask.unsqueeze(1)
    agvs = feasible & (agents[..., 0] > 0.5).unsqueeze(-1)
    pickers = feasible & (agents[..., 1] > 0.5).unsqueeze(-1)
    agv_count = agvs.sum(1)
    picker_count = pickers.sum(1)
    distance = (
        agents[:, :, None, 2:4] - obs["task_features"][:, None, :, :2]
    ).abs().sum(-1)
    output[..., 0] = agv_count / max(env_cfg.max_agvs, 1)
    output[..., 1] = picker_count / max(env_cfg.max_pickers, 1)
    output[..., 2] = (agv_count + picker_count) / max(env_cfg.max_agents, 1)
    output[..., 3] = obs["task_features"][..., 5]
    output[..., 4] = obs["task_features"][..., 4]
    output[..., 5] = distance.masked_fill(~agvs, 1.0).amin(1)
    output[..., 6] = distance.masked_fill(~pickers, 1.0).amin(1)
    agent_zone = torch.clamp(
        (agents[..., 2] * env_cfg.num_zones).long(), 0, env_cfg.num_zones - 1
    )
    task_zone = torch.clamp(
        (obs["task_features"][..., 0] * env_cfg.num_zones).long(),
        0,
        env_cfg.num_zones - 1,
    )
    output[..., 7] = (
        ((agent_zone.unsqueeze(1) == task_zone.unsqueeze(-1)) & agent_mask.unsqueeze(1))
        .sum(-1)
        / max(env_cfg.max_agents, 1)
    )
    output[..., 8] = (
        ((task_zone.unsqueeze(1) == task_zone.unsqueeze(-1)) & task_mask.unsqueeze(1))
        .sum(-1)
        / max(env_cfg.max_tasks, 1)
    )
    output[..., 9] = ((agv_count > 0) & (picker_count > 0)).to(output.dtype)
    return output * task_mask.unsqueeze(-1).to(output.dtype)
