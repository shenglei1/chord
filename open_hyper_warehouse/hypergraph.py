from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class IncidenceBatch:
    matrix: Tensor
    edge_mask: Tensor


def build_open_incidence(
    agent_features: Tensor,
    task_features: Tensor,
    agent_mask: Tensor,
    task_mask: Tensor,
    action_mask: Tensor,
    num_zones: int,
    top_k: int,
    include_tasks: bool = True,
    include_zones: bool = True,
) -> IncidenceBatch:
    """Build task-coalition and shared-zone hyperedges.

    A task edge contains the task plus the nearest feasible AGVs and pickers.
    Zone edges represent shared spatial resources. Construction is based only
    on observable features and is equivariant to entity permutations.
    """

    batch, num_agents, _ = agent_features.shape
    num_tasks = task_features.shape[1]
    num_nodes = num_agents + num_tasks
    max_edges = num_tasks + num_zones
    incidence = agent_features.new_zeros((batch, num_nodes, max_edges))
    edge_mask = torch.zeros((batch, max_edges), dtype=torch.bool, device=agent_features.device)

    for b in range(batch):
        active_agents = torch.nonzero(agent_mask[b], as_tuple=False).flatten()
        active_tasks = torch.nonzero(task_mask[b], as_tuple=False).flatten()
        if include_tasks:
            for task_slot in active_tasks.tolist():
                edge = task_slot
                edge_mask[b, edge] = True
                incidence[b, num_agents + task_slot, edge] = 1.0
                task_xy = task_features[b, task_slot, :2]
                for role_index in (0, 1):
                    candidates = active_agents[
                        (agent_features[b, active_agents, role_index] > 0.5)
                        & action_mask[b, active_agents, task_slot + 1]
                    ]
                    if candidates.numel() == 0:
                        continue
                    distances = torch.abs(agent_features[b, candidates, 2:4] - task_xy).sum(dim=-1)
                    # Include every candidate tied at the k-th distance.  A
                    # slot-index tie-break would silently violate permutation
                    # equivariance when two robots occupy the same position.
                    k = min(top_k, candidates.numel())
                    boundary = torch.kthvalue(distances, k).values
                    nearest = candidates[distances <= boundary + 1e-7]
                    incidence[b, nearest, edge] = 1.0

        for zone in range(num_zones if include_zones else 0):
            edge = num_tasks + zone
            lo, hi = zone / num_zones, (zone + 1) / num_zones
            agent_x = agent_features[b, :, 2]
            task_x = task_features[b, :, 0]
            in_agent_zone = agent_mask[b] & (agent_x >= lo) & ((agent_x < hi) | ((zone == num_zones - 1) & (agent_x <= hi)))
            in_task_zone = task_mask[b] & (task_x >= lo) & ((task_x < hi) | ((zone == num_zones - 1) & (task_x <= hi)))
            if in_agent_zone.any() or in_task_zone.any():
                edge_mask[b, edge] = True
                incidence[b, :num_agents, edge] = in_agent_zone.to(incidence.dtype)
                incidence[b, num_agents:, edge] = in_task_zone.to(incidence.dtype)
    return IncidenceBatch(incidence, edge_mask)


class DenseHypergraphLayer(nn.Module):
    """Permutation-equivariant node→hyperedge→node message passing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, incidence: Tensor, edge_mask: Tensor) -> Tensor:
        node_to_edge_degree = incidence.sum(dim=1).clamp_min(1.0)
        edges = torch.einsum("bve,bvh->beh", incidence, nodes) / node_to_edge_degree.unsqueeze(-1)
        edges = self.edge_mlp(edges) * edge_mask.unsqueeze(-1)
        edge_to_node_degree = incidence.sum(dim=2).clamp_min(1.0)
        messages = torch.einsum("bve,beh->bvh", incidence, edges) / edge_to_node_degree.unsqueeze(-1)
        update = self.node_mlp(torch.cat([nodes, messages], dim=-1))
        return self.norm(nodes + update)


class DenseHypergraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(DenseHypergraphLayer(hidden_dim) for _ in range(layers))

    def forward(self, nodes: Tensor, incidence: IncidenceBatch) -> Tensor:
        for layer in self.layers:
            nodes = layer(nodes, incidence.matrix, incidence.edge_mask)
        return nodes


def incidence_to_pairwise_adjacency(incidence: IncidenceBatch) -> Tensor:
    """Return the binary 2-section graph induced by a hypergraph incidence.

    The conversion deliberately discards hyperedge identity and cardinality:
    two nodes are adjacent iff they co-occur in at least one active hyperedge.
    This gives Graph-MAPPO exactly the same observable relations as the
    proposed method, while restricting it to pairwise message passing.
    """

    active = incidence.matrix * incidence.edge_mask.unsqueeze(1).to(incidence.matrix.dtype)
    adjacency = torch.bmm(active, active.transpose(1, 2)) > 0
    node_mask = active.sum(dim=2) > 0
    adjacency &= node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adjacency


class DensePairwiseGraphLayer(nn.Module):
    """Permutation-equivariant pairwise control matched to a hypergraph layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # These modules mirror DenseHypergraphLayer's parameter budget.  The
        # only controlled difference is one-hop pairwise aggregation versus
        # node-to-hyperedge-to-node aggregation.
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        weights = adjacency.to(nodes.dtype)
        degree = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        messages = torch.bmm(weights, self.message_mlp(nodes)) / degree
        update = self.node_mlp(torch.cat([nodes, messages], dim=-1))
        return self.norm(nodes + update)


class DensePairwiseGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(DensePairwiseGraphLayer(hidden_dim) for _ in range(layers))

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        for layer in self.layers:
            nodes = layer(nodes, adjacency)
        return nodes


def permute_incidence(incidence: Tensor, node_permutation: Tensor) -> Tensor:
    """Utility used by tests for the equivariance invariant."""

    return incidence.index_select(1, node_permutation)
