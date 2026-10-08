# EAAI warehouse decision protocol

## Scope

The study evaluates high-level assignment of available AGVs and pickers to
orders in an application-oriented warehouse simulation. The controller reads a
padded global observation and executes centrally. The implementation does not
model path planning, collision avoidance, robot dynamics, communication delay,
or physical deployment.

The engineering question is whether one structured policy can continue making
feasible joint commitments when team composition, role balance, pending orders,
and resource availability change.

## Locked training and evaluation design

- Independent training seeds: `1`, `45`, `789`, `2027`, `4096`, `8191`.
- Training budget: 1,024,000 environment interactions per learned method and seed.
- Checkpoint rule: final checkpoint; test scenarios do not select checkpoints.
- Evaluation: 100 deterministic episodes per seed and scenario.
- Statistical unit: independently trained policy seed.
- Primary uncertainty: 95% bootstrap confidence interval over six seed means.
- Common MAPPO settings: reward, discount, GAE, PPO clipping, optimizer,
  rollout length, epochs, minibatches, gradient clipping, and evaluation tapes.
- Method-specific settings are limited to intrinsic mechanisms. MOHITO retains
  its replay learner and uses the locked 512-transition replay cadence.

## Core-8 scenarios

| Scenario | Evaluation purpose |
|---|---|
| `in_distribution` | Nominal reference condition |
| `unseen_small` | Joint scale-load transfer to a smaller operating instance |
| `unseen_large` | Joint scale-load transfer to a larger operating instance |
| `agv_heavy_ratio` | Picker-scarce role imbalance |
| `picker_heavy_ratio` | AGV-scarce role imbalance |
| `picker_shift` | Scheduled picker replacement |
| `fixed_team_dynamic_orders` | Task openness with a fixed workforce |
| `robot_breakdown` | Targeted interruption and restoration of active AGV capacity |

The scale scenarios change both population and workload. They do not isolate a
pure population-size intervention. The breakdown scenario preferentially
interrupts busy AGVs so that an established service commitment is disrupted;
it is a targeted recovery stress test, not a field failure-probability model.

## Main comparison

- `chord_mappo`: complete proposed decision architecture.
- `plato_faithful`: paper-faithful clean-room pointer-policy adaptation.
- `mohito_official`: compute-adapted MOHITO-512 with an inactive-slot wrapper.
- `entity_mappo`: invariant set encoding without explicit relations.
- `graph_mappo`: binary 2-section relation encoding.
- `rolling_horizon`: project-constructed rolling-horizon MILP reference.

The main table compares complete decision systems. It changes representation,
action support, and conflict handling; it is not a pure encoder comparison.
Entity+CAHR and Graph+CAHR hold the reservation decoder fixed and provide the
matched representation controls.

## Mechanism study

The three locked scenarios are `in_distribution`, `agv_heavy_ratio`, and
`robot_breakdown`. They compare full CHORD with:

- `chord_no_reservation`;
- `chord_capacity_factorized`;
- `chord_no_capacity`.

Removing reservation also changes the action factorization, conditional masks,
entropy, and PPO likelihood. These rows therefore form a decoder-package and
capacity-context comparison rather than an orthogonal two-factor causal design.

## Sensitivity and structural transparency

Sensitivity reuses the six trained CHORD checkpoints and evaluates failure
fractions `0`, `0.10`, `0.25`, `0.40` and demand multipliers `0.5`, `1.0`,
`2.0`, `3.5`. Claims must remain within these tested levels.

Semantic analysis uses only `robot_breakdown`. Each edge is masked at a fixed
observation and the policy/value response is recomputed without a new rollout.
The output measures conditional prediction sensitivity, not physical causality
or human-grounded explanation.

## Primary metrics

- service level;
- deliveries per 100 simulator counter ticks;
- deadline-miss rate;
- mean flow time in simulator counter ticks.

Secondary diagnostics include order-balance counts, travel per delivery,
contention per delivery, resource utilization, joins/leaves, forced failures,
repairs, candidate counts, masked candidates, and reservation rounds.

## Claim boundary

The evidence supports an application-specific structured joint-decision
architecture under the tested simulation conditions. It does not establish
universal hypergraph superiority, decentralized execution, navigation safety,
hard real-time performance, industrial deployment, or sim-to-real transfer.
