# Baseline fidelity and comparison roles

## Main systems

- **PLATO-faithful** is a clean-room implementation of the published pointer
  actor, population statistics, bipartite critic, and MAPPO learner. It is not
  author code.
- **MOHITO-512** imports the authors' pinned Actor, Critic, and graph modules.
  The wrapper converts warehouse observations to the required graph and masks
  inactive slots. Replay, target networks, and deterministic updates are
  retained. The 512-transition replay cadence is a disclosed compute
  adaptation, so the paper does not label the complete configuration as an
  unmodified official baseline.
- **Entity-MAPPO** uses masked set pooling without explicit relation edges.
- **Graph-MAPPO** uses the binary 2-section induced from the same semantic
  incidence construction used by CHORD, followed by the common factorized
  MAPPO decoder.
- **Rolling-Horizon MILP** is a project-constructed operational reference. It
  is not an official solver from a cited paper.

## Matched encoder controls

Entity+CAHR and Graph+CAHR use the same candidate builder, capacity context,
sequential reservation decoder, conditional masks, and joint PPO update as
CHORD. CHORD and Graph+CAHR have matched trainable parameter counts. These rows
test representation choice under a shared decoder; they do not establish that
hypergraphs are universally preferable to graphs or sets.

## Mechanism variants

- `chord_no_reservation`: factorized policy without reservation or capacity context.
- `chord_capacity_factorized`: capacity context with factorized per-agent actions.
- `chord_no_capacity`: sequential reservation with capacity descriptors removed.
- `chord_mappo`: complete architecture.

The first two variants change action parameterization and PPO probability
factorization in addition to reservation. Interpret the comparison as a
decoder-package study. `chord_no_capacity` is the cleaner capacity-context
control.

## Fairness boundary

All learned methods receive the same environment, reward, interaction budget,
training seeds, evaluation tapes, and final-checkpoint rule. Common MAPPO
methods also share optimizer and PPO settings. MOHITO retains its intrinsic
off-policy learner and therefore shares the data budget rather than the PPO
update rule.

The machine-readable labels and provenance boundaries are defined in
`open_hyper_warehouse/baselines.py`. The pinned MOHITO source revision is
recorded in `external/mohito-public/UPSTREAM_SOURCE.txt`.
