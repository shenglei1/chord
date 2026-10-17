# CAHR implementation and experiment contract

This document records the code-level contract for the final EAAI method.
Archived development checkpoints that use the earlier independent factorized
decoder are development evidence only.

## 1. Capacity-aware ternary candidates

For each active order, the batched candidate builder selects the nearest `k`
feasible AGVs and nearest `k` feasible pickers from the environment action
mask, then forms at most `k^2` `{AGV, picker, task}` candidates. A one-sided
repair candidate is allowed only when the other role is already unavailable.
Each candidate carries role supply, feasible-resource count, order slack and
priority, role distances, zone load, and completeness. No padded or busy agent
can enter a valid candidate.

## 2. Capacity-aware hyperedge reservation (CAHR)

The policy scores all candidates once. At every reservation round it chooses
STOP or one currently available candidate. After a candidate is selected, all
candidates sharing its AGV, picker, or task are masked. The construction
therefore enforces one order per resource and one coalition per order for the
current decision step. It is a feasibility projection implemented inside the
policy, not an external post-processing rule.

## 3. Exact joint PPO probability

The rollout buffer stores the selected candidate sequence, every conditional
mask, active-round mask, and the sum of conditional log probabilities. During
the PPO update the same sequence is replayed under the new policy and the
importance ratio is computed from the two joint conditional log probabilities.
This prevents a mismatch between independently optimized agent proposals and a
different conflict resolver at execution.

## 4. Fair controls

The main literature/engineering table remains:

- CHORD-CAHR;
- PLATO-faithful;
- MOHITO-official with inactive slots;
- Entity-MAPPO;
- Graph-MAPPO;
- Rolling-Horizon MILP.

The four-way mechanism table contains full CHORD-CAHR, no reservation/no
capacity, capacity with factorized actions, and reservation without capacity.
Entity+CAHR and Graph+CAHR hold the candidate builder, decoder, masks, joint PPO
objective, training budget and ordinary hyperparameters fixed while changing
only the encoder. This is the key control showing whether an improvement comes
from higher-order representation rather than the reservation layer alone.

## 5. Saved evidence

Training telemetry includes policy/value losses, normalized entropy,
approximate KL, clip fraction, critic explained variance, candidate count,
selected coalition count, masked coalition count and reservation rounds.
Evaluation additionally stores decoder latency per environment step. Raw
episode CSVs retain these fields, and `decoder_diagnostics.csv` is generated
with the paper tables.

## 6. MOHITO scaling boundary

The official Actor, Critic, GNN, target networks, replay learner and inactive-
slot wrapper are unchanged. Actor/critic graph outputs are cached once for each
sampled current/target transition. The retained MOHITO-512 configuration uses
`experiences_before_update=512` and `backups_per_update=1`; the chosen cadence
and cache policy are written to the checkpoint.

## 7. Result provenance

Never combine a checkpoint whose `learner_revision` is
`agentwise-ppo-cardinality-critic-feasible-coalitions-v2` with one whose
revision is `cahr-joint-conditional-ppo-v1`. The evaluator can load both for
audit, but formal seed-level inference must use one revision only.
