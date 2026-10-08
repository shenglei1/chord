# Single-scenario semantic relation analysis

## Seeds and scenario

The analysis uses trained CHORD checkpoints for seeds `1`, `45`, `789`,
`2027`, `4096`, and `8191`. It does not train missing checkpoints.

Only `robot_breakdown` is evaluated. Each checkpoint is run on the same 20
execution seeds. Episodes are aggregated within each training seed before the
six training seeds are summarized. Evaluation episodes are not independent
training units.

For the 256-tick configuration, interruption occurs at tick 85 and service
capacity is restored at tick 170. Stored observations represent the
pre-interruption, post-interruption, and post-restoration stages. Environment
advancement also changes movement, arrivals, and other background state, so
stage differences are not a single-variable physical intervention. Capacity
restoration activates an available slot and does not guarantee that the same
physical robot identity returns.

## Run the analysis

From the project root:

```powershell
python -u scripts\explain_chord.py `
  --checkpoints-root results\main_comparison\chord_mappo `
  --seeds 1 45 789 2027 4096 8191 `
  --episodes 20 `
  --output results\semantic_analysis
```

The command performs inference only. Reusing an output directory replaces the
derived explanation files in that directory.

## Outputs

- `semantic_attribution.json`: observations, scenario parameters, assignments,
  order identifiers, and edge-level interventions.
- `edge_attributions.csv`: probability change, Jensen-Shannon divergence, and
  critic-value change for each masked edge.
- `semantic_event_summary.csv`: episode-first, then stage-level summaries.
- `case_timelines.csv`: service, delivery, expiry, contention, interruption,
  and restoration traces.
- `semantic_case_study.md`: median-service cases for each training seed.

Orders are tracked by `order_id` because padded task slots are reused. Resource
records include slot, active state, and join time; a slot is not treated as a
permanent physical identity.

## Interpretation boundary

The intervention holds the observation and original reservation trace fixed,
masks one encoded edge, and recomputes policy/value predictions. It measures
model-internal structural sensitivity. It does not establish physical
causality, human-grounded explanation quality, or a new closed-loop outcome.

Graph+CAHR uses a binary 2-section encoder followed by the same capacity-aware
triple-candidate reservation decoder. Entity+CAHR uses set pooling followed by
that decoder. Their encoder and decoder descriptions are therefore compatible:
the controls remove explicit hyperedge message passing, not triple candidates.
They are trained once per seed and evaluated on all Core-8 scenarios; semantic
edge masking is performed only for CHORD in `robot_breakdown`.
