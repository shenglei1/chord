# Paper results, statistics and figure plan

## Statistical unit and reporting rule

The independent unit is one independently trained policy seed. Each checkpoint
is evaluated for 100 episodes per scenario, and those episodes first produce a
single scenario mean for that checkpoint. Paper uncertainty is then computed
over the six policy seeds `1, 45, 789, 2027, 4096, 8191`. Evaluation episodes
must not be counted as 600 independent trained policies.

The default paper interval is a percentile 95% bootstrap confidence interval
over the six seed means. `mean ± standard deviation` is supported as an
alternative display, but one convention must be used consistently. Exact
paired sign-flip tests and Holm adjustment remain in the existing aggregate
files; with six seeds the smallest attainable two-sided exact p-value is
0.03125, so effect size and interval estimates must accompany p-values.

Rolling-horizon MILP is a deterministic operational reference, not a trained
six-seed learner. It is therefore shown as a point estimate. Its comparison
interval reflects variation among the six CHORD policies, not six independent
MILP trainings.

## Metric definitions and safe wording

| Metric | Code definition | Paper wording |
|---|---|---|
| Service level | deliveries / (created orders - cancellations) | fraction of eligible admitted orders served |
| Throughput / 100 ticks | 100 × deliveries / simulator counter ticks | deliveries per 100 simulator counter ticks |
| Deadline-miss rate | expired tasks / eligible admitted orders | fraction of admitted eligible orders that expire |
| Mean flow time | creation-to-delivery simulator steps / deliveries | simulator steps per delivered order |
| Contention / delivery | rejected duplicate same-role proposals / deliveries | coordination proposal conflicts, **not physical collisions** |
| Travel / delivery | grid moves / deliveries | grid moves per delivery, **not metres** |
| Utilization | busy active-agent steps / active-agent steps | dispatch utilization |

`orders_created` is not currently exported. The Core6 CSVs are internally
consistent and imply that every offered order was admitted in those runs. For
the `demand_350` sensitivity condition, however, an admission rate is needed to
rule out task-slot saturation. Add `orders_created` or `rejected_orders` before
running the sensitivity evaluation, and report both admitted-order service
level and offered-demand admission rate. This requires evaluation only; the six
CHORD policies do not need retraining.

## Main-text artifacts

1. `table_main_results_seed_level.csv`: absolute primary results for all six
   scenarios. Claim: descriptive performance of complete decision systems
   under the tested open-resource and open-order conditions. Report mean and
   95% CI over six training seeds and retain conditions where CHORD is not best.
2. `fig_main_primary_metrics`: four-panel absolute performance plot. X-axis:
   categorical Core6 scenarios. Y-axes: native metric units with zero included;
   ratio panels use [0, 1]. Claim: operational performance across scenarios.
3. `fig_chord_oriented_effects`: four-panel effect plot. X-axis: Core6
   scenarios. Y-axis: direction-normalized CHORD-minus-baseline difference, so
   positive always favors CHORD. Claim: magnitude and uncertainty of advantage.
4. `fig_scale_generalization`: unseen-small → training distribution →
   unseen-large. Y-axes: service level and throughput. Claim: zero-shot joint
   scale-load transfer; population and workload are not independently varied.
5. `fig_component_ablations`: CHORD versus three capacity/reservation variants
   on in-distribution, AGV-heavy and robot-breakdown scenarios. Claim: compare
   the complete decoder package and capacity context with the implemented
   factorized alternatives; this is not an orthogonal causal decomposition.
6. `fig_encoder_controls`: CHORD versus Entity+CAHR and Graph+CAHR with the
   decoder held fixed. Claim: scenario-dependent evidence about representation
   choice beyond the shared reservation decoder, including null or reverse effects.
7. `fig_sensitivity_dose_response` (pending): failure fraction and demand-rate
   multiplier on the x-axes; service level and throughput on the y-axes. Claim:
   graceful degradation under increasing operational stress.

Every multi-panel figure is exported both as a complete review figure and as
standalone LaTeX-ready panels. Standalone files use
`<figure>_A_<metric>`, `<figure>_B_<metric>`, and so on, in PDF, SVG and PNG.
Panel letters should be supplied by the LaTeX `subfigure`/`subcaption`
environment rather than embedded in the image.

## Appendix artifacts

- `fig_training_curves_unsmoothed`: environment interactions on x; service
  level and throughput on y. It displays every fifth logged checkpoint without
  a moving average, with seed-level intervals.
- `fig_semantic_edge_masking`: pre-breakdown, post-breakdown and post-repair on
  x; absolute policy-probability change after masking on y. It shows all three
  available training-seed points plus their mean. This is a structurally
  transparent sensitivity case, not a causal explanation.
- `table_chord_oriented_effects.csv`: all effect sizes and intervals.
- `table_metric_dictionary.csv`: definitions, units and optimization direction.
- Decoder latency and training wall time may only be placed in an appendix with
  a hardware label. Current runs mix RTX 3090, RTX 4060 Laptop and RTX 5060 Ti
  devices, and MOHITO lacks a recorded GPU name. A cross-method efficiency
  claim requires a same-machine inference benchmark; training does not need to
  be repeated unless wall-clock training efficiency is itself a paper claim.

## Ablation caveat

The three-scene ablation runner generates evaluation seeds from scenario
position. `agv_heavy_ratio` and `robot_breakdown` occupy different positions in
Core6 and the three-scene suite. The current means and seed-level intervals are
valid descriptive results, but the phrase “identical evaluation trajectories”
and strict paired-trajectory inference are not. To use that claim, reevaluate
all existing CHORD and ablation checkpoints with one common three-scene order.
No policy retraining is required.

## Reproducible commands

Install the plotting extra once:

```powershell
python -m pip install -e ".[plot]"
```

Audit tables without plotting:

```powershell
python -u .\scripts\generate_paper_artifacts.py `
  --results-root .\results `
  --output .\figs\paper_artifacts `
  --uncertainty ci95 `
  --audit-only
```

Generate PDF, SVG and PNG figures with 95% CIs:

```powershell
python -u .\scripts\generate_paper_artifacts.py `
  --results-root .\results `
  --output .\figs\paper_artifacts `
  --uncertainty ci95 `
  --formats pdf svg png
```

For mean ± standard deviation, replace `--uncertainty ci95` with
`--uncertainty std`. Do not combine CI and standard-deviation error bars in the
same figure.

Generate the twelve numbered panels currently referenced as Figs. 4, 7, 11,
and 12 in the EAAI manuscript:

```powershell
python -u .\scripts\generate_eaai_numbered_figures.py `
  --results-root .\results `
  --output .\figs\paper_artifacts\eaai_numbered_figures `
  --formats pdf svg png
```

The command reads, without modifying, the matched-encoder aggregate, paired
CHORD-minus-control effects, and sensitivity aggregate.  It writes
`FIG4A`--`FIG4D`, `FIG7A`--`FIG7D`, and `FIG11A`--`FIG12B` in each requested
format.  PDF is the LaTeX artwork; SVG is the editable vector copy; PNG is the
600-dpi preview.  `figure_provenance.csv` and `figure_provenance.json` preserve
the source filters and input hashes.
