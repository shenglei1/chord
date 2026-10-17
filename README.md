# CHORD-CAHR: structured warehouse coalition reservation

This repository contains the implementation and experiment evidence for a
high-level warehouse dispatch study. The decision layer assigns transport
robots and pickers to dynamically arriving orders. It does not implement path
planning, collision avoidance, robot dynamics, ROS, or physical-robot control.

The policy is centrally trained and centrally executed from a padded global
observation. The proposed method combines semantically constructed
order-coalition and shared-zone hyperedges with sequential feasible-resource
reservation. The reservation decoder prevents two selected coalitions from
sharing an agent or an order within one decision.

## Paper methods

The main comparison contains:

| CLI key | Paper role |
|---|---|
| `chord_mappo` | Proposed CHORD-CAHR policy |
| `plato_faithful` | Paper-faithful clean-room PLATO adaptation |
| `mohito_official` | Compute-adapted MOHITO-512 with a warehouse inactive-slot wrapper |
| `entity_mappo` | Set-encoder MAPPO control |
| `graph_mappo` | Pairwise 2-section MAPPO control |
| `rolling_horizon` | Project-constructed rolling-horizon MILP reference |

The decoder-matched representation controls are `entity_cahr` and
`graph_cahr`. The locked mechanism variants are `chord_no_reservation`,
`chord_capacity_factorized`, and `chord_no_capacity`.

PLATO is a clean-room implementation based on the paper rather than author
code. MOHITO imports the authors' public network implementation pinned to
commit `105ddccbacc92953a7aa6bd0d19f8ec299f7476d`. Its replay cadence is an
algorithm-specific setting; the final paper configuration uses one update per
512 collected transitions and is therefore described as compute-adapted.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
```

The local virtual environment is intentionally excluded from version control.

## Reproduce the locked experiments

The six independent training seeds are `1`, `45`, `789`, `2027`, `4096`, and
`8191`. Learned methods use 1,024,000 environment interactions. Evaluation
uses 100 deterministic episodes per seed and Core-6 scenario. Statistical
aggregates treat the six trained policies, not evaluation episodes, as the
independent units.

Run the main CUDA suite:

```powershell
python -u scripts\run_eaai_robotics_suite.py `
  --config configs\eaai_core6_cuda.toml `
  --scenario-set core6 `
  --seeds 1 45 789 2027 4096 8191 `
  --section main `
  --output eaai_core6_results
```

Run the compute-adapted MOHITO configuration separately:

```powershell
python -u scripts\run_eaai_robotics_suite.py `
  --config configs\eaai_core6_cuda_mohito_512.toml `
  --scenario-set core6 `
  --seeds 1 45 789 2027 4096 8191 `
  --section main `
  --algorithms mohito_official `
  --output eaai_mohito512_results
```

Run the locked three-scenario mechanism study:

```powershell
python -u scripts\run_eaai_ablation3.py `
  --config configs\eaai_core6_cuda.toml `
  --seeds 1 45 789 2027 4096 8191 `
  --algorithms chord_no_reservation chord_capacity_factorized chord_no_capacity `
  --output eaai_ablation3_results
```

Evaluate the trained CHORD policies across the failure and demand dose levels:

```powershell
python -u scripts\run_chord_sensitivity.py `
  --checkpoints-root results\main_comparison\chord_mappo `
  --config configs\cie_eaai_sensitivity.toml `
  --output results\sensitivity\chord
```

Generate the semantic breakdown analysis and publication artifacts:

```powershell
python -u scripts\explain_chord.py `
  --checkpoints-root results\main_comparison\chord_mappo `
  --episodes 20 `
  --output results\semantic_analysis

python -u scripts\build_semantic_paper_artifacts.py `
  --root results\semantic_analysis `
  --figures-output figs\semantic_analysis `
  --formats pdf svg png

python -u scripts\generate_paper_artifacts.py `
  --results-root results `
  --output figs\paper_artifacts `
  --formats pdf svg png
```

Regenerate the manuscript-specific scenario panels, matched-control effects,
and shared method legends directly in `figs`:

```powershell
python -u scripts\regenerate_retained_scenario_plots.py
python -u scripts\regenerate_method_legends.py
```

Generate the twelve derived panels used for the unseen-small,
stress-sensitivity, and complete matched-control figures:

```powershell
python -u scripts\generate_eaai_numbered_figures.py `
  --results-root results `
  --output figs\paper_artifacts\eaai_numbered_figures `
  --formats pdf svg png
```

Synchronize the mapped PDF panels to the filenames referenced by LaTeX:

```powershell
python -u scripts\sync_eaai_manuscript_figures.py `
  --paper-artifacts figs\paper_artifacts `
  --semantic-artifacts figs\semantic_analysis `
  --output-dir figs
```

This command is read-only with respect to experiment results.  The three input
aggregate CSV files and their SHA-256 hashes are recorded in
`figure_provenance.json`.

## Repository layout

- `open_hyper_warehouse/`: environment, policies, trainers, evaluation, and analysis.
- `configs/`: locked publication configurations.
- `scripts/`: experiment, audit, aggregation, explanation, and figure entry points.
- `figs/`: all generated manuscript figures and intermediate artwork.
- `external/mohito-public/`: vendored upstream MOHITO source without nested VCS metadata.
- `results/`: consolidated final checkpoints, raw evaluation records, and statistics.
- `docs/`: EAAI protocol, method boundaries, metric definitions, and claim-evidence records.

The upstream MOHITO repository and pinned revision are recorded in
`external/mohito-public/UPSTREAM_SOURCE.txt`. No nested repository metadata is
required to run the baseline.

The simulator is an application-oriented abstraction. Reported conclusions
are limited to the tested high-level assignment regimes and must not be
presented as physical-robot deployment evidence.
