# CHORD-CAHR warehouse dispatch code


## Environment

Python 3.10 or newer is required. All commands below are run from the
repository root, so paths remain relative and portable between machines.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
```

The dependency declarations are stored in `pyproject.toml` and
`requirements.txt`. The local `.venv/` directory is intentionally ignored.

## Locked paper protocol

The retained evaluation scenarios are:

1. `in_distribution`
2. `unseen_small`
3. `unseen_large`
4. `agv_heavy_ratio`
5. `fixed_team_dynamic_orders`
6. `robot_breakdown`

The six trained-policy seeds are `1`, `45`, `789`, `2027`, `4096`, and
`8191`. Learned methods use 1,024,000 environment interactions and are
evaluated for 100 deterministic episodes per seed and scenario. The trained
seed is the independent statistical unit.

The current paper methods are `chord_mappo`, `plato_faithful`,
`mohito_official`, `entity_mappo`, `graph_mappo`, and the
`rolling_horizon` operational comparator. Matched encoder controls and
mechanism variants are retained under `results/mechanism_controls/` and
`results/mechanism_ablations/`.

## Experiment entry points

Run the Core-6 main comparison:

```powershell
python -u scripts/run_eaai_robotics_suite.py `
  --config configs/eaai_core6_cuda.toml `
  --scenario-set core6 `
  --seeds 1 45 789 2027 4096 8191 `
  --section main `
  --output results/main_comparison
```

Run the compute-adapted MOHITO configuration with the same relative output
root:

```powershell
python -u scripts/run_eaai_robotics_suite.py `
  --config configs/eaai_core6_cuda_mohito_512.toml `
  --scenario-set core6 `
  --seeds 1 45 789 2027 4096 8191 `
  --section main `
  --algorithms mohito_official `
  --output results/main_comparison
```

Run the three-scenario mechanism study and retained sensitivity sweeps:

```powershell
python -u scripts/run_eaai_ablation3.py `
  --config configs/eaai_core6_cuda.toml `
  --seeds 1 45 789 2027 4096 8191 `
  --algorithms chord_no_reservation chord_capacity_factorized chord_no_capacity `
  --output results/mechanism_ablations

python -u scripts/run_chord_sensitivity.py `
  --checkpoints-root results/main_comparison/chord_mappo `
  --config configs/cie_eaai_sensitivity.toml `
  --output results/sensitivity/chord
```

## Figure generation

All English artwork is generated below `figs/`:

```powershell
python -u scripts/generate_all_figures.py
```

The Chinese plotting code is an independent copy under `scripts-zh/`. It
reads the same archived numerical results, keeps algorithm names unchanged,
translates visible non-algorithm labels, and writes all artwork below
`figs-zh/`:

```powershell
python -u scripts-zh/generate_all_figures.py
```

Either entry point accepts `--group figure3` through `--group figure9`, or
`--group appendix-b`, `appendix-d`, and `appendix-j`, to regenerate one final
figure group.  The Figure 5 group includes `Figure_5L.pdf`.  The complete
command-to-output mapping is in `scripts/FIGURE_GENERATION_COMMANDS.txt`.

Both pipelines preserve the existing `Figure_*.pdf` naming convention. Their
intermediate PDF/SVG/PNG files and provenance records remain under the
corresponding `paper_artifacts/` and `semantic_analysis/` subdirectories.

## Repository layout

- `open_hyper_warehouse/`: simulator, models, training, evaluation, and analysis.
- `configs/`: retained Core-6, ablation, and sensitivity configurations.
- `scripts/`: experiment and English plotting entry points.
- `scripts-zh/`: Chinese plotting entry points.
- `results/`: current paper checkpoints, evaluations, and aggregate statistics.
- `figs/`: generated English figures.
- `figs-zh/`: generated Chinese figures.
- `external/mohito-public/wildfire/mohito/`: pinned MOHITO runtime package used by the adapter.

The external revision is recorded in
`external/mohito-public/UPSTREAM_SOURCE.txt`. No absolute local filesystem
paths are required by the project entry points.
