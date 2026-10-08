# EAAI result-data contract

## Statistical unit

Each checkpoint first yields one scenario mean over its evaluation episodes.
Paper-level uncertainty is then computed across the six independently trained
policies. Evaluation episodes must not be treated as independent training runs.

## Required evidence groups

1. Main comparison across six complete decision systems and eight scenarios.
2. Decoder-matched CHORD, Entity+CAHR, and Graph+CAHR encoder controls.
3. Three-scenario decoder-package and capacity-context study.
4. CHORD-only failure and demand dose-response evaluation.
5. Six-seed semantic edge-masking diagnostics for `robot_breakdown`.
6. Order accounting: created/admitted, delivered, cancelled, expired, and
   pending-at-termination counts where available.

## Metric definitions

| Field | Interpretation |
|---|---|
| `service_level` | Deliveries divided by eligible admitted orders |
| `throughput_per_100_steps` | Deliveries per 100 simulator counter ticks |
| `deadline_miss_rate` | Expired eligible orders divided by eligible admitted orders |
| `mean_flow_time` | Mean creation-to-delivery counter ticks among delivered orders |
| `contention_per_delivery` | Rejected duplicate assignment proposals per delivery |
| `travel_per_delivery` | Grid moves per delivery |

Contention is an assignment conflict and must not be described as a physical
collision. Flow-time and throughput units are simulator counter ticks, not
calibrated seconds.

## Source-of-truth order

1. Episode CSV and training JSONL records.
2. Per-seed evaluation summaries.
3. Derived aggregate CSV files and paper tables.
4. Figures generated from the aggregate files.

Derived files must be reproducible from the raw records. Never hand-edit a
paper number or derived CSV.

## Result layout

- `results/main_comparison/`: six main systems.
- `results/mechanism_controls/`: Entity+CAHR and Graph+CAHR.
- `results/mechanism_ablations/`: three mechanism variants and aggregates.
- `results/sensitivity/`: failure and demand dose-response records.
- `results/semantic_analysis/`: edge masking and case timelines.
- `results/paper_artifacts/`: publication tables and figures.

Parameter counts and device-independent complexity may be compared across
methods. Wall-clock latency from different machines must not be used as a
cross-method ranking.
