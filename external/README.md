# External baseline sources

`mohito-public` contains the runtime subset of the MOHITO authors' public implementation, pinned by
`scripts/fetch_official_baselines.ps1` to commit
`105ddccbacc92953a7aa6bd0d19f8ec299f7476d`. Its wildfire Python package
declares AGPL-3.0-or-later and therefore remains a separate external checkout;
it is not relicensed under this project's MIT license.

The warehouse code imports the official `Actor`, `Critic`, and `GNN` classes at
runtime. Unused domains, notebooks, examples, and upstream test drivers are
not included. `open_hyper_warehouse/mohito_official.py` contains only the warehouse
observation conversion, inactive-slot handling, replay loop, and checkpoint
adapter used in this study. Nested version-control metadata is not distributed;
the exact upstream revision is recorded in `mohito-public/UPSTREAM_SOURCE.txt`.
