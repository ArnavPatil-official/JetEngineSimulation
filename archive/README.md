# archive/

Code and third-party material outside the manifest pipeline (Phase 6, P6.8).
Nothing here is imported by production code or tests. Every entry is a
`git mv`; reverse with the opposite `git mv`.

| Old path | New path | Why |
|---|---|---|
| `simulation/emissions.py` | `archive/code/simulation/emissions.py` | Not imported by the pipeline; its own docstring marks its constants as invented. Production emissions: `EmissionsEstimator` in `integrated_engine.py`. |
| `scripts/visualization/pareto_visual.py` | `archive/code/scripts/visualization/pareto_visual.py` | Figures for the superseded unequal-thrust blend studies. |
| `scripts/visualization/visualize_results.py` | `archive/code/scripts/visualization/visualize_results.py` | Same. |
| `dashboard.py` | `archive/code/dashboard.py` | Exploratory dashboard; not in the manifest. |
| `fetch_and_build_cfd_data.py` | `archive/code/fetch_and_build_cfd_data.py` | Superseded dataset builder; `scripts/parse_sajben_cfd.py` writes `data/processed/master_shock_dataset.pt`. |
| `data/raw/cfd_datasets/github/nozzle_flow_cfd-main/` | `archive/third_party/nozzle_flow_cfd-main/` | Vendored generic SU2 nozzle repository; not used by any route (see `scripts/validation/sajben_data_audit.py`). No licence file was included in the vendored copy. |
| `data/raw/cfd_datasets/github/nozzle_flow_cfd.zip` | `archive/third_party/nozzle_flow_cfd.zip` | Zip of the same repository. |
