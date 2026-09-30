# `catjet-pycycle` reference environment (Phase 8 D2)

This environment reproduces the published pyCycle 4.4.0 high-bypass turbofan
example as the code-to-code target for P8.4. It is separate from `.venv` and
`catjet-cpp`. The example is never calibration or experimental validation data.

## Recreate

On Apple silicon, create the exact conda package set from
`conda-lock-osx-arm64.txt`, then install the hash-pinned pyCycle wheel:

```sh
~/miniforge3/bin/conda create -n catjet-pycycle --file envs/pycycle/conda-lock-osx-arm64.txt
~/miniforge3/envs/catjet-pycycle/bin/python -m pip install --require-hashes -r envs/pycycle/pip-requirements-lock.txt
```

`environment.yml` describes the same intended versions. `pip-freeze.txt`
is a diagnostic record of the original environment; conda packages there
carry build-host file URLs, so use the explicit conda lock and hash-pinned
wheel above for reproduction. The runner checks the important version pins.

## Reference run

`envs/pycycle/upstream/SOURCES.json` records the upstream tag, commit and
SHA-256 of each unmodified file used. The installed pyCycle package supplies
the matching component maps. Run the wrapper from the repository root:

```sh
~/miniforge3/envs/catjet-pycycle/bin/python scripts/phase8/pycycle/run_hbtf_reference.py
```

The wrapper refuses to overwrite `outputs/phase8/pycycle_hbtf_reference.json`
and `.csv`. It runs the upstream benchmark configuration and the published
example's full sweep, recording native and SI station quantities. The frozen
reference in this repository passed all 26 upstream asserted values, the
upstream pytest test and the upstream testflo test. The full example made 126
`run_model` calls. Six consecutive `OD_full_pwr` calls (indices 6–11) did not
converge in the upstream Newton solver; they are recorded as such in the JSON
and must not be used as successful code-to-code targets. The three-point
benchmark case did converge and is the initial P8.4 comparison target.

Do not edit the upstream files or the write-once reference outputs. If the
reference must be rerun, use `--out-prefix` with a new path and record the
reason and hashes before comparing it with this one.
