# Reproducing the v5 results

Every manuscript-bound number is a row in `outputs/ARTIFACT_MANIFEST.md`, with
the script that makes it and the file it lives in. This page covers setting up
the environment, running the tests, and regenerating rows.

## 1. Environment (fresh clone, fresh virtualenv)

```bash
git clone https://github.com/ArnavPatil-official/JetEngineSimulation.git
cd JetEngineSimulation
git checkout phase4                      # until merged to main
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the v5 environment (Python 3.12.4, macOS arm64). On
Linux or Windows, install the CPU build of torch first (the command is in the
file's header). Nothing needs a GPU.

## 2. Tests

```bash
.venv/bin/python -m pytest tests/ -v                    # 190 passed, 1 skipped at v5
.venv/bin/python scripts/test_emissions.py
.venv/bin/python scripts/validation/verify_protected_hashes.py
```

## 3. Regenerating manifest rows

Outputs are write-once: every pipeline script refuses to overwrite its output.
To regenerate a row, move its committed artifact aside and re-run the command in
the manifest. `scripts/reproduce_check.py` automates this in a **disposable
clone**. It moves each artifact to `reproduce_committed/`, re-runs the command,
and compares the two files: numbers to a relative tolerance of 1e-9, strings
exactly, wall-clock timestamps ignored.

```bash
.venv/bin/python scripts/reproduce_check.py --in-clone            # default rows, ~15 min
.venv/bin/python scripts/reproduce_check.py --in-clone --long     # adds full fit, V2 profile, V8 bands (~3 h more)
```

Rows run in dependency order: V1 pilot → V1 A2 selection → V3, V5, V6 → E3, E4,
E9 → B1 → B2/B3. The long rows are the registered full fit (~17 min), the full
identifiability profile (~90 min) and the P6.2 bands (~60 min). B1 and E9 read
the committed P6.2 bands, so the default set does not need the long rows.

Every stochastic step is seeded (42; the P6.2 draws and CORSIA draws record
their seeds). Solution reuse in Cantera is reset to its as-constructed state per
call, so cycle results are bit-identical to fresh `Solution` objects
(`tests/test_solution_reuse.py`).

## 4. Regenerating the documents

```bash
.venv/bin/python scripts/build_manifest.py          # outputs/ARTIFACT_MANIFEST.md, docs/model_map.md
.venv/bin/python scripts/build_manifest.py --check  # exit 1 if they are stale
```

## Verification record

See the "Fresh-clone verification" entry in `outputs/phase6_execution_status.md`.
