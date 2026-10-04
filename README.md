# CAT-JET SAF blend screening

The screening API predicts fuel flow, T4, EI-CO2, lifecycle CO2e and in-domain
Brem nvPM changes from blend mass fractions and thrust. Predictions are
**conditional on v6 calibration**. The surrogate measures fidelity to the
v6 C++ simulator and computational speed; real-world accuracy is inherited
from v6. The historical A1 failure on the penalty guard remains disclosed.

The registered post-chain workflow creates the saved model bundle shown
below. Numerical work starts only after the original AC chain releases its lease.
The default product loader requires passing
fidelity, ranking and measured-speed gates and verified model hashes.
A fully computed failed study retains diagnostics and does not enable the tool.

## CLI

From the repository root, after successful product verification:

```sh
.venv/bin/python scripts/phase8/screen_blends.py --model outputs/phase8/saf_surrogate/attempt_001/product.json --grid-step 0.1 --thrust-fraction 1.0 --out outputs/phase8/screening_tool/grid.json
```

Candidate JSON files use IDs and nonnegative `JetA`, `HEFA`, `FT`, `ATJ`
mass fractions summing to one; supply them with `--candidates candidates.json`
in place of `--grid-step`. Thrust fractions range from 0.07 to 1.0.
Use `--verify-top-k 10 --verification-out outputs/phase8/screening_tool/verify_001`
to verify selected candidates with the simulator. Verification requires idle
AC power and a fresh output directory. Existing results are never overwritten.

## Python API

```python
from scripts.phase8.screening_product import screen_blends

result = screen_blends(
    [{"id": "half_hefa", "JetA": 0.5, "HEFA": 0.5, "FT": 0.0, "ATJ": 0.0}],
    model="outputs/phase8/saf_surrogate/attempt_001/product.json",
    thrust_fraction=1.0,
)
```

Outputs include bands across the 64 fixed parameter draws, separate seed
spread and training-envelope flags. Draw bands are sensitivity bands, not
confidence intervals. Invalid inputs receive flags and missing predictions.
Nominal inputs outside the empirical training envelope stay visibly flagged.
Brem nvPM is a relative change only inside its registered domain; it is
unavailable elsewhere. Exact fuel-property and CO2/lifecycle relations are
computed from composition and fuel flow.

Procedures and numerical bars are in the prospective registrations under
`docs/`. Quantitative freeze artifacts belong in `outputs/freeze/`; no results
prose or remote publication is generated.
