# Digitised figures (WebPlotDigitizer)

Three independent repeats per curve, laid out as
`<source_id>/<figure>/<curve>_r{1,2,3}.csv` plus `<figure>_r{1,2,3}.json`
(one WPD project per repeat) and a `NOTES.txt` point inventory.

How to digitise, which figures, and naming: [../DIGITISING.md](../DIGITISING.md).
Check one curve: `.venv/bin/python scripts/phase8/wpd_import.py check <this dir>/<source_id>/<figure> <curve>`.
Import: `scripts/phase8/wpd_import.py` (`enter_digitised_curve`), called from an
entry module in `scripts/phase8/empirical_entries/`. Never edit these files by
hand; re-digitise instead.
