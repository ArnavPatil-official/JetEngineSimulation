# Report Generation Tools

This directory contains scripts for generating comprehensive code documentation and reports.

## Scripts

### `generate_outputs.py`

Generates curated demo outputs for inclusion in the Full Report:
- Runs combustor simulations with different fuels (Jet-A1, SAF blends)
- Runs compressor example calculations
- Creates comparison plots and saves to `outputs/` directory
- Exports numeric results to JSON

**Usage:**
```bash
python tools/generate_outputs.py
```

**Requirements:**
- Cantera (install via: `conda install -c conda-forge cantera` or `pip install cantera`)
- NumPy, Matplotlib
- Mechanism file: `data/creck_c1c16_full.yaml`

**Note:** If the mechanism file is missing, the script will print a warning and exit gracefully.

### `make_report.py`

Creates a comprehensive Markdown report containing:
- All Python source files in the repository (with syntax highlighting)
- Demo outputs from `generate_outputs.py` (stdout/stderr)
- Generated figures embedded in the report

Optionally converts the Markdown report to PDF using Pandoc.

**Usage:**
```bash
# Generate Markdown report only
python tools/make_report.py

# Generate both Markdown and PDF
python tools/make_report.py --pdf
```

**Requirements:**
- Python standard library (no extra dependencies for Markdown generation)
- For PDF generation:
  - Pandoc (`apt install pandoc` or download from https://pandoc.org/)
  - LaTeX engine (xelatex, lualatex, or pdflatex)

**Output Files:**
- `Full_Report.md` - Complete Markdown documentation
- `Full_Report.pdf` - PDF version (if `--pdf` flag used and Pandoc available)

## Features

- **Safe execution:** Runs demos with timeout protection
- **Non-GUI plotting:** Uses Agg backend for matplotlib
- **Graceful degradation:** Warns about missing dependencies instead of failing
- **Selective file inclusion:** Automatically excludes common directories (.git, __pycache__, venv, etc.)
- **Timestamp tracking:** Each report includes generation timestamp

## Examples

```bash
# Quick test - just generate outputs
python tools/generate_outputs.py

# Create full documentation report
python tools/make_report.py

# Create report with PDF export
python tools/make_report.py --pdf

# Or use environment variable for PDF
REPORT_PDF=1 python tools/make_report.py
```
