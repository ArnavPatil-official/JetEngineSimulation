#!/usr/bin/env python3
"""
Create a single Markdown file (Full_Report.md) containing:
- All .py source code in the repo (excluding common ignores)
- Outputs from tools/generate_outputs.py (stdout + saved figures)

Optionally convert to PDF via Pandoc/LaTeX:
  python tools/make_report.py --pdf

Usage:
  python tools/make_report.py
  python tools/make_report.py --pdf
"""

import os, sys, subprocess, shutil
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = REPO_ROOT / "Full_Report.md"
OUT_PDF = REPO_ROOT / "Full_Report.pdf"
OUTPUTS_DIR = REPO_ROOT / "outputs"

IGNORES = {
    ".git", ".github", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    "data/raw", "data/processors", "build", "dist"
}

def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if any(ig in parts for ig in IGNORES):
        return False
    return path.suffix == ".py"

def collect_py_files():
    return sorted([p for p in REPO_ROOT.rglob("*.py") if should_include(p)])

def run_generate_outputs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "generate_outputs.py")]
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                              capture_output=True, text=True, timeout=600)
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        stdout, stderr = "", "[ERROR] generate_outputs timed out."
    return stdout, stderr

def write_header(md):
    md.write(f"# JetEngineSimulation — Full Code and Outputs\n\n")
    md.write(f"- Generated: {datetime.now().astimezone().isoformat()}\n")
    md.write(f"- Repo: {REPO_ROOT.name}\n\n")
    md.write("---\n\n")

def embed_code(md, files):
    md.write("## Source Code Listing\n\n")
    for p in files:
        rel = p.relative_to(REPO_ROOT)
        md.write(f"### {rel}\n\n")
        md.write("```python\n")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            text = f"# [ERROR] Could not read file: {e}\n"
        md.write(text)
        md.write("\n```\n\n")

def embed_outputs(md, stdout, stderr):
    md.write("## Generated Outputs (stdout)\n\n")
    md.write("```text\n")
    md.write(stdout if stdout else "[no stdout]\n")
    md.write("\n```\n\n")
    if stderr.strip():
        md.write("### stderr\n\n```text\n")
        md.write(stderr)
        md.write("\n```\n\n")

    # Embed any images saved under outputs/
    imgs = sorted(list(OUTPUTS_DIR.glob("*.png")))
    if imgs:
        md.write("## Figures\n\n")
        for img in imgs:
            rel = img.relative_to(REPO_ROOT)
            md.write(f"![{rel}]({rel})\n\n")

def pandoc_to_pdf():
    if shutil.which("pandoc") is None:
        print("[INFO] pandoc not found; skipping PDF. Install pandoc to enable PDF export.")
        return False
    # Choose LaTeX engine if available
    engine = None
    for e in ["xelatex", "lualatex", "pdflatex"]:
        if shutil.which(e):
            engine = e
            break
    args = ["pandoc", str(OUT_MD), "-o", str(OUT_PDF)]
    if engine:
        args.extend(["--pdf-engine", engine])
    print("[INFO] Running:", " ".join(args))
    try:
        subprocess.check_call(args, cwd=str(REPO_ROOT))
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] pandoc failed: {e}")
        return False

def main():
    py_files = collect_py_files()
    stdout, stderr = run_generate_outputs()

    with open(OUT_MD, "w", encoding="utf-8") as md:
        write_header(md)
        embed_code(md, py_files)
        embed_outputs(md, stdout, stderr)

    print(f"[OK] Wrote {OUT_MD}")

    if "--pdf" in sys.argv or os.environ.get("REPORT_PDF") == "1":
        if pandoc_to_pdf():
            print(f"[OK] Wrote {OUT_PDF}")
        else:
            print("[INFO] PDF generation not completed. See logs above.")

if __name__ == "__main__":
    main()
