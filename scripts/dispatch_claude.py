#!/usr/bin/env python3
"""
Dispatch Claude Code to execute docs/plan.md.

Reads the plan, scores complexity, selects the appropriate Claude model,
and invokes Claude Code in non-interactive mode.

Usage:
    python scripts/dispatch_claude.py              # Execute plan
    python scripts/dispatch_claude.py --dry-run    # Show selection only
    python scripts/dispatch_claude.py --plan PATH  # Use alternate plan file
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ── Model mapping ────────────────────────────────────────────────────────────
MODELS = {
    "low":    "claude-haiku",
    "medium": "claude-sonnet",
    "high":   "claude-opus",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = REPO_ROOT / "docs" / "plan.md"


def read_plan(plan_path: Path) -> str:
    """Read and return plan contents."""
    if not plan_path.exists():
        print(f"ERROR: Plan not found at {plan_path}")
        sys.exit(1)
    return plan_path.read_text()


def score_complexity(plan_text: str) -> tuple[int, list[str]]:
    """
    Score plan complexity on a 0-10 scale.
    Returns (score, list_of_reasons).
    """
    score = 0
    reasons = []

    # 1. Count files touched (READ/MODIFY/CREATE/DELETE rows in table)
    file_rows = re.findall(r"^\|\s*(READ|MODIFY|CREATE|DELETE)\s*\|", plan_text, re.MULTILINE)
    modify_create = [r for r in file_rows if r in ("MODIFY", "CREATE", "DELETE")]
    n_files = len(modify_create)
    if n_files >= 6:
        score += 3
        reasons.append(f"{n_files} files to modify/create/delete (high)")
    elif n_files >= 3:
        score += 2
        reasons.append(f"{n_files} files to modify/create/delete (medium)")
    elif n_files >= 1:
        score += 1
        reasons.append(f"{n_files} file(s) to modify/create/delete (low)")

    # 2. Number of implementation phases
    phases = re.findall(r"^###\s+Phase\s+\d+", plan_text, re.MULTILINE)
    n_phases = len(phases)
    if n_phases >= 4:
        score += 2
        reasons.append(f"{n_phases} implementation phases")
    elif n_phases >= 2:
        score += 1
        reasons.append(f"{n_phases} implementation phases")

    # 3. Multi-subsystem span (combustor, nozzle, turbine, compressor, emissions, PINN)
    subsystems = ["combustor", "nozzle", "turbine", "compressor", "emissions", "pinn"]
    touched = [s for s in subsystems if s in plan_text.lower()]
    if len(touched) >= 3:
        score += 2
        reasons.append(f"Spans {len(touched)} subsystems: {', '.join(touched)}")
    elif len(touched) >= 2:
        score += 1
        reasons.append(f"Spans {len(touched)} subsystems: {', '.join(touched)}")

    # 4. Refactor / debugging language
    refactor_terms = ["refactor", "rewrite", "restructure", "migrate", "overhaul"]
    debug_terms = ["debug", "diagnose", "investigate", "root cause", "bisect"]
    has_refactor = any(t in plan_text.lower() for t in refactor_terms)
    has_debug = any(t in plan_text.lower() for t in debug_terms)
    if has_refactor:
        score += 1
        reasons.append("Contains refactoring language")
    if has_debug:
        score += 1
        reasons.append("Contains debugging/investigation language")

    # 5. Test burden
    test_lines = re.findall(r"test", plan_text, re.IGNORECASE)
    if len(test_lines) >= 10:
        score += 1
        reasons.append("Heavy test burden")

    # Cap at 10
    score = min(score, 10)
    return score, reasons


def select_model(score: int) -> tuple[str, str]:
    """Map complexity score to model tier."""
    if score <= 3:
        return MODELS["low"], "low"
    elif score <= 6:
        return MODELS["medium"], "medium"
    else:
        return MODELS["high"], "high"


def build_prompt(plan_path: Path) -> str:
    """Build the prompt for Claude Code."""
    return (
        f"Read the plan at {plan_path.relative_to(REPO_ROOT)} and execute it completely. "
        "Follow every instruction in the plan. Run validation commands after each phase. "
        "If blocked, stop and explain clearly. Do not deviate from the plan."
    )


def run_claude(model: str, prompt: str, dry_run: bool) -> int:
    """Invoke Claude Code CLI."""
    cmd = [
        "claude",
        "--model", model,
        "--print",
        "--dangerously-skip-permissions",
        "-p", prompt,
    ]

    if dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return 0

    print(f"\n[EXECUTING] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Dispatch Claude Code to execute a plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Path to plan file")
    parser.add_argument("--dry-run", action="store_true", help="Show model selection without executing")
    parser.add_argument("--force-model", choices=["haiku", "sonnet", "opus"], help="Override model selection")
    args = parser.parse_args()

    plan_text = read_plan(args.plan)
    score, reasons = score_complexity(plan_text)
    model, tier = select_model(score)

    if args.force_model:
        model = MODELS.get(
            {"haiku": "low", "sonnet": "medium", "opus": "high"}[args.force_model],
            model,
        )
        tier = args.force_model
        reasons.append(f"Model overridden to {args.force_model}")

    # Print summary
    print("=" * 60)
    print("CLAUDE DISPATCHER")
    print("=" * 60)
    print(f"Plan:       {args.plan}")
    print(f"Score:      {score}/10")
    print(f"Tier:       {tier}")
    print(f"Model:      {model}")
    print(f"Reasons:")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 60)

    prompt = build_prompt(args.plan)
    rc = run_claude(model, prompt, args.dry_run)

    if rc != 0:
        print(f"\n[ERROR] Claude exited with code {rc}")
    else:
        print(f"\n[DONE] Claude completed successfully.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
