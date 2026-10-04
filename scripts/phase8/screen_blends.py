"""Screen SAF candidates with a validated conditional v6 surrogate bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase8.screening_product import screen_blends


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path, help="JSON list of candidate mass-fraction objects")
    source.add_argument("--grid-step", type=float)
    p.add_argument("--thrust-fraction", type=float, default=1.0)
    p.add_argument("--verify-top-k", type=int, default=0)
    p.add_argument("--verification-out", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("Output exists; refusing to overwrite")
    candidates = json.loads(args.candidates.read_text()) if args.candidates else None
    result = screen_blends(candidates, model=args.model, thrust_fraction=args.thrust_fraction,
                           grid_step=args.grid_step, verify_top_k=args.verify_top_k,
                           verification_out=args.verification_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"candidates": len(result["candidates"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
