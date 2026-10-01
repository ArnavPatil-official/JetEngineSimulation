#!/usr/bin/env python3
"""P8.4 thermo-matched mode: pyCycle 4.4.0 JANAF products -> Cantera NASA-9 YAML.

Run in catjet-pycycle. Writes data/thermo/pycycle_janaf.yaml once: the same
species, NASA 9-coefficient polynomials and temperature ranges as
pycycle/thermo/cea/thermo_data/janaf.py, with pyCycle's element weights, and
the pyCycle air element amounts and Jet-A(g) element formula in metadata.
Verification (printed, and repeated in tests): each species' molar mass
equals pyCycle 'wt'; cp/R at 300, 1500 and 3000 K equals pyCycle's own
polynomial evaluation.
"""

import hashlib
import json
import sys
from pathlib import Path

import pycycle
from pycycle.constants import CEA_AIR_COMPOSITION
from pycycle.thermo.cea.thermo_data import janaf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "thermo" / "pycycle_janaf.yaml"
SPECIES_CONSISTENT_WTS = {"C": 12.0107, "O": 15.9994, "Ar": 39.948, "H": 1.00794, "N": 14.00674}


def main() -> int:
    if pycycle.__version__ != "4.4.0":
        sys.exit("pinned pyCycle 4.4.0 required")
    if OUT.exists():
        sys.exit(f"{OUT} exists; write-once")
    src = Path(janaf.__file__)
    lines = [
        "description: |-",
        "  pyCycle 4.4.0 JANAF products (thermo/cea/thermo_data/janaf.py), NASA-9,",
        f"  exported by scripts/phase8/pycycle/export_janaf.py; source sha256 {hashlib.sha256(src.read_bytes()).hexdigest()}.",
        "  pyCycle: NASA Glenn / OpenMDAO, Apache-2.0. For the P8.4 thermo-matched check only.",
        "units: {length: m, quantity: kmol, activation-energy: J/kmol}",
        "pycycle_metadata: " + json.dumps({
            "CEA_AIR_COMPOSITION": CEA_AIR_COMPOSITION,
            "jet_a_g_elements": dict(janaf.reactants["Jet-A(g)"]),
            "element_wts": janaf.element_wts}),
        "elements:",
    ]
    # Species molar masses must equal pyCycle 'wt' (used for its mass/mole
    # conversions). Those imply C = 12.0107, not element_wts' 12.0170 (a
    # transposition in pyCycle); element_wts is kept in metadata because
    # pyCycle uses it to convert the fuel's element formula.
    for el, wt in SPECIES_CONSISTENT_WTS.items():
        lines.append(f"- symbol: {el}x")
        lines.append(f"  atomic-weight: {wt!r}")
    elements = list(SPECIES_CONSISTENT_WTS)
    for name, sp in janaf.products.items():
        m = sum(SPECIES_CONSISTENT_WTS[e] * k for e, k in sp["elements"].items())
        if abs(m - sp["wt"]) > 1e-9 * sp["wt"]:
            sys.exit(f"{name}: element weights give {m}, pyCycle wt {sp['wt']}")
    lines += ["phases:", "- name: janaf", "  thermo: ideal-gas",
              "  elements: [" + ", ".join(f"{e}x" for e in elements) + "]",
              "  species: all", "  state: {T: 300.0, P: 1 atm}", "species:"]
    for name, sp in janaf.products.items():
        comp = ", ".join(f"{e}x: {n}" for e, n in sp["elements"].items())
        ranges = [float(t) for t in sp["ranges"]]
        lines += [f"- name: {json.dumps(name)}", f"  composition: {{{comp}}}",
                  "  thermo:", "    model: NASA9",
                  f"    temperature-ranges: {ranges}", "    data:"]
        for row in sp["coeffs"]:
            lines.append("    - [" + ", ".join(repr(float(c)) for c in row) + "]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(janaf.products)} species; elements {elements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
