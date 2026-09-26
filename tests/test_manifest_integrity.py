"""
P6.8: the manifest and model map are generated from one registry
(scripts/build_manifest.py) and cannot drift; every path they list exists; and
no tracked file under outputs/ is unreferenced (orphan sweep).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_manifest as bm  # noqa: E402


def test_every_listed_path_exists():
    missing = [p for p in bm.all_paths() if not (ROOT / p).exists()]
    assert not missing, missing


def test_documents_equal_the_registry_output():
    assert bm.MANIFEST.read_text() == bm.build_manifest(), "run scripts/build_manifest.py"
    assert bm.MODEL_MAP.read_text() == bm.build_model_map(), "run scripts/build_manifest.py"


def test_every_manifest_row_is_in_the_model_map():
    mm = bm.MODEL_MAP.read_text()
    ids = [r.id for r in bm.ROWS] + [p[0] for p in bm.PINN_ROWS]
    assert len(ids) == len(set(ids))
    for i in ids:
        assert f"| {i} |" in mm, i


def test_no_orphan_outputs():
    tracked = subprocess.run(["git", "ls-files", "outputs"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    listed = bm.all_paths()
    dirs = tuple(p for p in listed if p.endswith("/"))
    files = set(listed)
    orphans = [f for f in tracked if f not in files and not f.startswith(dirs)
               and f != "outputs/ARTIFACT_MANIFEST.md" and not f.endswith(".DS_Store")]
    assert not orphans, orphans
