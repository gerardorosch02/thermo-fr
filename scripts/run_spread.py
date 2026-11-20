r"""Launcher of the spread logger from a code snapshot: runs this folder's own src/, whatever is installed in the environment.

    python <snapshot>\scripts\run_spread.py selftest
    python <snapshot>\scripts\run_spread.py dry-run
    python <snapshot>\scripts\run_spread.py issue
    python <snapshot>\scripts\run_spread.py status

The logger's folder (static/, live/) is the parent of the snapshot. The production folder is read only; it defaults to
..\thermo-fr\thermo-fr beside the logger's folder and can be given as a second argument.
"""
import sys
from pathlib import Path

snapshot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(snapshot / "src"))  # ahead of any installed copy of thermo_fr

from thermo_fr.forecast import spread_live  # noqa: E402

root = snapshot.parent
production = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parent / "thermo-fr" / "thermo-fr"
command = sys.argv[1] if len(sys.argv) > 1 else "status"
if not Path(spread_live.__file__).resolve().is_relative_to(snapshot):
    sys.exit(f"thermo_fr was imported from {spread_live.__file__}, not from this snapshot")
problems = [str(p) for p in (root / "static" / "inputs.csv", root / "static" / "reference.json", root / "static" / "raw_first_test") if not p.exists()]
if command in ("issue", "dry-run"):
    problems += [str(p) for p in (production / "data" / "shadow" / "fresh", production / "published" / "model" / "wind_proxy.json") if not p.exists()]
if problems:
    print("Not found, so nothing was run:")
    for problem in problems:
        print("  " + problem)
    sys.exit(2)
sys.exit(spread_live.main([command, "--root", str(root), "--production", str(production)]))
