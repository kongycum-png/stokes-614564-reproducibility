"""Verify downloaded file integrity; this is not a scientific validation."""
from pathlib import Path
import argparse
import hashlib
import json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-only", action="store_true", help="Check Git-tree files; skip release-only NPZ arrays")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "DISTRIBUTION_MANIFEST.json").read_text())
    missing = []; mismatched = []; checked = 0; skipped = 0
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        if args.git_only and rel.suffix == ".npz" and rel.parts[0] != "figure_data":
            skipped += 1
            continue
        p = root / rel
        if not p.is_file():
            missing.append(rel.as_posix())
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024**2), b""):
                h.update(block)
        checked += 1
        if p.stat().st_size != entry["bytes"] or h.hexdigest() != entry["sha256"]:
            mismatched.append(rel.as_posix())
    result = {"version": manifest["version"], "checked": checked, "skipped_release_arrays": skipped,
              "missing": missing, "mismatched": mismatched, "scientific_validation": False}
    print(json.dumps(result, indent=2))
    return 1 if missing or mismatched else 0

if __name__ == "__main__":
    raise SystemExit(main())
