"""Check that two training runs produced identical results.

Usage:
    python scripts/compare_metrics.py reports/metrics_before.json reports/metrics.json

Exits with code 1 if the selected model, its parameters or any metric differ.
Run-specific fields (such as the MLflow run id) are ignored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE = 1e-6
COMPARED = ["selected_model", "best_params", "cv_metrics", "test_metrics"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def main(old_path: str, new_path: str) -> int:
    old = json.loads(Path(old_path).read_text())
    new = json.loads(Path(new_path).read_text())
    old_flat = flatten({k: old.get(k) for k in COMPARED})
    new_flat = flatten({k: new.get(k) for k in COMPARED})

    diffs = []
    for key in sorted(set(old_flat) | set(new_flat)):
        a, b = old_flat.get(key), new_flat.get(key)
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > TOLERANCE:
                diffs.append(f"  {key}: {a:.6f} -> {b:.6f}")
        elif a != b:
            diffs.append(f"  {key}: {a!r} -> {b!r}")

    if diffs:
        print("NOT reproducible. Differences:")
        print("\n".join(diffs))
        return 1
    print(f"Reproducible: {len(new_flat)} values identical "
          f"(model: {new['selected_model']}, test ROC-AUC: {new['test_metrics']['roc_auc']:.4f})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
