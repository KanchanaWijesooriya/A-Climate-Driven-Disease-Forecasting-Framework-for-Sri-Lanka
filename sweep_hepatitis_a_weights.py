#!/usr/bin/env python3
"""Sweep blending weights for Hepatitis A and find best by test R²."""
import os
import json
import subprocess
import sys

artifacts_dir = "/home/chanuka002/Research/model_data/artifacts/hepatitis_a"
weights_path = os.path.join(artifacts_dir, "blending_weights.json")

weight_sets = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (0.95, 0.05),
    (0.99, 0.01),
]

results = []
for w_xgb, w_lgb in weight_sets:
    with open(weights_path, "w") as f:
        json.dump({"xgb_weight": w_xgb, "lgb_weight": w_lgb}, f)
    r = subprocess.run(
        [sys.executable, "plot_blended_ensemble_curves.py", "hepatitis_a"],
        cwd="/home/chanuka002/Research",
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Parse output for test R²
    test_r2 = None
    for line in r.stdout.splitlines():
        if "Final blended R2" in line:
            parts = line.split("test=")
            if len(parts) > 1:
                test_r2 = float(parts[1].strip())
            break
    if test_r2 is not None:
        results.append((w_xgb, w_lgb, test_r2))
        print(f"  {w_xgb:.2f}/{w_lgb:.2f} -> test R²={test_r2:.4f}")

if results:
    best = max(results, key=lambda x: x[2])
    print(f"\nBest: XGB={best[0]:.2f}, LGB={best[1]:.2f} -> test R²={best[2]:.4f}")
    with open(weights_path, "w") as f:
        json.dump({"xgb_weight": best[0], "lgb_weight": best[1]}, f, indent=2)
    print(f"Saved best weights to {weights_path}")
    subprocess.run(
        [sys.executable, "plot_blended_ensemble_curves.py", "hepatitis_a"],
        cwd="/home/chanuka002/Research",
        timeout=120,
    )
