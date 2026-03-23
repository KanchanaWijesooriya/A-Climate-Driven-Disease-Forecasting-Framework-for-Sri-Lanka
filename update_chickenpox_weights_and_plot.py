#!/usr/bin/env python3
"""
Update Chickenpox blend weights and regenerate the blended curves plot.
Usage: python update_chickenpox_weights_and_plot.py <xgb_weight> <lgb_weight>
Example: python update_chickenpox_weights_and_plot.py 0.1 0.9
"""
import json
import os
import subprocess
import sys

artifacts_dir = "/home/chanuka002/Research/model_data/artifacts/chickenpox"
script_dir = "/home/chanuka002/Research"

if len(sys.argv) != 3:
    print("Usage: python update_chickenpox_weights_and_plot.py <xgb_weight> <lgb_weight>")
    print("Example: python update_chickenpox_weights_and_plot.py 0.1 0.9")
    sys.exit(1)

try:
    w_xgb = float(sys.argv[1])
    w_lgb = float(sys.argv[2])
except ValueError:
    print("Weights must be numbers (e.g. 0.1 0.9)")
    sys.exit(1)

# Update blending_weights.json
weights_path = os.path.join(artifacts_dir, "blending_weights.json")
with open(weights_path, "w") as f:
    json.dump({"xgb_weight": w_xgb, "lgb_weight": w_lgb}, f, indent=2)
print(f"Updated weights: XGB={w_xgb}, LGB={w_lgb}")

# Run plot script
plot_script = os.path.join(script_dir, "plot_chickenpox_blended_curves.py")
result = subprocess.run(
    [sys.executable, plot_script],
    cwd=script_dir,
)
sys.exit(result.returncode)
