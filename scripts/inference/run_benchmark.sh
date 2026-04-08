#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/5] Starting Topo-A* Benchmark..."
python benchmarks/run_astar.py

echo "[2/5] Starting Topo-RRT Benchmark..."
python benchmarks/run_rrt.py

echo "[3/5] Starting MPD Benchmark..."
python benchmarks/run_mpd.py

echo "[4/5] Starting TMPD (Ours) Benchmark..."
python benchmarks/run_tmpd.py

echo "[5/5] All algorithms finished! Generating 1x4 Comparison Figures..."
python plotting/plot_all.py

echo "All tasks completed successfully!"
