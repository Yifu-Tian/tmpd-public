#!/bin/bash

echo "🚀 [1/5] Starting Topo-A* Benchmark..."
python run_astar.py

echo "🚀 [2/5] Starting Topo-RRT Benchmark..."
python run_rrt.py

echo "🚀 [3/5] Starting Vanilla MPD Benchmark..."
python run_mpd.py

echo "🚀 [4/5] Starting TMPD (Ours) Benchmark..."
python run_tmpd.py

echo "🎨 [5/5] All algorithms finished! Generating 1x4 Comparison Figures..."
python plot_all_figures.py

echo "✅ All tasks completed successfully!"