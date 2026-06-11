#!/usr/bin/env bash
# Run every WOTS benchmark then build the figures.
# Results go to results/ as json. Figures go to plots/ as png.
set -e
cd "$(dirname "$0")"

python3 runners/bench_wots_cpu.py
python3 runners/bench_wots_gpu.py
python3 runners/bench_ecdsa.py
python3 plots/plot_wots.py

echo "done. see results/ for json and plots/ for png"
