#!/usr/bin/env bash
# Run every benchmark then build the figures.
# Results go to results/ as json. Figures go to plots/ as png.
set -e
cd "$(dirname "$0")"

# WOTS
python3 runners/bench_wots_cpu.py
python3 runners/bench_wots_gpu.py

# XMSS
python3 runners/bench_xmss_cpu.py
python3 runners/bench_xmss_gpu.py

# ECDSA reference, writes P-256 for WOTS and P-521 for XMSS
python3 runners/bench_ecdsa.py

# figures
python3 plots/plot_wots.py
python3 plots/plot_xmss.py

echo "done. see results/ for json and plots/ for png"
