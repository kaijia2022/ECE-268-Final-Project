# WOTS benchmark

Run everything and build the figures:

```
bash run_all.sh
```

Or run one step at a time:

```
python3 runners/bench_wots_cpu.py   # C baseline, writes results/wots_cpu.json
python3 runners/bench_wots_gpu.py   # CUDA kernel, writes results/wots_gpu.json
python3 runners/bench_ecdsa.py      # ECDSA P-256 reference, writes results/ecdsa.json
python3 plots/plot_wots.py          # reads results/*.json, writes plots/*.png
```

The fair CPU vs GPU number is CPU C chain only vs GPU CUDA kernel only.
ECDSA is only a size and speed reference. XMSS is not benchmarked yet.
