# WOTS and XMSS benchmark

Run everything and build the figures:

```
bash run_all.sh
```

Or run one step at a time:

```
python3 runners/bench_wots_cpu.py   # C baseline, writes results/wots_cpu.json
python3 runners/bench_wots_gpu.py   # CUDA kernel, writes results/wots_gpu.json
python3 runners/bench_xmss_cpu.py   # C baseline, writes results/xmss_cpu.json
python3 runners/bench_xmss_gpu.py   # CUDA kernels, writes results/xmss_gpu.json
python3 runners/bench_ecdsa.py      # ECDSA P-256 and P-521, writes results/ecdsa*.json
python3 plots/plot_wots.py          # reads results/*.json, writes plots/*.png
python3 plots/plot_xmss.py          # reads results/*.json, writes plots/*.png
```

The fair CPU vs GPU number is the CPU software SHA compute vs the GPU CUDA kernel
only time. ECDSA is a size and speed reference at matched security: P-256 for
WOTS at 128 bit, P-521 for XMSS at 256 bit.
