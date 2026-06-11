# Make the WOTS figures for the report from the json results.
# Three plots
#   1 per op latency  cpu chain vs gpu kernel vs ecdsa
#   2 gpu batch keygen throughput vs batch size with the cpu line for reference
#   3 key and signature sizes  wots vs ecdsa
# Labels and units are spelled out so each figure reads on its own.

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

OUT = c.PLOTS_DIR
OPS = ["keygen", "sign", "verify"]


def latency_per_op(cpu, gpu, ecdsa):
    series = [
        ("CPU  C chain only", [cpu["timing_us"]["chain_only"][o] for o in OPS]),
        ("GPU  CUDA kernel only", [gpu["single_sig_timing"]["kernel_only_us"][o] for o in OPS]),
        ("ECDSA P-256", [ecdsa["timing_us"][o] for o in OPS]),
    ]
    x = np.arange(len(OPS))
    width = 0.27
    offs = [-width, 0.0, width]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (label, vals), off in zip(series, offs):
        ax.bar(x + off, vals, width, label=label)
        for i in range(len(OPS)):
            ax.annotate("%.0f" % vals[i], (x[i] + off, vals[i]), ha="center",
                        va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_ylabel("time per operation  microseconds  log scale")
    ax.set_xticks(x)
    ax.set_xticklabels(OPS)
    ax.set_title("WOTS compute vs ECDSA P-256  per operation latency  single signature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "latency_per_op.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def gpu_batch_throughput(cpu, gpu):
    rows = gpu["batch_keygen"]
    keys = [r["keys"] for r in rows]
    ker = [r["kernel_keys_per_sec"] for r in rows]
    e2e = [r["end_to_end_keys_per_sec"] for r in rows]
    cpu_kps = 1e6 / cpu["timing_us"]["chain_only"]["keygen"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(keys, ker, "o-", label="GPU  kernel only")
    ax.plot(keys, e2e, "s-", label="GPU  end to end incl host and copies")
    ax.axhline(cpu_kps, color="gray", ls="--", label="CPU  C single core")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size  keys generated in one launch")
    ax.set_ylabel("throughput  keygens per second")
    ax.set_title("WOTS keygen throughput  GPU batching vs single core CPU")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "gpu_batch_throughput.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def sizes(cpu, ecdsa):
    fields = ["signature", "public_key", "private_key"]
    series = [
        ("WOTS  n32 w16", [cpu["sizes_bytes"][f] for f in fields]),
        ("ECDSA P-256", [ecdsa["sizes_bytes"][f] for f in fields]),
    ]
    x = np.arange(len(fields))
    width = 0.35
    offs = [-width / 2, width / 2]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (label, vals), off in zip(series, offs):
        ax.bar(x + off, vals, width, label=label)
        for i in range(len(fields)):
            ax.annotate("%d B" % vals[i], (x[i] + off, vals[i]),
                        ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylabel("size  bytes  log scale")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in fields])
    ax.set_title("WOTS vs ECDSA P-256  key and signature sizes  at matched security")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "sizes.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def main():
    cpu = c.load_json("wots_cpu")
    gpu = c.load_json("wots_gpu")
    ecdsa = c.load_json("ecdsa")
    os.makedirs(OUT, exist_ok=True)
    latency_per_op(cpu, gpu, ecdsa)
    gpu_batch_throughput(cpu, gpu)
    sizes(cpu, ecdsa)


if __name__ == "__main__":
    main()
