# Make the XMSS figures for the report from the json results.
# Same three figure set as the WOTS plots
#   1 per op latency  cpu whole op vs gpu kernel vs ecdsa p-521
#   2 gpu keygen throughput vs tree height with the cpu line for reference
#   3 key and signature sizes  xmss vs ecdsa p-521
# Labels and units are spelled out so each figure reads on its own.
# ECDSA here is P-521 because our XMSS sits at the 256 bit classical tier.

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
        ("CPU  C whole op", [cpu["timing_us"][o] for o in OPS]),
        ("GPU  CUDA kernel only", [gpu["single_sig_timing"]["kernel_only_us"][o] for o in OPS]),
        ("ECDSA P-521", [ecdsa["timing_us"][o] for o in OPS]),
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
    ax.set_title("XMSS h 3 compute vs ECDSA P-521  per operation latency  single signature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "xmss_latency_per_op.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def keygen_scaling(cpu, gpu):
    rows = gpu["keygen_scaling"]
    heights = [r["height"] for r in rows]
    ker = [r["kernel_leaves_per_sec"] for r in rows]
    e2e = [r["end_to_end_leaves_per_sec"] for r in rows]
    # the cpu builds one leaf at a fixed cost so its leaves per second is flat
    # derive it from the single height cpu keygen number
    cpu_leaves = (1 << cpu["params"]["height"])
    cpu_lps = cpu_leaves / (cpu["timing_us"]["keygen"] * 1e-6)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(heights, ker, "o-", label="GPU  kernel only")
    ax.plot(heights, e2e, "s-", label="GPU  end to end incl host and copies")
    ax.axhline(cpu_lps, color="gray", ls="--", label="CPU  C single core")

    ax.set_yscale("log")
    ax.set_xticks(heights)
    ax.set_xlabel("tree height  leaves are 2 to the height")
    ax.set_ylabel("throughput  leaves built per second")
    ax.set_title("XMSS keygen throughput  GPU scaling with tree height vs single core CPU")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "xmss_keygen_scaling.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def sizes(cpu, ecdsa):
    fields = ["signature", "public_key", "private_key"]
    series = [
        ("XMSS  h 3 n32 w16", [cpu["sizes_bytes"][f] for f in fields]),
        ("ECDSA P-521", [ecdsa["sizes_bytes"][f] for f in fields]),
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
    ax.set_title("XMSS vs ECDSA P-521  key and signature sizes  at matched security")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "xmss_sizes.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def main():
    cpu = c.load_json("xmss_cpu")
    gpu = c.load_json("xmss_gpu")
    ecdsa = c.load_json("ecdsa_p521")
    os.makedirs(OUT, exist_ok=True)
    latency_per_op(cpu, gpu, ecdsa)
    keygen_scaling(cpu, gpu)
    sizes(cpu, ecdsa)


if __name__ == "__main__":
    main()
