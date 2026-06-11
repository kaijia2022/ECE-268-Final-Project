# GPU benchmark for XMSS
# xmss-cuda is pure CUDA so unlike WOTS there is no pycuda module to call from
# python. Instead we build and run xmss-cuda/bench.cu which times the kernels
# the same way bench_wots_gpu does, just inside a compiled program.
# The fair number vs the C whole op is the kernel only time measured with cuda
# events. It excludes the host malloc and the host device copies. The end to end
# time is also reported so we can see the launch and copy cost.
# The keygen scaling sweep is the XMSS analog of the WOTS batch sweep, it shows
# how the gpu pulls ahead as a taller tree feeds it more parallel leaves.

import os
import re
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

XMSS_CUDA_DIR = os.path.join(c.REPO, "xmss-cuda")
BENCH_BIN = os.path.join(XMSS_CUDA_DIR, "build", "bench")

HEIGHT = int(os.environ.get("XMSS_HEIGHT", "3"))
KERNEL_ITERS = int(os.environ.get("XMSS_KERNEL_ITERS", "200"))
SCALE_HEIGHTS = os.environ.get("XMSS_SCALE_HEIGHTS", "3,5,7,10")


def build():
    subprocess.check_call(["make", "bench"], cwd=XMSS_CUDA_DIR,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_binary():
    out = subprocess.check_output(
        [BENCH_BIN, c.BENCH_MSG_FILE, str(HEIGHT), str(KERNEL_ITERS), SCALE_HEIGHTS]
    ).decode()
    print(out)
    return out


def parse(out):
    single = {"kernel_only_us": {}, "end_to_end_us": {}}
    scaling = []
    for line in out.splitlines():
        # single signature latency lines  kernel_keygen_us 2569.236
        m = re.match(r"^kernel_(\w+)_us\s+([\d.]+)", line)
        if m:
            single["kernel_only_us"][m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"^e2e_(\w+)_us\s+([\d.]+)", line)
        if m:
            single["end_to_end_us"][m.group(1)] = float(m.group(2))
            continue
        # scaling sweep lines, pull every key value pair off the line
        if line.startswith("SCALE "):
            kv = dict(zip(line.split()[1::2], line.split()[2::2]))
            leaves = int(kv["leaves"])
            ker_total = float(kv["kernel_total_us"])
            e2e = float(kv["e2e_us"])
            scaling.append({
                "height": int(kv["height"]),
                "leaves": leaves,
                "kernel_leaves_us": float(kv["kernel_leaves_us"]),
                "kernel_root_us": float(kv["kernel_root_us"]),
                "kernel_total_us": ker_total,
                "end_to_end_us": e2e,
                # throughput in leaves per second, the invariant unit across heights
                "kernel_leaves_per_sec": leaves / (ker_total * 1e-6),
                "end_to_end_leaves_per_sec": leaves / (e2e * 1e-6),
            })
    return single, scaling


def main():
    build()
    single, scaling = parse(run_binary())
    msg = c.bench_message()

    data = {
        "scheme": "xmss",
        "impl": "cuda",
        "params": {"n": c.N, "w": c.W, "len": c.LEN, "height": HEIGHT},
        "seed": "fixed 0x11 0x22 0x33",
        "message_bytes": len(msg),
        "kernel_iters": KERNEL_ITERS,
        "single_sig_timing": single,
        "keygen_scaling": scaling,
        "source": "xmss-cuda/bench.cu kernels timed with cuda events on the gpu",
    }
    c.write_json("xmss_gpu", data)


if __name__ == "__main__":
    main()
