# CPU benchmark for XMSS
# Builds and runs the C baseline then collects its timings.
# xmss_c is a strict byte for byte port of the xmss-cuda kernels using the same
# software SHA256, so CPU vs GPU only differs by the gpu parallelism.
# The whole op timing is the fair compute number here, there is no chain only
# split like WOTS because an XMSS op is the tree build plus the wots chains.
# No python reference is benchmarked on purpose, only C vs GPU.

import os
import re
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

XMSS_C_DIR = os.path.join(c.REPO, "xmss_c")
HEIGHT = int(os.environ.get("XMSS_HEIGHT", "3"))
ITERS = int(os.environ.get("XMSS_ITERS", "200"))


def build():
    subprocess.check_call(["make", "-s", "xmss"], cwd=XMSS_C_DIR)


def run_binary():
    out = subprocess.check_output(
        [os.path.join(XMSS_C_DIR, "xmss"), c.BENCH_MSG_FILE, str(HEIGHT), str(ITERS)]
    ).decode()
    print(out)
    return out


def parse(out):
    # timing lines look like  keygen  8922.219 us per op
    # hash count lines look like  keygen_hashes 26796
    times, hashes = {}, {}
    for line in out.splitlines():
        m = re.match(r"^(\w+)\s+([\d.]+) us per op", line)
        if m:
            times[m.group(1)] = float(m.group(2))
        m = re.match(r"^(\w+)_hashes\s+(\d+)", line)
        if m:
            hashes[m.group(1)] = int(m.group(2))
    return times, hashes


def main():
    build()
    times, hashes = parse(run_binary())
    msg = c.bench_message()

    data = {
        "scheme": "xmss",
        "impl": "c",
        "params": {"n": c.N, "w": c.W, "len": c.LEN, "height": HEIGHT},
        "iters": ITERS,
        "seed": "fixed 0x11 0x22 0x33",
        "message_bytes": len(msg),
        "sizes_bytes": c.xmss_sizes(HEIGHT),
        # per op sha256 counts measured by the binary on this exact message
        "hash_counts": {
            "keygen": {"total_sha256": hashes.get("keygen")},
            "sign": {"total_sha256": hashes.get("sign")},
            "verify": {"total_sha256": hashes.get("verify")},
        },
        # microseconds per op, whole op is the fair compute number vs the kernels
        "timing_us": {
            "keygen": times.get("keygen"),
            "sign": times.get("sign"),
            "verify": times.get("verify"),
        },
        "source": "xmss_c/xmss.c built with -O3 -march=native single thread software sha256",
    }
    c.write_json("xmss_cpu", data)


if __name__ == "__main__":
    main()
