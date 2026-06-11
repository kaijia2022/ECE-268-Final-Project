# CPU benchmark for WOTS
# Builds and runs the C baseline then collects its timings.
# The fair compute number is the chain only timing since that is the exact
# work the cuda kernel does. The whole op timing is kept for context.
# The python baseline in wots_baseline is not benchmarked on purpose.

import os
import re
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

WOTS_C_DIR = os.path.join(c.REPO, "wots_c")
ITERS = int(os.environ.get("ITERS", "5000"))


def build():
    subprocess.check_call(["make", "-s"], cwd=WOTS_C_DIR)


def run_binary():
    out = subprocess.check_output(
        [os.path.join(WOTS_C_DIR, "wots"), c.BENCH_MSG_FILE, str(ITERS)]
    ).decode()
    print(out)
    return out


def parse(out):
    # lines look like  keygen  339.221 us per op  or  chain_sign  120.5 us per op
    times = {}
    for line in out.splitlines():
        m = re.match(r"^(\w+)\s+([\d.]+) us per op", line)
        if m:
            times[m.group(1)] = float(m.group(2))
    return times


def main():
    build()
    times = parse(run_binary())
    msg = c.bench_message()

    data = {
        "scheme": "wots",
        "impl": "c",
        "params": {"n": c.N, "w": c.W, "len": c.LEN},
        "iters": ITERS,
        "seed": "bench_seed",
        "message_bytes": len(msg),
        "sizes_bytes": c.SIZES_BYTES,
        "hash_counts": c.hash_counts(msg),
        # microseconds per op
        "timing_us": {
            # chain only is the fair compute number vs the cuda kernel
            "chain_only": {
                "keygen": times.get("chain_keygen"),
                "sign": times.get("chain_sign"),
                "verify": times.get("chain_verify"),
            },
            # whole op also counts seed derive and H_msg
            "whole_op": {
                "keygen": times.get("keygen"),
                "sign": times.get("sign"),
                "verify": times.get("verify"),
            },
        },
        "source": "wots_c/wots.c built with -O3 -march=native single thread",
    }
    c.write_json("wots_cpu", data)


if __name__ == "__main__":
    main()
