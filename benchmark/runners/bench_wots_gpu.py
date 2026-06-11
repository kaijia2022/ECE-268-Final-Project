# GPU benchmark for WOTS
# The fair number vs the C chain timing is the kernel only time measured with
# cuda events. It excludes the python host work and the host device copies.
# The end to end time is also reported so we can see the launch and copy cost.
# A batch sweep shows how the gpu catches up when many keys run in one launch.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

sys.path.insert(0, os.path.join(c.REPO, "wots_pycuda"))
import wots_pycuda as w
import pycuda.driver as cuda

KERNEL_ITERS = int(os.environ.get("KERNEL_ITERS", "2000"))
E2E_ITERS = int(os.environ.get("E2E_ITERS", "1000"))
BATCH_SIZES = [1, 4, 16, 64, 256, 1024, 4096]

_chain = w._mod.get_function("wots_chain")


def timed_kernel(start_words, steps, iters, block=256):
    # time only the kernel launches with cuda events
    # buffers are filled once so no host device copy is counted
    Nc = len(steps)
    st = steps.astype(np.int32)
    out = np.empty_like(start_words)
    s_gpu = cuda.mem_alloc(start_words.nbytes); cuda.memcpy_htod(s_gpu, start_words)
    st_gpu = cuda.mem_alloc(st.nbytes); cuda.memcpy_htod(st_gpu, st)
    o_gpu = cuda.mem_alloc(out.nbytes)
    grid = (Nc + block - 1) // block
    args = (s_gpu, st_gpu, o_gpu, np.int32(Nc))
    cfg = {"block": (block, 1, 1), "grid": (grid, 1)}
    _chain(*args, **cfg)            # warmup
    cuda.Context.synchronize()
    start = cuda.Event(); end = cuda.Event()
    start.record()
    for _ in range(iters):
        _chain(*args, **cfg)
    end.record(); end.synchronize()
    us = start.time_till(end) / iters * 1000.0
    for m in (s_gpu, st_gpu, o_gpu):
        m.free()
    return us


def timed_e2e(start_words, steps, iters):
    # full host path through run_chain so alloc copy launch copy are all counted
    st = steps.astype(np.int32)
    return c.time_us(lambda: w.run_chain(start_words, st), iters)


def main():
    msg = c.bench_message()
    d = w.H_msg(msg)

    sk = [w.prg(c.BENCH_SEED, i) for i in range(w.LEN)]
    sk_words = w.pack(sk)

    keygen_steps = np.full(w.LEN, w.W - 1, dtype=np.int32)
    sign_steps = np.array(w.msg_digits(d), dtype=np.int32)
    verify_steps = np.array([(w.W - 1) - m for m in w.msg_digits(d)], dtype=np.int32)

    sig = w.unpack(w.run_chain(sk_words, sign_steps), w.LEN)
    sig_words = w.pack(sig)

    single = {
        "kernel_only_us": {
            "keygen": timed_kernel(sk_words, keygen_steps, KERNEL_ITERS),
            "sign": timed_kernel(sk_words, sign_steps, KERNEL_ITERS),
            "verify": timed_kernel(sig_words, verify_steps, KERNEL_ITERS),
        },
        "end_to_end_us": {
            "keygen": timed_e2e(sk_words, keygen_steps, E2E_ITERS),
            "sign": timed_e2e(sk_words, sign_steps, E2E_ITERS),
            "verify": timed_e2e(sig_words, verify_steps, E2E_ITERS),
        },
    }

    # batch keygen throughput
    # K keys means LEN times K chains run in one launch
    batch = []
    for K in BATCH_SIZES:
        batch_words = w.pack(sk * K)
        batch_steps = np.full(w.LEN * K, w.W - 1, dtype=np.int32)
        ker_us = timed_kernel(batch_words, batch_steps, max(50, KERNEL_ITERS // K))
        e2e_us = timed_e2e(batch_words, batch_steps, max(20, E2E_ITERS // K))
        batch.append({
            "keys": K,
            "kernel_us": ker_us,
            "end_to_end_us": e2e_us,
            "kernel_keys_per_sec": K / (ker_us * 1e-6),
            "end_to_end_keys_per_sec": K / (e2e_us * 1e-6),
        })

    data = {
        "scheme": "wots",
        "impl": "pycuda",
        "params": {"n": c.N, "w": c.W, "len": c.LEN},
        "seed": "bench_seed",
        "message_bytes": len(msg),
        "kernel_iters": KERNEL_ITERS,
        "e2e_iters": E2E_ITERS,
        "single_sig_timing": single,
        "batch_keygen": batch,
        "source": "wots_pycuda kernel wots_chain on the gpu",
    }
    c.write_json("wots_gpu", data)


if __name__ == "__main__":
    main()
