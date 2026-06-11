# Shared helpers for the WOTS benchmarks
# Holds the WOTS params, the fixed bench message, size and hash count math,
# a tiny timer, env capture and a json writer. Keep all output as json.

import os
import json
import math
import time
import hashlib

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

# the same message all implementations sign so the numbers line up
BENCH_MSG_FILE = os.path.join(REPO, "wots_pycuda", "plaintext", "short.txt")

# the fixed seed the bench uses so keys are reproducible
BENCH_SEED = b"bench_seed"

# ---------- WOTS params ----------
N = 32
W = 16
LEN1 = math.ceil(8 * N / math.log2(W))          # 64
LEN2 = math.floor(math.log2(LEN1 * (W - 1)) / math.log2(W)) + 1  # 3
LEN = LEN1 + LEN2                               # 67

# every key and the signature is LEN atoms of N bytes
SIZES_BYTES = {
    "private_key": LEN * N,
    "public_key": LEN * N,
    "signature": LEN * N,
}


# ---------- XMSS sizes ----------
# XMSS sizes depend on the tree height so they are a function not a constant.
# signature  = leaf_idx 4 + randomness r N + wots_sig LEN*N + auth_path height*N
# public key = the merkle root, N bytes
# private key = three seeds 3*N plus the 4 byte next_idx state index

def xmss_sizes(height):
    return {
        "private_key": 3 * N + 4,
        "public_key": N,
        "signature": 4 + N + LEN * N + height * N,
    }


def bench_message():
    with open(BENCH_MSG_FILE, "rb") as f:
        return f.read()


# ---------- WOTS digit math ----------
# same base_w and checksum as the python baseline and the cuda kernel

def base_w(x, out_len):
    digits = []
    for byte in x:
        digits.append(byte >> 4)
        digits.append(byte & 0x0f)
    return digits[:out_len]


def H_msg(message):
    # toByte(2, 32) is 31 zero bytes then a 2, matches RFC 8391 and the baseline
    return hashlib.sha256((2).to_bytes(N, "big") + message).digest()


def msg_digits(digest32):
    msg = base_w(digest32, LEN1)
    csum = sum((W - 1) - d for d in msg)
    csum <<= (8 - ((LEN2 * 4) % 8)) % 8
    csum_bytes = csum.to_bytes(math.ceil(LEN2 * 4 / 8), "big")
    return msg + base_w(csum_bytes, LEN2)


def digits_for_message(message):
    # the per chain step counts B[i] used by sign
    return msg_digits(H_msg(message))


def hash_counts(message):
    # count the sha256 calls each op does on this message
    # F is one sha per chain step, H_msg is one extra sha for sign and verify
    # keygen also derives LEN private atoms with one sha each
    b = digits_for_message(message)
    sign_f = sum(b)
    verify_f = sum((W - 1) - d for d in b)
    keygen_f = LEN * (W - 1)
    return {
        "keygen": {"f_calls": keygen_f, "seed_derive_sha": LEN,
                   "total_sha256": keygen_f + LEN},
        "sign": {"f_calls": sign_f, "h_msg_sha": 1,
                 "total_sha256": sign_f + 1},
        "verify": {"f_calls": verify_f, "h_msg_sha": 1,
                   "total_sha256": verify_f + 1},
    }


# ---------- timing ----------

def time_us(fn, iters):
    # run fn iters times and return microseconds per call
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


# ---------- json output ----------

def write_json(name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", path)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name + ".json")) as f:
        return json.load(f)
