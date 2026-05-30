import os
import numpy as np
import hashlib, math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#WOTS parameters
N    = 32
W    = 16
LEN1 = math.ceil(8 * N / math.log2(W))
LEN2 = math.floor(math.log2(LEN1 * (W - 1)) / math.log2(W)) + 1
LEN  = LEN1 + LEN2

# import CUDA kernel
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLAINTEXT_DIR = os.path.join(_HERE, "plaintext")
with open(os.path.join(_HERE, "wots_kernel.cu"), "r") as f:
    _CUDA_SRC = f.read()
_mod = SourceModule(_CUDA_SRC, no_extern_c=True, options=['-allow-unsupported-compiler'])
_chain = _mod.get_function("wots_chain")

#host
def pack(elems):
    return np.frombuffer(b"".join(elems), dtype=">u4").astype(np.uint32)

def unpack(arr, count):
    b = arr.astype(">u4").tobytes()
    return [b[i*32:(i+1)*32] for i in range(count)]

def run_chain(start_words, steps, block=256):
    Nc = len(steps)
    out = np.empty_like(start_words)
    s_gpu  = cuda.mem_alloc(start_words.nbytes); cuda.memcpy_htod(s_gpu, start_words)
    st_gpu = cuda.mem_alloc(steps.nbytes);       cuda.memcpy_htod(st_gpu, steps)
    o_gpu  = cuda.mem_alloc(out.nbytes)
    grid   = (Nc + block - 1) // block
    _chain(s_gpu, st_gpu, o_gpu, np.int32(Nc), block=(block, 1, 1), grid=(grid, 1))
    cuda.memcpy_dtoh(out, o_gpu)
    for m in (s_gpu, st_gpu, o_gpu): m.free()
    return out

def base_w(x: bytes, out_len: int):
    digits = []
    for byte in x:
        digits.append(byte >> 4)
        digits.append(byte & 0x0f)
    return digits[:out_len]

def msg_digits(digest32: bytes):
    msg = base_w(digest32, LEN1)
    csum = sum((W - 1) - d for d in msg)
    csum <<= (8 - ((LEN2 * 4) % 8)) % 8
    csum_bytes = csum.to_bytes(math.ceil(LEN2 * 4 / 8), "big")
    return msg + base_w(csum_bytes, LEN2)

def H_msg(message: bytes) -> bytes:
    return hashlib.sha256(b'\x02' * N + message).digest()

def hash_file(file_name: str) -> bytes:
    with open(os.path.join(_PLAINTEXT_DIR, file_name), "rb") as f:
        return H_msg(f.read())

def prg(seed: bytes, i: int):
    return hashlib.sha256(seed + i.to_bytes(4, "big")).digest()

# WOTS KeyGen
def keygen(seed: bytes):
    sk = [prg(seed, i) for i in range(LEN)]
    steps = np.full(LEN, W - 1, dtype=np.int32)
    pk = unpack(run_chain(pack(sk), steps), LEN)
    return sk, pk
#WOTS Sign
def sign(sk, file_name: str):
    d = hash_file(file_name)
    steps = np.array(msg_digits(d), dtype=np.int32)
    return unpack(run_chain(pack(sk), steps), LEN)
#WOTS Verify
def verify(pk, file_name: str, sig):
    d = hash_file(file_name)
    md = msg_digits(d)
    steps = np.array([(W - 1) - m for m in md], dtype=np.int32)
    return unpack(run_chain(pack(sig), steps), LEN) == pk

