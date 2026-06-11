# ECDSA reference for the size and speed comparison
# ECDSA is not implemented by us. We compare each of our schemes against the
# ECDSA curve at the same classical security level. Sizes come from the curve.
# Speed is timed for context.
#
# P-256 ~128 bit classical pairs with our plain WOTS. That WOTS has no chain
# bitmask and a deterministic message hash so a forger only needs a SHA-256
# collision which is 2^128, the same level as P-256.
#
# P-521 ~256 bit classical pairs with our XMSS. XMSS uses W-OTS+ with bitmasks
# and a secret randomized message hash that binds the root and the leaf index.
# That kills the offline collision shortcut so security rests on second preimage
# which is ~2^256 minus a tiny multi target loss, the 256 bit tier, like P-521.
# A 256 bit curve is paired with SHA-512 so the hash matches the curve strength.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ITERS = int(os.environ.get("ECDSA_ITERS", "5000"))

# name is the output json file, curve is the cryptography curve object,
# hash is the message hash paired with the curve at its security level
CURVES = [
    ("ecdsa", "secp256r1", ec.SECP256R1(), hashes.SHA256()),
    ("ecdsa_p521", "secp521r1", ec.SECP521R1(), hashes.SHA512()),
]


def bench_curve(name, label, curve, hash_alg):
    msg = c.bench_message()

    priv = ec.generate_private_key(curve)
    pub = priv.public_key()
    sig = priv.sign(msg, ec.ECDSA(hash_alg))

    # sizes come from the field size so nothing is hard coded
    field_bytes = (curve.key_size + 7) // 8
    pub_compressed = len(pub.public_bytes(Encoding.X962, PublicFormat.CompressedPoint))

    keygen_us = c.time_us(lambda: ec.generate_private_key(curve), ITERS)
    sign_us = c.time_us(lambda: priv.sign(msg, ec.ECDSA(hash_alg)), ITERS)
    verify_us = c.time_us(
        lambda: pub.verify(sig, msg, ec.ECDSA(hash_alg)), ITERS)

    data = {
        "scheme": "ecdsa",
        "impl": "cryptography-" + label,
        "curve": label,
        "message_hash": hash_alg.name,
        "iters": ITERS,
        # raw r and s is 2 field elements, the der encoding is a bit larger
        "sizes_bytes": {
            "private_key": field_bytes,
            "public_key": pub_compressed,
            "signature": 2 * field_bytes,
            "signature_der": len(sig),
        },
        # ecdsa hashes the message once, the rest is curve math not hashing
        "hash_counts": {
            "keygen": {"total_sha256": 0},
            "sign": {"total_sha256": 1},
            "verify": {"total_sha256": 1},
        },
        "timing_us": {
            "keygen": keygen_us,
            "sign": sign_us,
            "verify": verify_us,
        },
        "source": "cryptography library openssl backend on the cpu",
    }
    c.write_json(name, data)


def main():
    for name, label, curve, hash_alg in CURVES:
        bench_curve(name, label, curve, hash_alg)


if __name__ == "__main__":
    main()
