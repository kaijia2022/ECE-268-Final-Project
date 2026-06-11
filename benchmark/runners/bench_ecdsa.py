# ECDSA reference for the size and speed comparison
# ECDSA is not implemented by us. We use P-256 because our plain WOTS with a
# deterministic message hash is bounded by SHA-256 collision resistance which
# is ~128 bit classical, the same level as P-256. So this is an equal level
# comparison. Sizes are derived from the curve. Speed is timed for context.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ITERS = int(os.environ.get("ECDSA_ITERS", "5000"))

# name is the output json file, curve is the cryptography curve object
CURVES = [
    ("ecdsa", "secp256r1", ec.SECP256R1()),
]


def bench_curve(name, label, curve):
    msg = c.bench_message()

    priv = ec.generate_private_key(curve)
    pub = priv.public_key()
    sig = priv.sign(msg, ec.ECDSA(hashes.SHA256()))

    # sizes come from the field size so nothing is hard coded
    field_bytes = (curve.key_size + 7) // 8
    pub_compressed = len(pub.public_bytes(Encoding.X962, PublicFormat.CompressedPoint))

    keygen_us = c.time_us(lambda: ec.generate_private_key(curve), ITERS)
    sign_us = c.time_us(lambda: priv.sign(msg, ec.ECDSA(hashes.SHA256())), ITERS)
    verify_us = c.time_us(
        lambda: pub.verify(sig, msg, ec.ECDSA(hashes.SHA256())), ITERS)

    data = {
        "scheme": "ecdsa",
        "impl": "cryptography-" + label,
        "curve": label,
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
    for name, label, curve in CURVES:
        bench_curve(name, label, curve)


if __name__ == "__main__":
    main()
