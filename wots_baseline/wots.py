import hashlib
import math
import secrets
from pathlib import Path

KEY_SIZE = 32 # n = 32
BASE_DIR = Path(__file__).resolve().parent
PLAINTEXT_DIR = BASE_DIR / "plaintext"

# param calculation
# len_1 = ceil(8n / lg(w)) and len_2 = floor(lg(len_1 *(w - 1)) / lg(w)) + 1
# n = 32
w = 16
len_1 = math.ceil(8*KEY_SIZE / math.log2(w))
len_2 = math.floor(math.log2(len_1 *(w - 1)) / math.log2(w)) + 1
NUM_KEY = len_1 + len_2


def check_bytes(value, value_name):
    '''
    Check that a value is a 32-byte bytes object.

    Args:
        value: Object to check.
        value_name: Name used in error messages.

    Raises:
        AssertionError: If `value` is not bytes or is not 32 bytes long.
    '''
    assert isinstance(value, bytes), f"{value_name} must be bytes"
    assert len(value) == KEY_SIZE, f"{value_name} must be 32 bytes, got {len(value)} bytes"


def check_key(key, key_name):
    '''
    Check the shape and byte length of a WOTS key or signature.

    A key is a flat list of `NUM_KEY = len_1 + len_2` atoms. For w = 16
    and n = 32, this is 67 atoms, and each atom is 32 bytes.

    Args:
        key: List of WOTS atoms.
        key_name: Name used in error messages, such as "sk" or "signature".

    Raises:
        AssertionError: If the key does not have the expected shape.
    '''
    assert isinstance(key, list), f"{key_name} is not of type list"
    assert len(key) == NUM_KEY, f"{key_name} has incorrect length"

    for i in range(NUM_KEY):
        check_bytes(key[i], f"{key_name}[{i}]")


def F(M):
    '''
    WOTS chaining hash F.

    Computes SHA2-256(toByte(0, 32) || M), where the 32-byte prefix provides
    domain separation from H_msg. F takes a single 32-byte input and is
    iterated to build each chain.

    Args:
        M: 32-byte message to compress.

    Returns:
        The 32-byte hash output.
    '''
    return hashlib.sha256((0).to_bytes(KEY_SIZE, "big") + M).digest()


def H_msg(M):
    '''
    WOTS message hash H_msg.

    Computes SHA2-256(toByte(2, 32) || M), where the 32-byte prefix provides
    domain separation from F. The message hash takes the message alone.

    Args:
        M: Arbitrary-length message bytes.

    Returns:
        The 32-byte hash output.
    '''
    return hashlib.sha256((2).to_bytes(KEY_SIZE, "big") + M).digest()


def get_file_path(file_name):
    '''
    Convert a plaintext file name into the actual file path used by this module.

    Args:
        file_name: Name of the file under the local plaintext folder.
            Must be of type `str`.

    Returns:
        A `Path` object pointing to `./plaintext/<file_name>`, relative to
        this source file instead of the current terminal working directory.

    Raises:
        AssertionError: If `file_name` is not a string, tries to leave the
            plaintext folder, or does not point to a file.
    '''
    assert isinstance(file_name, str), "file_name should be string"

    file_path = (PLAINTEXT_DIR / file_name).resolve()
    plaintext_dir = PLAINTEXT_DIR.resolve()
    assert plaintext_dir in file_path.parents, "file_name should stay inside plaintext folder"
    assert file_path.is_file(), f"file does not exist: {file_path}"
    return file_path


def hash_file(file_name):
    '''
    Compute the WOTS message hash H_msg over a plaintext file.

    The whole file is read into memory and hashed in one shot, matching
    how the CUDA version feeds the message to the GPU kernel.

    Args:
        file_name: Name of the file under the local plaintext folder.

    Returns:
        The 32-byte H_msg digest of the file contents.
    '''
    file_path = get_file_path(file_name)
    with open(file_path, "rb") as file:
        return H_msg(file.read())


def chain(X, i, s):
    '''
    Apply `s` WOTS chaining steps starting at chain index `i`.

    Each step is an iteration of F: temp = F(temp). The chain depends only on
    its input node.

    Args:
        X: Current chain node, 32 bytes.
        i: Starting hash index in the chain.
        s: Number of steps to apply.

    Returns:
        The 32-byte chain output after `s` steps.

    Raises:
        AssertionError: If inputs have invalid type, size, or chain range.
    '''
    check_bytes(X, "X")
    assert isinstance(i, int), "i should be int"
    assert isinstance(s, int), "s should be int"
    assert i >= 0, "i should be non-negative"
    assert s >= 0, "s should be non-negative"
    assert i + s <= w - 1, "chain step exceeds maximum chain length"

    temp = X
    for _ in range(i, i + s):
        temp = F(temp)
    return temp


def key_gen(seed = None):
    '''
    Generate a WOTS public/private key pair.

    The private key contains `NUM_KEY` random 32-byte atoms. The public key is
    produced by taking every private-key atom to the end of its WOTS chain.

    Args:
        seed: Optional bytes used to deterministically derive the private key.
            If None, randomness comes from `secrets.token_bytes`.

    Returns:
        A tuple `(pk, sk)`, where both are lists of `NUM_KEY` 32-byte atoms.
    '''
    if seed is None:
        nums = secrets.token_bytes(KEY_SIZE*NUM_KEY)
    else :
        assert isinstance(seed, bytes), "seed must be bytes"
        nums = b"".join(
            hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            for i in range(NUM_KEY)
        )

    private_key = [
    nums[i * KEY_SIZE : (i + 1) * KEY_SIZE]
    for i in range(NUM_KEY)]

    check_key(private_key, "private_key")

    public_key = []
    for i in range(len(private_key)):
        public_key.append(chain(private_key[i], 0, w - 1))

    check_key(public_key, "public_key")
    return (public_key,private_key)


def base_w(data, out_len, w):
    '''
    Convert bytes into base-w digits.

    This implementation is intended for w = 16, where each byte cleanly splits
    into two base-16 digits. It is used for both the message digest and the
    WOTS checksum.

    Args:
        data: Input bytes to convert.
        out_len: Number of base-w digits to output.
        w: Winternitz parameter. Must be a power of two.

    Returns:
        A list of `out_len` integers in the range [0, w - 1].
    '''
    assert isinstance(data, bytes), "data should be bytes"
    assert isinstance(out_len, int), "out_len should be int"
    assert isinstance(w, int), "w should be int"
    assert out_len >= 0, "out_len should be non-negative"
    assert w > 1 and (w & (w - 1)) == 0, "w should be a power of two"

    log_w = int(math.log2(w))
    assert 8 % log_w == 0, "this base_w implementation needs log2(w) to divide 8"
    assert len(data) * 8 >= out_len * log_w, "not enough data for requested out_len"

    out = []
    bits = 0
    total = 0
    in_idx = 0
    for _ in range(out_len):
        if bits == 0:
            total = data[in_idx]
            in_idx += 1
            bits = 8
        bits -= log_w
        out.append((total >> bits) & (w - 1))
    return out


def get_checksum(M):
    '''
    Compute the WOTS checksum digits for a base-w message.

    Args:
        M: List of `len_1` base-w digits from the message digest.

    Returns:
        A list of `len_2` checksum digits in base w.
    '''
    assert isinstance(M, list), "M should be list"
    assert len(M) == len_1, "M has incorrect length"
    assert all(isinstance(m_i, int) for m_i in M), "M should contain integers"
    assert all(0 <= m_i < w for m_i in M), "M entries should be base-w digits"

    csum = sum(w - 1 - m_i for m_i in M)
    csum_bits = len_2 * int(math.log2(w))
    shift = (8 - (csum_bits % 8)) % 8
    csum <<= shift
    csum_bytes = csum.to_bytes(math.ceil(csum_bits / 8), "big")
    return base_w(csum_bytes, len_2, w)


def get_msg_digits(file_name):
    '''
    Convert a message file into the full WOTS base-w digit vector.

    Args:
        file_name: Name of the file under the local plaintext folder.

    Returns:
        A list `B = M + checksum`, with length `NUM_KEY`.
    '''
    file_hash = hash_file(file_name)
    M = base_w(file_hash, len_1, w)
    M_csum = get_checksum(M)
    B = M + M_csum

    assert len(B) == NUM_KEY, "message digit vector has incorrect length"
    return B


def sign(file_name, sk):
    '''
    Produce a WOTS signature over the contents of a file.

    The message is hashed with SHA-256, converted to base-w digits, extended
    with the WOTS checksum, and each private-key atom is chained forward by
    the corresponding digit.

    Args:
        file_name: Name of the file to sign, relative to this module's
            `plaintext` folder.
        sk: WOTS private key returned by `key_gen`.

    Returns:
        A list of `NUM_KEY` 32-byte atoms forming the WOTS signature.
    '''
    check_key(sk, "sk")

    B = get_msg_digits(file_name)

    signature = []
    for i in range(len(B)):
        signature.append(chain(sk[i], 0, B[i]))

    check_key(signature, "signature")
    return signature


def verify(file_name, pk, signature):
    '''
    Verify a WOTS signature over the contents of a file.

    Rebuilds the public key candidate by chaining each signature atom from its
    message digit to the end of the chain, then compares it to `pk`.

    Args:
        file_name: Name of the file to verify, relative to this module's
            `plaintext` folder.
        pk: WOTS public key returned by `key_gen`.
        signature: WOTS signature returned by `sign`.

    Returns:
        True if the signature is valid for the file under the given public key,
        False otherwise.
    '''
    check_key(pk, "pk")
    check_key(signature, "signature")

    B = get_msg_digits(file_name)

    sig=[]
    for i in range(len(B)):
        sig.append(chain(signature[i], B[i], w - 1 - B[i]))

    check_key(sig, "recomputed_pk")
    return sig == pk


def test_roundtrip():
    '''
    Test that a valid signature verifies and a modified signature fails.
    '''
    pk, sk = key_gen()
    sig = sign("short.txt", sk)
    assert verify("short.txt", pk, sig) is True

    sig[0] = bytes(32)
    assert verify("short.txt", pk, sig) is False
    print("roundtrip test passed")


def test_key_gen():
    '''
    Test that seeded key generation is deterministic.
    '''
    pk1, sk1 = key_gen(b"test_seed")
    pk2, sk2 = key_gen(b"test_seed")
    assert pk1 == pk2, "Public key mismatch: Same seed should produce same result"
    assert sk1 == sk2, "Private key mismatch: Same seed should produce same result"
    print("key generation test passed")


def test_checks():
    '''
    Test a few input checks for malformed keys, signatures, and paths.
    '''
    pk, sk = key_gen(b"test_seed_checks")
    sig = sign("short.txt", sk)

    bad_pk = pk[:]
    bad_pk[0] = b"short"
    try:
        verify("short.txt", bad_pk, sig)
        assert False, "bad public key should fail"
    except AssertionError:
        pass

    bad_sig = sig[:]
    bad_sig[0] = b"short"
    try:
        verify("short.txt", pk, bad_sig)
        assert False, "bad signature should fail"
    except AssertionError:
        pass

    try:
        sign("../wots_plain.py", sk)
        assert False, "path outside plaintext should fail"
    except AssertionError:
        pass

    print("input check test passed")


if __name__ == "__main__":
    test_key_gen()
    test_roundtrip()
    test_checks()
    print("all test passed")
