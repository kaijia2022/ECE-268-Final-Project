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
    Check the shape and byte length of a WOTS+ key or signature.

    WOTS+ uses a flat list of `NUM_KEY = len_1 + len_2` atoms. For w = 16
    and n = 32, this is 67 atoms, and each atom is 32 bytes.

    Args:
        key: List of WOTS+ atoms.
        key_name: Name used in error messages, such as "sk" or "signature".

    Raises:
        AssertionError: If the key does not have the expected shape.
    '''
    assert isinstance(key, list), f"{key_name} is not of type list"
    assert len(key) == NUM_KEY, f"{key_name} has incorrect length"

    for i in range(NUM_KEY):
        check_bytes(key[i], f"{key_name}[{i}]")


def xor_bytes(a, b):
    '''
    XOR two 32-byte strings.

    Args:
        a: First 32-byte input.
        b: Second 32-byte input.

    Returns:
        The byte-wise XOR of `a` and `b`.
    '''
    check_bytes(a, "a")
    check_bytes(b, "b")

    return bytes(x ^ y for x, y in zip(a, b))


def F(KEY, M):
    '''
    WOTS+ keyed hash F as defined in RFC 8391 Section 5.1.

    Computes SHA2-256(toByte(0, 32) || KEY || M), where the 32-byte
    prefix provides domain separation from PRF and H_msg.

    Args:
        KEY: 32-byte key derived from PRF.
        M: 32-byte message to compress.

    Returns:
        The 32-byte hash output.
    '''
    return hashlib.sha256((0).to_bytes(KEY_SIZE, "big") + KEY + M).digest()


def PRF(KEY, M):
    '''
    WOTS+ pseudorandom function PRF as defined in RFC 8391 Section 5.1.

    Computes SHA2-256(toByte(3, 32) || KEY || M), where the 32-byte
    prefix provides domain separation from F and H_msg.

    Args:
        KEY: 32-byte key, typically the public SEED.
        M: 32-byte input, typically a serialized ADRS.

    Returns:
        The 32-byte pseudorandom output.
    '''
    return hashlib.sha256((3).to_bytes(KEY_SIZE, "big") + KEY + M).digest()


def H_msg(KEY, M):
    '''
    WOTS+ keyed message hash H_msg as defined in RFC 8391 Section 5.1.

    Computes SHA2-256(toByte(2, 32) || KEY || M), where the 32-byte
    prefix provides domain separation from F and PRF.

    Args:
        KEY: 32-byte key, typically the public SEED.
        M: Arbitrary-length message bytes.

    Returns:
        The 32-byte hash output.
    '''
    return hashlib.sha256((2).to_bytes(KEY_SIZE, "big") + KEY + M).digest()


class ADRS:
    """
    WOTS+ address.

    32 bytes = 8 words = 8 * 4 bytes.

    For OTS hash address:
        word[0] = layer address
        word[1] = tree address high 32 bits
        word[2] = tree address low 32 bits
        word[3] = type, 0 for OTS hash address
        word[4] = OTS address
        word[5] = chain address
        word[6] = hash address
        word[7] = keyAndMask
    """

    def __init__(self):
        self.words = [0] * 8

    def copy(self):
        '''
        Return a copy of the current address.
        '''
        other = ADRS()
        other.words = self.words[:]
        return other

    def to_bytes(self):
        '''
        Convert the address to 32 bytes in big-endian word order.
        '''
        return b"".join(word.to_bytes(4, "big") for word in self.words)

    def set_layer_address(self, layer):
        '''
        Set the XMSS layer address word.
        '''
        assert isinstance(layer, int), "layer should be int"
        assert layer >= 0, "layer should be non-negative"
        self.words[0] = layer & 0xFFFFFFFF

    def set_tree_address(self, tree):
        '''
        Set the 64-bit XMSS tree address.
        '''
        assert isinstance(tree, int), "tree address should be int"
        assert tree >= 0 and tree < (1 << 64), "tree address must fit in 64 bits"

        self.words[1] = (tree >> 32) & 0xFFFFFFFF
        self.words[2] = tree & 0xFFFFFFFF

    def set_type(self, type_value):
        '''
        Set the address type and clear type-specific trailing words.

        0 = OTS hash address, 1 = L-tree address, 2 = hash tree address.
        '''
        assert isinstance(type_value, int), "type_value should be int"
        assert type_value >= 0, "type_value should be non-negative"
        self.words[3] = type_value & 0xFFFFFFFF

        self.words[4] = 0
        self.words[5] = 0
        self.words[6] = 0
        self.words[7] = 0

    def set_ots_address(self, ots):
        '''
        Set the one-time-signature address word.
        '''
        assert isinstance(ots, int), "ots should be int"
        assert ots >= 0, "ots should be non-negative"
        self.words[4] = ots & 0xFFFFFFFF

    def set_chain_address(self, chain):
        '''
        Set the chain address word.
        '''
        assert isinstance(chain, int), "chain should be int"
        assert chain >= 0, "chain should be non-negative"
        self.words[5] = chain & 0xFFFFFFFF

    def set_hash_address(self, hash_addr):
        '''
        Set the hash address word inside one chain.
        '''
        assert isinstance(hash_addr, int), "hash_addr should be int"
        assert hash_addr >= 0, "hash_addr should be non-negative"
        self.words[6] = hash_addr & 0xFFFFFFFF

    def set_key_and_mask(self, key_and_mask):
        '''
        Set the keyAndMask word used to derive KEY and BM.
        '''
        assert isinstance(key_and_mask, int), "key_and_mask should be int"
        assert key_and_mask >= 0, "key_and_mask should be non-negative"
        self.words[7] = key_and_mask & 0xFFFFFFFF


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


def hash_file(file_name, SEED):
    '''
    Compute the WOTS+ message hash H_msg over a plaintext file.

    The whole file is read into memory and hashed in one shot, matching
    how the CUDA version feeds the message to the GPU kernel.

    Args:
        file_name: Name of the file under the local plaintext folder.
        SEED: 32-byte public seed used as the H_msg key.

    Returns:
        The 32-byte H_msg digest of the file contents.
    '''
    check_bytes(SEED, "SEED")
    file_path = get_file_path(file_name)
    with open(file_path, "rb") as file:
        return H_msg(SEED, file.read())


def chain(X, i, s, SEED, adrs_obj):
    '''
    Apply `s` WOTS+ chaining steps starting at chain index `i`.

    Args:
        X: Current chain node, 32 bytes.
        i: Starting hash index in the chain.
        s: Number of steps to apply.
        SEED: Public seed used to derive KEY and BM.
        adrs_obj: ADRS object containing the current chain address.

    Returns:
        The 32-byte chain output after `s` steps.

    Raises:
        AssertionError: If inputs have invalid type, size, or chain range.
    '''
    check_bytes(X, "X")
    check_bytes(SEED, "SEED")
    assert isinstance(i, int), "i should be int"
    assert isinstance(s, int), "s should be int"
    assert i >= 0, "i should be non-negative"
    assert s >= 0, "s should be non-negative"
    assert i + s <= w - 1, "chain step exceeds maximum chain length"
    assert isinstance(adrs_obj, ADRS), "adrs_obj should be ADRS"

    temp = X
    adrs = adrs_obj.copy()
    for j in range(i,i+s):
        adrs.set_hash_address(j)

        adrs.set_key_and_mask(0)
        KEY = PRF(SEED, adrs.to_bytes())
        adrs.set_key_and_mask(1)
        BM = PRF(SEED, adrs.to_bytes())
        temp = F(KEY, xor_bytes(temp, BM))
    return temp


def key_gen(public_seed,seed = None):
    '''
    Generate a WOTS+ public/private key pair.

    The private key contains `NUM_KEY` random 32-byte atoms. The public key is
    produced by taking every private-key atom to the end of its WOTS+ chain.

    Args:
        public_seed: 32-byte public seed used by the WOTS+ chain function.
        seed: Optional bytes used to deterministically derive the private key.
            If None, randomness comes from `secrets.token_bytes`.

    Returns:
        A tuple `(pk, sk)`, where both are lists of `NUM_KEY` 32-byte atoms.
    '''
    check_bytes(public_seed, "public_seed")

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
    adrs = ADRS()
    adrs.set_type(0)
    adrs.set_ots_address(0)
    for i in range(len(private_key)):
        adrs.set_chain_address(i)
        public_key.append(chain(private_key[i],0,w-1,public_seed,adrs))

    check_key(public_key, "public_key")
    return (public_key,private_key)


def base_w(data, out_len, w):
    '''
    Convert bytes into base-w digits.

    This implementation is intended for w = 16, where each byte cleanly splits
    into two base-16 digits. It is used for both the message digest and the
    WOTS+ checksum.

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
    Compute the WOTS+ checksum digits for a base-w message.

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


def get_msg_digits(file_name, SEED):
    '''
    Convert a message file into the full WOTS+ base-w digit vector.

    Args:
        file_name: Name of the file under the local plaintext folder.
        SEED: 32-byte public seed used as the H_msg key.

    Returns:
        A list `B = M + checksum`, with length `NUM_KEY`.
    '''
    file_hash = hash_file(file_name, SEED)
    M = base_w(file_hash, len_1, w)
    M_csum = get_checksum(M)
    B = M + M_csum

    assert len(B) == NUM_KEY, "message digit vector has incorrect length"
    return B


def sign(file_name,public_seed, sk):
    '''
    Produce a WOTS+ signature over the contents of a file.

    The message is hashed with SHA-256, converted to base-w digits, extended
    with the WOTS+ checksum, and each private-key atom is chained forward by
    the corresponding digit.

    Args:
        file_name: Name of the file to sign, relative to this module's
            `plaintext` folder.
        public_seed: 32-byte public seed used by the WOTS+ chain function.
        sk: WOTS+ private key returned by `key_gen`.

    Returns:
        A list of `NUM_KEY` 32-byte atoms forming the WOTS+ signature.
    '''
    check_bytes(public_seed, "public_seed")
    check_key(sk, "sk")

    B = get_msg_digits(file_name, public_seed)

    signature = []
    adrs = ADRS()
    adrs.set_type(0)
    adrs.set_ots_address(0)
    for i in range(len(B)):
        adrs.set_chain_address(i)
        signature.append(chain(sk[i], 0, B[i], public_seed, adrs))

    check_key(signature, "signature")
    return signature


def verify(file_name,public_seed,pk,signature):
    '''
    Verify a WOTS+ signature over the contents of a file.

    Rebuilds the public key candidate by chaining each signature atom from its
    message digit to the end of the chain, then compares it to `pk`.

    Args:
        file_name: Name of the file to verify, relative to this module's
            `plaintext` folder.
        public_seed: 32-byte public seed used by the WOTS+ chain function.
        pk: WOTS+ public key returned by `key_gen`.
        signature: WOTS+ signature returned by `sign`.

    Returns:
        True if the signature is valid for the file under the given public key,
        False otherwise.
    '''
    check_bytes(public_seed, "public_seed")
    check_key(pk, "pk")
    check_key(signature, "signature")

    B = get_msg_digits(file_name, public_seed)

    sig=[]
    adrs = ADRS()
    adrs.set_type(0)
    adrs.set_ots_address(0)
    for i in range(len(B)):
        adrs.set_chain_address(i)
        sig.append(chain(signature[i], B[i], w - 1 - B[i], public_seed, adrs))

    check_key(sig, "recomputed_pk")
    return sig == pk


def test_roundtrip():
    '''
    Test that a valid signature verifies and a modified signature fails.
    '''
    public_seed = secrets.token_bytes(KEY_SIZE)
    pk, sk = key_gen(public_seed)
    sig = sign("short.txt", public_seed, sk)
    assert verify("short.txt", public_seed, pk, sig) is True

    sig[0] = bytes(32)
    assert verify("short.txt", public_seed, pk, sig) is False
    print("roundtrip test passed")


def test_key_gen():
    '''
    Test that seeded key generation is deterministic.
    '''
    public_seed = b"public_seed" + b"\x00" * 21
    pk1, sk1 = key_gen(public_seed, b"test_seed")
    pk2, sk2 = key_gen(public_seed, b"test_seed")
    assert pk1 == pk2, "Public key mismatch: Same seed should produce same result"
    assert sk1 == sk2, "Private key mismatch: Same seed should produce same result"
    print("key generation test passed")


def test_checks():
    '''
    Test a few input checks for malformed keys, signatures, and paths.
    '''
    public_seed = b"public_seed" + b"\x00" * 21
    pk, sk = key_gen(public_seed, b"test_seed_checks")
    sig = sign("short.txt", public_seed, sk)

    bad_pk = pk[:]
    bad_pk[0] = b"short"
    try:
        verify("short.txt", public_seed, bad_pk, sig)
        assert False, "bad public key should fail"
    except AssertionError:
        pass

    bad_sig = sig[:]
    bad_sig[0] = b"short"
    try:
        verify("short.txt", public_seed, pk, bad_sig)
        assert False, "bad signature should fail"
    except AssertionError:
        pass

    try:
        sign("../wots.py", public_seed, sk)
        assert False, "path outside plaintext should fail"
    except AssertionError:
        pass

    print("input check test passed")


if __name__ == "__main__":
    test_key_gen()
    test_roundtrip()
    test_checks()
    print("all test passed")
