import secrets
import hashlib
from pathlib import Path

KEY_SIZE = 32
NUM_KEY_PAIR = 256
NUM_KEY = NUM_KEY_PAIR*2
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB
BASE_DIR = Path(__file__).resolve().parent
PLAINTEXT_DIR = BASE_DIR / "plaintext"


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
        AssertionError: If `file_name` is not a string or the path is not a file.
    '''
    assert isinstance(file_name, str), "file_name should be string"

    file_path = (PLAINTEXT_DIR / file_name).resolve()
    plaintext_dir = PLAINTEXT_DIR.resolve()
    assert plaintext_dir in file_path.parents, "file_name should stay inside plaintext folder"
    assert file_path.is_file(), f"file does not exist: {file_path}"
    return file_path


def hash_file(file_name):
    '''
    Compute SHA-256 over a plaintext file.

    The file is read in chunks, so this function can be used as a CPU baseline
    for large-message tests without loading the full file into memory.

    Args:
        file_name: Name of the file under the local plaintext folder.

    Returns:
        The 32-byte SHA-256 digest of the file contents.
    '''
    h = hashlib.sha256()
    file_path = get_file_path(file_name)

    with open(file_path, "rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            h.update(chunk)
    return h.digest()


def digest_to_bits(digest):
    '''
    Convert a SHA-256 digest into 256 bits in MSB-first order.

    Args:
        digest: A 32-byte SHA-256 digest.

    Returns:
        A list of 256 integers, where each entry is either 0 or 1.

    Raises:
        AssertionError: If `digest` is not exactly 32 bytes.
    '''
    assert isinstance(digest, bytes), "digest should be bytes"
    assert len(digest) == KEY_SIZE, "digest should be 32 bytes"

    return [(byte >> (7 - j)) & 1 for byte in digest for j in range(8)]


def check_key(key, key_name):
    '''
    Check the shape and byte length of a Lamport public or private key.

    Args:
        key: A nested list of shape [256][2].
        key_name: Name used in error messages, such as "private_key".

    Raises:
        AssertionError: If the key does not have the expected Lamport shape
            or if any atom is not a 32-byte bytes object.
    '''
    assert isinstance(key, list), f"{key_name} is not of type list"
    assert len(key) == NUM_KEY_PAIR, f"{key_name} has incorrect length"

    for i in range(NUM_KEY_PAIR):
        assert isinstance(key[i], list), f"{key_name}[{i}] is not of type list"
        assert len(key[i]) == 2, f"{key_name}[{i}] has incorrect length"

        for j in range(2):
            assert isinstance(key[i][j], bytes), f"{key_name}[{i}][{j}] should be bytes"
            assert len(key[i][j]) == KEY_SIZE, f"{key_name}[{i}][{j}] should be 32 bytes"


def check_signature(signature):
    '''
    Check the shape and byte length of a Lamport signature.

    Args:
        signature: A list of 256 revealed private-key atoms.

    Raises:
        AssertionError: If the signature is not a list of 256 32-byte atoms.
    '''
    assert isinstance(signature, list), "signature is not of type list"
    assert len(signature) == NUM_KEY_PAIR, "signature has incorrect length"

    for i in range(NUM_KEY_PAIR):
        assert isinstance(signature[i], bytes), f"signature[{i}] should be bytes"
        assert len(signature[i]) == KEY_SIZE, f"signature[{i}] should be 32 bytes"


def key_gen(seed = None):
    '''
    Generate a Lamport one-time signature key pair.

    Produces 256 pairs of (private, public) atoms. Each private atom is
    a 32-byte value; each public atom is its SHA-256 hash.

    When `seed` is provided, all randomness is derived deterministically
    via SHA-256, producing the same key pair on every call with the same seed.
    This is useful for testing and for CPU/GPU baseline comparisons. When
    `seed` is None, randomness comes from `secrets.token_bytes`.

    Args:
        seed: Optional bytes used to deterministically derive the key pair.
            Must be of type `bytes` if provided.

    Returns:
        A tuple `(public_key, private_key)` where each is a nested list of
        shape [256][2], and each leaf is a 32-byte `bytes` object.
    '''

    if seed is None :
        nums = secrets.token_bytes(KEY_SIZE*NUM_KEY)

    else:
        assert isinstance(seed, bytes), "seed must be bytes"
        nums = b"".join(
            hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            for i in range(NUM_KEY)
        )

    private_key = [[nums[(i*2 + j)*KEY_SIZE : (i*2 + j)*KEY_SIZE + KEY_SIZE]
        for j in range(2)]
       for i in range(NUM_KEY_PAIR)]
    public_key = [[hashlib.sha256(private_key[i][j]).digest() for j in range(2)]
              for i in range(NUM_KEY_PAIR)]

    check_key(private_key, "private_key")
    check_key(public_key, "public_key")
    return (public_key,private_key)


def sign(file_name,private_key,consume_private_key = False):
    '''
    Produce a Lamport one-time signature over the contents of a file.

    Hashes the file `./plaintext/<file_name>` with SHA-256, then for each of
    the 256 bits of the digest reveals one atom of the corresponding private
    key pair. Bit 0 reveals `private_key[i][0]`, and bit 1 reveals
    `private_key[i][1]`.

    Args:
        file_name: Name of the file to sign, relative to this module's
            `plaintext` folder. Must be of type `str`.
        private_key: A nested list of shape [256][2] where each leaf is a
            32-byte `bytes` object, as returned by `key_gen`.
        consume_private_key: If True, clear `private_key` after signing to
            reduce the chance of accidentally reusing a Lamport private key.

    Returns:
        A list of 256 `bytes` objects, each 32 bytes long, forming the
        Lamport signature.
    '''
    check_key(private_key, "private_key")

    bits = digest_to_bits(hash_file(file_name))
    signature = [private_key[i][bits[i]] for i in range(NUM_KEY_PAIR)]

    check_signature(signature)
    if consume_private_key:
        private_key.clear()
    return signature


def verify(file_name,signature,public_key):
    '''
    Verify a Lamport one-time signature over the contents of a file.

    Hashes `./plaintext/<file_name>` with SHA-256, extracts 256 bits in
    MSB-first order, and checks that each revealed signature atom hashes to
    the selected public-key atom.

    Args:
        file_name: Name of the file to verify, relative to this module's
            `plaintext` folder. Must be of type `str`.
        signature: List of 256 `bytes` objects, each 32 bytes long.
        public_key: A nested list of shape [256][2] where each leaf is a
            32-byte `bytes` object, as returned by `key_gen`.

    Returns:
        True if the signature is valid for the file under the given public
        key, False otherwise.
    '''
    check_signature(signature)
    check_key(public_key, "public_key")

    bits = digest_to_bits(hash_file(file_name))
    for i in range(NUM_KEY_PAIR):
        if hashlib.sha256(signature[i]).digest() != public_key[i][bits[i]]:
            return False
    return True


def test_roundtrip(file_name = "short.txt"):
    '''
    Test that a valid signature verifies and a modified signature fails.
    '''
    pk, sk = key_gen(b"test_seed_v1_padding_padding")
    sig = sign(file_name, sk)
    assert verify(file_name, sig, pk) == True

    sig[0] = b"\x00" * KEY_SIZE
    assert verify(file_name, sig, pk) == False
    print("roundtrip test passed")


def test_key_gen():
    '''
    Test that seeded key generation is deterministic.
    '''
    # print("random generation consistency check...")
    pk1, sk1 = key_gen(b"test_seed")
    pk2, sk2 = key_gen(b"test_seed")
    assert pk1 == pk2, "Public key mismatch: Same seed should produce same result"
    assert sk1 == sk2, "Private key mismatch: Same seed should produce same result"
    print("random generation consistency check passed")


def test_checks():
    '''
    Test a few input checks for malformed keys and signatures.
    '''
    pk, sk = key_gen(b"test_seed_checks")
    sig = sign("short.txt", sk)

    bad_pk = [row[:] for row in pk]
    bad_pk[0][0] = b"short"
    try:
        verify("short.txt", sig, bad_pk)
        assert False, "bad public key should fail"
    except AssertionError:
        pass

    bad_sig = sig[:]
    bad_sig[0] = b"short"
    try:
        verify("short.txt", bad_sig, pk)
        assert False, "bad signature should fail"
    except AssertionError:
        pass

    pk, sk = key_gen(b"test_seed_consume")
    sig = sign("short.txt", sk, consume_private_key = True)
    assert len(sk) == 0, "private key should be cleared after consume_private_key=True"
    assert verify("short.txt", sig, pk) == True

    print("input check test passed")


if __name__ == "__main__":
    test_key_gen()
    test_roundtrip()
    test_checks()
    print("all test passed")
