# Single-tree XMSS built on top of the WOTS+ scheme in wots.py.
import sys
import secrets
import hashlib
from pathlib import Path


# Reuse the WOTS+ primitives from wots.py.
import wots
from wots import (
    KEY_SIZE,
    NUM_KEY,
    ADRS,
    PRF,
    H_msg,
    xor_bytes,
    check_bytes,
    to_digits,
    read_file,
)

ADDR_TYPE_OTS = 0
ADDR_TYPE_LTREE = 1
ADDR_TYPE_HASHTREE = 2

sys.path.insert(0, str(Path(__file__).resolve().parent))


def H(KEY, M):
    '''
    XMSS keyed tree hash H as defined in RFC 8391 Section 5.1.

    Computes SHA2-256(toByte(1, 32) || KEY || M), where the 32-byte prefix
    provides domain separation from F (0), H_msg (2) and PRF (3). `M` is the
    64-byte concatenation of the two masked child nodes of an internal node.

    Args:
        KEY: 32-byte key derived from PRF.
        M: 64-byte message, the two masked children concatenated.

    Returns:
        The 32-byte hash output.

    Raises:
        AssertionError: If `KEY` is not 32 bytes or `M` is not 64 bytes.
    '''
    check_bytes(KEY, "KEY")
    assert isinstance(M, bytes), "M must be bytes"
    assert len(M) == 2 * KEY_SIZE, f"M must be 64 bytes, got {len(M)} bytes"
    return hashlib.sha256((1).to_bytes(KEY_SIZE, "big") + KEY + M).digest()


# wots.py's ADRS only exposes the OTS-hash setters, so set words 4..6
# (L-tree address, tree height, tree index) directly here.
def set_ltree_address(adrs, ltree):
    '''
    Set the L-tree address word (word 4) on an ADRS object.

    Used only for L-tree addresses (type 1), where word 4 selects which
    leaf's WOTS+ public key is being compressed.

    Args:
        adrs: ADRS object to modify in place.
        ltree: Leaf index of the L-tree.

    Raises:
        AssertionError: If `ltree` is not a non-negative int.
    '''
    assert isinstance(ltree, int) and ltree >= 0, "ltree should be non-negative int"
    adrs.words[4] = ltree & 0xFFFFFFFF


def set_tree_height(adrs, height):
    '''
    Set the tree-height word (word 5) on an ADRS object.

    Shared by L-tree and hash-tree addresses; identifies the layer of the
    node currently being hashed.

    Args:
        adrs: ADRS object to modify in place.
        height: Layer index, counting from the leaves up.

    Raises:
        AssertionError: If `height` is not a non-negative int.
    '''
    assert isinstance(height, int) and height >= 0, "height should be non-negative int"
    adrs.words[5] = height & 0xFFFFFFFF


def set_tree_index(adrs, index):
    '''
    Set the tree-index word (word 6) on an ADRS object.

    Shared by L-tree and hash-tree addresses; identifies the position of the
    node within its layer.

    Args:
        adrs: ADRS object to modify in place.
        index: Position of the node within the current layer.

    Raises:
        AssertionError: If `index` is not a non-negative int.
    '''
    assert isinstance(index, int) and index >= 0, "index should be non-negative int"
    adrs.words[6] = index & 0xFFFFFFFF


def rand_hash(left, right, SEED, adrs):
    '''
    Randomized tree hash RAND_HASH as defined in RFC 8391 Section 4.1.5.

    Derives a key and two bitmasks from `SEED` and the current address, XORs
    the children with their masks, and hashes the result with H. The masks
    make every node use an effectively distinct hash function, which is where
    the "eXtended" in XMSS comes from and provides second-preimage resistance.

    Args:
        left: 32-byte left child node.
        right: 32-byte right child node.
        SEED: 32-byte public seed used to derive the key and bitmasks.
        adrs: ADRS object addressing this internal node. Its keyAndMask word
            is overwritten by this function.

    Returns:
        The 32-byte parent node.

    Raises:
        AssertionError: If any of `left`, `right`, or `SEED` is not 32 bytes.
    '''
    check_bytes(left, "left")
    check_bytes(right, "right")
    check_bytes(SEED, "SEED")

    # Reuse wots.PRF to derive the key and the two bitmasks.
    adrs.set_key_and_mask(0)
    KEY = PRF(SEED, adrs.to_bytes())
    adrs.set_key_and_mask(1)
    bm_left = PRF(SEED, adrs.to_bytes())
    adrs.set_key_and_mask(2)
    bm_right = PRF(SEED, adrs.to_bytes())

    # xor_bytes reused from wots.py.
    masked = xor_bytes(left, bm_left) + xor_bytes(right, bm_right)
    return H(KEY, masked)


def ltree(wots_pk, SEED, adrs):
    '''
    Compress a WOTS+ public key into a single L-tree root node.

    The L-tree is an (unbalanced) binary tree over the `NUM_KEY` WOTS+ public
    atoms. Each layer is folded with RAND_HASH; an odd node at any layer is
    lifted unchanged to the next layer.

    Args:
        wots_pk: List of `NUM_KEY` 32-byte WOTS+ public-key atoms.
        SEED: 32-byte public seed.
        adrs: ADRS object with type already set to L-tree and the L-tree
            address set to the leaf index. Modified in place.

    Returns:
        The 32-byte L-tree root, used as one XMSS leaf.

    Raises:
        AssertionError: If `wots_pk` does not have the expected shape.
    '''
    wots.check_key(wots_pk, "wots_pk")

    nodes = list(wots_pk)
    length = len(nodes)
    height = 0
    while length > 1:
        for i in range(length // 2):
            set_tree_height(adrs, height)
            set_tree_index(adrs, i)
            nodes[i] = rand_hash(nodes[2 * i], nodes[2 * i + 1], SEED, adrs)
        if length % 2 == 1:
            nodes[length // 2] = nodes[length - 1]
        length = (length // 2) + (length % 2)
        height += 1
    return nodes[0]


def leaf_secret_seed(secret_seed, idx):
    '''
    Derive the per-leaf WOTS+ secret seed from the master secret seed.

    Uses the canonical RFC 8391 / reference `getSeed` construction: the per-OTS
    seed is `PRF(SK_SEED, ADRS)` where ADRS is an OTS-hash address whose OTS
    address word is the leaf index and whose chain/hash/keyAndMask words are
    zero. This binds the seed to the leaf through the address (rather than an
    ad-hoc `SHA256(seed || idx)`) and gives PRF its domain-separation prefix.

    Args:
        secret_seed: 32-byte master secret seed (SK_SEED).
        idx: Leaf index.

    Returns:
        A 32-byte seed used as the deterministic seed for `wots.key_gen`.

    Raises:
        AssertionError: If `secret_seed` is not 32 bytes or `idx` is negative.
    '''
    check_bytes(secret_seed, "secret_seed")
    assert isinstance(idx, int) and idx >= 0, "idx should be non-negative int"

    adrs = ADRS()
    adrs.set_type(ADDR_TYPE_OTS)
    adrs.set_ots_address(idx)
    # set_type already zeroed chain/hash/keyAndMask, matching getSeed.
    return PRF(secret_seed, adrs.to_bytes())


def compute_leaf(idx, public_seed, secret_seed):
    '''
    Compute one XMSS leaf from its index.

    Generates the leaf's WOTS+ key pair and compresses the public key with the
    L-tree into a single leaf node.

    Args:
        idx: Leaf index.
        public_seed: 32-byte public seed shared across the tree.
        secret_seed: 32-byte master secret seed.

    Returns:
        The 32-byte leaf node.
    '''
    leaf_seed = leaf_secret_seed(secret_seed, idx)
    # wots.key_gen reused to produce the leaf's WOTS+ public key, with the OTS
    # address word set to the leaf index for per-leaf domain separation.
    wots_pk, _ = wots.key_gen(public_seed, leaf_seed, ots_address=idx)

    adrs = ADRS()
    adrs.set_type(ADDR_TYPE_LTREE)
    set_ltree_address(adrs, idx)
    return ltree(wots_pk, public_seed, adrs)


def build_hash_tree(leaves, public_seed):
    '''
    Build the full Merkle hash tree from the leaf layer up to the root.

    Args:
        leaves: List of `2^h` 32-byte leaf nodes. The count must be a power
            of two so the tree is balanced.
        public_seed: 32-byte public seed.

    Returns:
        A list of layers, where `tree[0]` is the leaf layer and `tree[-1]` is
        the single-element root layer. Keeping every layer lets signing extract
        authentication paths without recomputing the tree.

    Raises:
        AssertionError: If the number of leaves is not a power of two.
    '''
    assert len(leaves) >= 1 and (len(leaves) & (len(leaves) - 1)) == 0, \
        "number of leaves must be a power of two"

    tree = [list(leaves)]
    adrs = ADRS()
    adrs.set_type(ADDR_TYPE_HASHTREE)

    height = 0
    level = tree[0]
    while len(level) > 1:
        next_level = []
        for i in range(len(level) // 2):
            set_tree_height(adrs, height)
            set_tree_index(adrs, i)
            next_level.append(rand_hash(level[2 * i], level[2 * i + 1], public_seed, adrs))
        tree.append(next_level)
        level = next_level
        height += 1
    return tree


def get_auth_path(tree, idx):
    '''
    Extract the authentication path for a leaf from a pre-built hash tree.

    The authentication path is the list of sibling nodes encountered while
    climbing from the leaf to the root; a verifier uses it to recompute the
    root.

    Args:
        tree: Tree layers returned by `build_hash_tree`.
        idx: Leaf index to authenticate.

    Returns:
        A list of `h` sibling nodes, ordered bottom-up.

    Raises:
        AssertionError: If `idx` is out of range for the tree.
    '''
    height = len(tree) - 1
    assert 0 <= idx < len(tree[0]), "leaf index out of range"

    path = []
    node_idx = idx
    for h in range(height):
        path.append(tree[h][node_idx ^ 1])
        node_idx >>= 1
    return path


def root_from_auth_path(leaf, idx, auth_path, public_seed):
    '''
    Recompute the Merkle root from a leaf and its authentication path.

    At each layer the current node is combined with its sibling from the
    authentication path; the leaf index decides whether the current node is the
    left or right child.

    Args:
        leaf: 32-byte leaf node (the L-tree root of the reconstructed WOTS+ pk).
        idx: Leaf index.
        auth_path: List of sibling nodes returned by `get_auth_path`.
        public_seed: 32-byte public seed.

    Returns:
        The 32-byte candidate root.

    Raises:
        AssertionError: If `leaf` is not 32 bytes.
    '''
    check_bytes(leaf, "leaf")

    adrs = ADRS()
    adrs.set_type(ADDR_TYPE_HASHTREE)

    node = leaf
    node_idx = idx
    for h, sibling in enumerate(auth_path):
        set_tree_height(adrs, h)
        set_tree_index(adrs, node_idx >> 1)
        if node_idx % 2 == 0:
            node = rand_hash(node, sibling, public_seed, adrs)
        else:
            node = rand_hash(sibling, node, public_seed, adrs)
        node_idx >>= 1
    return node


def randomized_msg_digits(file_name, r, root, idx):
    '''
    Compute the randomized XMSS message digits for a file.

    Implements the RFC 8391 randomized message hash
    `M' = H_msg(r || root || toByte(idx, n), M)` and converts the resulting
    digest into the WOTS+ base-w digit vector. Binding the digest to the
    per-signature randomizer `r`, the public `root`, and the leaf index `idx`
    is what defends XMSS against multi-target and message-forgery attacks; a
    plain unkeyed hash of the message would not.

    Args:
        file_name: Name of the message file under the local plaintext folder.
        r: 32-byte per-signature randomizer, `PRF(SK_PRF, toByte(idx, 32))`.
        root: 32-byte XMSS public root.
        idx: Leaf index used for this signature.

    Returns:
        The WOTS+ digit vector `B` of length `NUM_KEY`.

    Raises:
        AssertionError: If `r` or `root` is not 32 bytes.
    '''
    check_bytes(r, "r")
    check_bytes(root, "root")

    h_msg_key = r + root + idx.to_bytes(KEY_SIZE, "big")
    digest = H_msg(h_msg_key, read_file(file_name))
    return to_digits(digest)


def key_gen(height, public_seed=None, secret_seed=None, sk_prf=None):
    '''
    Generate an XMSS key pair for a tree of the given height.

    Builds all `2^height` WOTS+ leaves, folds them into a Merkle tree, and
    takes the root as the public key. The full tree is cached in the private
    key so that signing can extract authentication paths cheaply.

    Args:
        height: Tree height h. The tree holds `2^h` one-time WOTS+ key pairs.
        public_seed: Optional 32-byte public seed (SEED). Random if None.
        secret_seed: Optional 32-byte master secret seed (SK_SEED). Random if
            None.
        sk_prf: Optional 32-byte secret PRF key (SK_PRF) used to derive the
            per-signature message randomizer `r`. Random if None.

    Returns:
        A tuple `(pk, sk)`:
            pk = {"root", "public_seed", "height"}
            sk = {"secret_seed", "sk_prf", "public_seed", "root",
                  "height", "tree"}

    Raises:
        AssertionError: If `height` is not a positive int, or a supplied seed
            is not 32 bytes.
    '''
    assert isinstance(height, int) and height >= 1, "height must be a positive int"

    if public_seed is None:
        public_seed = secrets.token_bytes(KEY_SIZE)
    if secret_seed is None:
        secret_seed = secrets.token_bytes(KEY_SIZE)
    if sk_prf is None:
        sk_prf = secrets.token_bytes(KEY_SIZE)
    check_bytes(public_seed, "public_seed")
    check_bytes(secret_seed, "secret_seed")
    check_bytes(sk_prf, "sk_prf")

    num_leaves = 1 << height
    leaves = [compute_leaf(i, public_seed, secret_seed) for i in range(num_leaves)]
    tree = build_hash_tree(leaves, public_seed)
    root = tree[-1][0]

    pk = {"root": root, "public_seed": public_seed, "height": height}
    sk = {
        "secret_seed": secret_seed,
        "sk_prf": sk_prf,
        "public_seed": public_seed,
        "root": root,
        "height": height,
        "tree": tree,
    }
    return pk, sk


def sign(file_name, sk, idx):
    '''
    Produce an XMSS signature over a file using the leaf at index `idx`.

    Signs the file with the one-time WOTS+ key pair at leaf `idx` and attaches
    that leaf's authentication path so a verifier can rebuild the root.

    Args:
        file_name: Name of the file to sign, under the local plaintext folder.
        sk: XMSS private key returned by `key_gen`.
        idx: Leaf index of the one-time key pair to use.

    Returns:
        A signature dict {"idx", "r", "wots_sig", "auth_path"}, where `r` is
        the per-signature randomizer needed to recompute the message digest.

    Raises:
        AssertionError: If `idx` is out of range for the tree.
    '''
    height = sk["height"]
    assert 0 <= idx < (1 << height), "leaf index out of range"

    public_seed = sk["public_seed"]

    # RFC 8391 randomized message hash: r = PRF(SK_PRF, toByte(idx, 32)) keys
    # the digest together with the root and the leaf index.
    r = PRF(sk["sk_prf"], idx.to_bytes(KEY_SIZE, "big"))
    B = randomized_msg_digits(file_name, r, sk["root"], idx)

    leaf_seed = leaf_secret_seed(sk["secret_seed"], idx)
    # wots.key_gen reused for the one-time key pair; the OTS address word is the
    # leaf index, and sign_digits signs the randomized digest at that address.
    _, leaf_wots_sk = wots.key_gen(public_seed, leaf_seed, ots_address=idx)
    wots_sig = wots.sign_digits(B, public_seed, leaf_wots_sk, ots_address=idx)

    return {
        "idx": idx,
        "r": r,
        "wots_sig": wots_sig,
        "auth_path": get_auth_path(sk["tree"], idx),
    }


def verify(file_name, pk, signature):
    '''
    Verify an XMSS signature over the contents of a file.

    Reconstructs the WOTS+ public key from the signature, compresses it with
    the L-tree to recover the leaf, climbs the authentication path to a
    candidate root, and compares it against the public root.

    Args:
        file_name: Name of the file to verify, under the local plaintext folder.
        pk: XMSS public key returned by `key_gen`.
        signature: Signature dict returned by `sign`.

    Returns:
        True if the signature is valid under `pk`, False otherwise.

    Raises:
        AssertionError: If `idx` is out of range for the tree.
    '''
    public_seed = pk["public_seed"]
    root = pk["root"]
    idx = signature["idx"]
    assert 0 <= idx < (1 << pk["height"]), "leaf index out of range"

    # Recompute the randomized digest from the signature's r, the public root,
    # and the claimed index, then complete the WOTS+ chains at OTS address idx.
    B = randomized_msg_digits(file_name, signature["r"], root, idx)
    wots_pk = wots.pk_from_sig_digits(B, public_seed, signature["wots_sig"], ots_address=idx)

    adrs = ADRS()
    adrs.set_type(ADDR_TYPE_LTREE)
    set_ltree_address(adrs, idx)
    leaf = ltree(wots_pk, public_seed, adrs)

    candidate_root = root_from_auth_path(leaf, idx, signature["auth_path"], public_seed)
    return candidate_root == root


# ---------------------------------------------------------------------------
# Canonical serialization.
#
# The byte layouts below follow the RFC 8391 ordering so the structures can be
# moved over the wire or to disk as flat byte strings instead of Python dicts.
# A 4-byte big-endian height field stands in for the RFC's algorithm OID,
# making each blob self-describing about its tree size.
#
#   public key : toByte(h, 4) || root(n) || SEED(n)
#   signature  : toByte(idx, 4) || r(n) || wots_sig(len*n) || auth(h*n)
#   private key: toByte(h, 4) || SK_SEED(n) || SK_PRF(n) || root(n) || SEED(n)
# ---------------------------------------------------------------------------


def serialize_signature(signature):
    '''
    Serialize an XMSS signature dict into its canonical byte layout.

    Layout: `toByte(idx, 4) || r || wots_sig || auth_path`, with every node
    32 bytes. The height is recoverable from the total length, so it is not
    stored here.

    Args:
        signature: Signature dict returned by `sign`.

    Returns:
        The signature encoded as `bytes`.
    '''
    check_bytes(signature["r"], "r")
    wots.check_key(signature["wots_sig"], "wots_sig")
    for i, node in enumerate(signature["auth_path"]):
        check_bytes(node, f"auth_path[{i}]")

    return b"".join([
        signature["idx"].to_bytes(4, "big"),
        signature["r"],
        b"".join(signature["wots_sig"]),
        b"".join(signature["auth_path"]),
    ])


def deserialize_signature(data):
    '''
    Parse a canonical signature blob back into a signature dict.

    Args:
        data: Bytes produced by `serialize_signature`.

    Returns:
        A signature dict {"idx", "r", "wots_sig", "auth_path"}.

    Raises:
        AssertionError: If `data` has an inconsistent length.
    '''
    assert isinstance(data, bytes), "data must be bytes"
    n = KEY_SIZE
    fixed = 4 + n + NUM_KEY * n
    assert len(data) >= fixed, "signature blob too short"
    assert (len(data) - fixed) % n == 0, "signature blob has a partial auth node"

    idx = int.from_bytes(data[0:4], "big")
    off = 4
    r = data[off:off + n]
    off += n
    wots_sig = [data[off + i * n: off + (i + 1) * n] for i in range(NUM_KEY)]
    off += NUM_KEY * n
    height = (len(data) - off) // n
    auth_path = [data[off + i * n: off + (i + 1) * n] for i in range(height)]

    return {"idx": idx, "r": r, "wots_sig": wots_sig, "auth_path": auth_path}


def serialize_public_key(pk):
    '''
    Serialize an XMSS public key into its canonical byte layout.

    Layout: `toByte(height, 4) || root || public_seed`.

    Args:
        pk: Public key dict returned by `key_gen`.

    Returns:
        The public key encoded as `bytes`.
    '''
    check_bytes(pk["root"], "root")
    check_bytes(pk["public_seed"], "public_seed")
    return pk["height"].to_bytes(4, "big") + pk["root"] + pk["public_seed"]


def deserialize_public_key(data):
    '''
    Parse a canonical public-key blob back into a public-key dict.

    Args:
        data: Bytes produced by `serialize_public_key`.

    Returns:
        A public key dict {"root", "public_seed", "height"}.

    Raises:
        AssertionError: If `data` has the wrong length.
    '''
    assert isinstance(data, bytes), "data must be bytes"
    n = KEY_SIZE
    assert len(data) == 4 + 2 * n, "public-key blob has the wrong length"

    height = int.from_bytes(data[0:4], "big")
    root = data[4:4 + n]
    public_seed = data[4 + n:4 + 2 * n]
    return {"root": root, "public_seed": public_seed, "height": height}


def serialize_private_key(sk):
    '''
    Serialize the secret material of an XMSS private key.

    Layout: `toByte(height, 4) || SK_SEED || SK_PRF || root || SEED`. The cached
    hash tree is *not* stored; `deserialize_private_key` rebuilds it from the
    seeds, which is what keeps this format canonical and compact.

    Args:
        sk: Private key dict returned by `key_gen`.

    Returns:
        The private key encoded as `bytes`.
    '''
    check_bytes(sk["secret_seed"], "secret_seed")
    check_bytes(sk["sk_prf"], "sk_prf")
    check_bytes(sk["root"], "root")
    check_bytes(sk["public_seed"], "public_seed")
    return b"".join([
        sk["height"].to_bytes(4, "big"),
        sk["secret_seed"],
        sk["sk_prf"],
        sk["root"],
        sk["public_seed"],
    ])


def deserialize_private_key(data):
    '''
    Parse a canonical private-key blob and rebuild the full private key.

    Regenerates the tree from the stored seeds via `key_gen` and checks that
    the recomputed root matches the stored root, catching corrupted blobs.

    Args:
        data: Bytes produced by `serialize_private_key`.

    Returns:
        A private key dict identical in shape to the one `key_gen` returns.

    Raises:
        AssertionError: If `data` has the wrong length or the rebuilt root does
            not match the stored root.
    '''
    assert isinstance(data, bytes), "data must be bytes"
    n = KEY_SIZE
    assert len(data) == 4 + 4 * n, "private-key blob has the wrong length"

    height = int.from_bytes(data[0:4], "big")
    off = 4
    secret_seed = data[off:off + n]; off += n
    sk_prf = data[off:off + n]; off += n
    root = data[off:off + n]; off += n
    public_seed = data[off:off + n]

    _, sk = key_gen(height, public_seed, secret_seed, sk_prf)
    assert sk["root"] == root, "rebuilt root does not match the stored root"
    return sk


def test_roundtrip():
    '''
    Test that a valid signature verifies and a tampered one fails.
    '''
    pk, sk = key_gen(height=3)
    sig = sign("short.txt", sk, idx=5)
    assert verify("short.txt", pk, sig) is True

    bad_sig = dict(sig)
    bad_sig["wots_sig"] = list(sig["wots_sig"])
    bad_sig["wots_sig"][0] = bytes(KEY_SIZE)
    assert verify("short.txt", pk, bad_sig) is False
    print("roundtrip test passed")


def test_all_leaves():
    '''
    Test that every leaf in a small tree produces a valid signature.
    '''
    height = 3
    pk, sk = key_gen(height=height)
    for idx in range(1 << height):
        sig = sign("short.txt", sk, idx)
        assert verify("short.txt", pk, sig) is True, f"leaf {idx} failed to verify"
    print("all-leaves test passed")


def test_deterministic_key_gen():
    '''
    Test that seeded key generation is deterministic.
    '''
    public_seed = b"public_seed" + b"\x00" * 21
    secret_seed = b"secret_seed" + b"\x00" * 21
    pk1, _ = key_gen(3, public_seed, secret_seed)
    pk2, _ = key_gen(3, public_seed, secret_seed)
    assert pk1["root"] == pk2["root"], "same seeds should give the same root"
    print("deterministic key generation test passed")


def test_wrong_index():
    '''
    Test that a signature does not verify against a forged leaf index.
    '''
    pk, sk = key_gen(height=3)
    sig = sign("short.txt", sk, idx=2)
    forged = dict(sig)
    forged["idx"] = 3
    assert verify("short.txt", pk, forged) is False
    print("wrong-index test passed")


def test_h_msg_randomization():
    '''
    Test that H_msg randomization is wired in and tamper-evident.

    Each leaf gets a distinct randomizer r, and verification depends on r being
    correct, so swapping it (or the digest it keys) must break the signature.
    '''
    pk, sk = key_gen(height=3)
    sig0 = sign("short.txt", sk, idx=0)
    sig1 = sign("short.txt", sk, idx=1)
    assert sig0["r"] != sig1["r"], "different indices must use different r"

    tampered = dict(sig0)
    tampered["r"] = sig1["r"]
    assert verify("short.txt", pk, tampered) is False, "wrong r must not verify"
    print("H_msg randomization test passed")


def test_serialization():
    '''
    Test that keys and signatures survive a serialize/deserialize round trip.
    '''
    pk, sk = key_gen(height=3)
    sig = sign("short.txt", sk, idx=4)

    pk2 = deserialize_public_key(serialize_public_key(pk))
    assert pk2 == pk, "public-key round trip changed the key"

    sig2 = deserialize_signature(serialize_signature(sig))
    assert sig2 == sig, "signature round trip changed the signature"
    assert verify("short.txt", pk2, sig2) is True, "round-tripped data must verify"

    sk2 = deserialize_private_key(serialize_private_key(sk))
    resigned = sign("short.txt", sk2, idx=6)
    assert verify("short.txt", pk, resigned) is True, "rebuilt sk must still sign"
    print("serialization test passed")


if __name__ == "__main__":
    test_deterministic_key_gen()
    test_roundtrip()
    test_all_leaves()
    test_wrong_index()
    test_h_msg_randomization()
    test_serialization()
    print("all test passed")
