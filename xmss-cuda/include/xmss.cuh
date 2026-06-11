#ifndef XMSS_CUH
#define XMSS_CUH

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

struct XMSS_Signature {
    uint32_t leaf_idx;
    uint8_t msg_randomness[32];
    uint8_t wots_sig[67 * 32];
    uint8_t* auth_path;
    uint32_t auth_path_len;
};

// Holds one XMSS keypair plus the state it needs to stay safe.
// next_idx is the state. Every signature must use a fresh leaf so we
// remember which leaf comes next and never sign with the same one twice.
struct XMSS_Keypair {
    uint32_t tree_height;
    uint32_t num_leaves;
    uint32_t next_idx;         // next unused one time leaf, this is the state
    uint8_t sk_seed[32];       // master secret, derives every wots secret seed
    uint8_t sk_prf[32];        // secret used to randomize the message hash
    uint8_t pub_seed[32];      // public seed for the bitmasks
    uint8_t root[32];          // public key, the merkle root
    uint8_t* tree_buffer;      // cached leaf nodes, num_leaves * 32, used to build auth paths
};

// KeyGen: build a full real tree from the seeds and fill in the keypair.
// Computes every leaf for real, builds the merkle root, and resets the state.
void xmss_keygen(
    XMSS_Keypair* kp, uint32_t tree_height,
    const uint8_t* sk_seed, const uint8_t* sk_prf, const uint8_t* pub_seed
);

// Free the leaf cache we malloced in keygen.
void xmss_keypair_free(XMSS_Keypair* kp);

// Sign: consume the next free leaf, hash the message for real, and advance the state.
// Returns false if the tree is used up so we never reuse a leaf.
bool xmss_sign(
    XMSS_Keypair* kp,
    const uint8_t* msg, size_t msg_len,
    XMSS_Signature* out_sig
);

// Verify: returns true if the signature is valid, false if not.
bool xmss_verify(
    const XMSS_Signature* sig,
    const uint8_t* msg, size_t msg_len,
    const uint8_t* expected_pub_key,
    uint32_t tree_height,
    const uint8_t* pub_seed
);

#endif // XMSS_CUH
