#ifndef XMSS_H
#define XMSS_H

// XMSS the stateful merkle signature scheme
// It stacks many WOTS+ leaves into one tree and exposes a single public root
// State management makes sure every signature uses a fresh leaf
// because reusing a WOTS+ leaf leaks its private key

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "wots.h"

// Holds one XMSS keypair plus the state it needs to stay safe
// next_idx is the state we remember which leaf comes next and never
// sign with the same one twice
typedef struct {
    uint32_t tree_height;
    uint32_t num_leaves;
    uint32_t next_idx;        // next unused leaf this is the state
    uint8_t  sk_seed[N];      // master secret derives every wots secret seed
    uint8_t  sk_prf[N];       // secret used to randomize the message hash
    uint8_t  pub_seed[N];     // public seed for the bitmasks
    uint8_t  root[N];         // public key the merkle root
    uint8_t* tree_buffer;     // cached leaf nodes used to build auth paths
} XMSS_Keypair;

typedef struct {
    uint32_t leaf_idx;
    uint8_t  msg_randomness[N];
    uint8_t  wots_sig[WOTS_LEN * N];
    uint8_t* auth_path;
    uint32_t auth_path_len;
} XMSS_Signature;

// KeyGen build a full real tree from the seeds and fill in the keypair
// computes every leaf for real builds the merkle root and resets the state
void xmss_keygen(XMSS_Keypair* kp, uint32_t tree_height,
                 const uint8_t sk_seed[N], const uint8_t sk_prf[N], const uint8_t pub_seed[N]);

// free the leaf cache we malloced in keygen
void xmss_keypair_free(XMSS_Keypair* kp);

// Sign consume the next free leaf hash the message for real and advance the state
// returns false if the tree is used up so we never reuse a leaf
bool xmss_sign(XMSS_Keypair* kp, const uint8_t* msg, size_t msg_len, XMSS_Signature* out_sig);

// Verify returns true if the signature is valid false if not
bool xmss_verify(const XMSS_Signature* sig, const uint8_t* msg, size_t msg_len,
                 const uint8_t expected_pub_key[N], uint32_t tree_height, const uint8_t pub_seed[N]);

#endif // XMSS_H
