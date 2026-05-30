#ifndef XMSS_CUH
#define XMSS_CUH

#include <stdint.h>
#include <stdbool.h>

struct XMSS_Signature {
    uint32_t leaf_idx;
    uint8_t msg_randomness[32];
    uint8_t wots_sig[67 * 32]; 
    uint8_t* auth_path;        
    uint32_t auth_path_len;
};

// Frontend API: Sign a message
void xmss_sign(
    const uint8_t* msg, size_t msg_len, 
    uint32_t leaf_idx, uint32_t tree_height,
    const uint8_t* secret_seeds, const uint8_t* pub_seed, 
    uint8_t* precomputed_tree_buffer,
    XMSS_Signature* out_sig
);

// Frontend API: Verify a signature
// Returns true if valid, false if invalid
bool xmss_verify(
    const XMSS_Signature* sig,
    const uint8_t* msg, size_t msg_len,
    const uint8_t* expected_pub_key, 
    uint32_t tree_height,
    const uint8_t* pub_seed
);

#endif // XMSS_CUH