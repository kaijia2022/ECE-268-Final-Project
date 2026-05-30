#ifndef WOTS_CUH
#define WOTS_CUH

#include <stdint.h>
#include "adrs.h"

// Hashes a WOTS+ start_value forward by 'steps' iterations. (max 15 steps)
// start_idx: the current index in the chain (usually 0 when generating keys/signing)
__device__ void wots_chain(const uint8_t* start_value, uint32_t start_idx, uint32_t steps, 
                           const uint8_t* pub_seed, ADRS* adrs, uint8_t* output);

// Takes a 32-byte message hash and converts it into 67 chain lengths (0-15)
// msg_hash: 32-byte input array
// lengths: 67-byte output array where each byte represents the length of a WOTS+ chain
__host__ __device__ void get_chain_lengths(const uint8_t* msg_hash, uint8_t* lengths);

// The WOTS+ Signing Kernel
__global__ void wots_sign_kernel(const uint8_t* msg_hash, const uint8_t* secret_seeds, const uint8_t* pub_seed, uint32_t leaf_idx, uint8_t* wots_signature);

// The WOTS+ Verifying Kernel
__global__ void wots_verify_kernel(const uint8_t* msg_hash, const uint8_t* wots_sig, const uint8_t* pub_seed, uint32_t leaf_idx, uint8_t* wots_pk_out);

#endif // WOTS_CUH