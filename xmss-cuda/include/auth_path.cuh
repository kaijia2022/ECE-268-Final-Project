#ifndef AUTH_PATH_CUH
#define AUTH_PATH_CUH

#include <stdint.h>
#include "adrs.h"

// Computes the Authentication Path for a specific leaf index.
// tree_buffer: Pre-populated array of all leaf nodes (destroyed during execution)
// num_leaves: Must be a power of 2 for standard XMSS (e.g., 1024)
// leaf_index: The index of the WOTS+ keypair used to sign the message
// pub_seed: 32-byte public seed
// auth_path_out: Output array of size (Tree_Height * 32) bytes
__device__ void extract_auth_path(uint8_t* tree_buffer, uint32_t num_leaves, uint32_t leaf_index, const uint8_t* pub_seed, ADRS* adrs, uint8_t* auth_path_out);

// Calls extract_auth_path
__global__ void extract_auth_path_kernel(uint8_t* tree_buffer, uint32_t num_leaves, uint32_t leaf_index, const uint8_t* pub_seed, uint8_t* auth_path_out);

#endif // AUTH_PATH_CUH