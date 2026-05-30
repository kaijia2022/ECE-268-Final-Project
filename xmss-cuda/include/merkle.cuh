#ifndef MERKLE_CUH
#define MERKLE_CUH

#include <stdint.h>
#include "adrs.h"

// Builds the Main Merkle Hash Tree.
// tree_buffer: A globally allocated array containing the leaf nodes (overwritten during execution)
// num_leaves: The total number of leaf nodes (must be a power of 2 for standard XMSS) <=> Merkle Tree is balanced
// pub_seed: 32-byte public seed
// root_out: 32-byte buffer to store the final Merkle root
__device__ void build_hash_tree(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, ADRS* adrs, uint8_t* root_out);

#endif // MERKLE_CUH