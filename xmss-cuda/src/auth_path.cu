#include "auth_path.cuh"
#include "hash_wrappers.cuh"

__device__ void extract_auth_path(uint8_t* tree_buffer, uint32_t num_leaves, uint32_t leaf_index, const uint8_t* pub_seed, ADRS* adrs, uint8_t* auth_path_out) {
    uint32_t len = num_leaves;
    uint32_t height = 0;
    
    // Keeps track of our target node's position as we climb the tree
    uint32_t current_target_idx = leaf_index;

    adrs->setType(XMSS_ADDR_TYPE_HASHTREE);

    while (len > 1) {
        // 1. Save the sibling for the authentication path
        // The sibling is always (index XOR 1) => Flip the lsb
        uint32_t sibling_idx = current_target_idx ^ 1; //e.g.current_target_idx = 2 (0b010), sibling_idx = 2 ^ 1 (0b001) = 3 (0b011)
        
        for (int j = 0; j < 32; j++) {
            auth_path_out[height * 32 + j] = tree_buffer[sibling_idx * 32 + j];
        }

        // Standard Hash Tree Logic => Compress the layer
        for (uint32_t i = 0; i < (len >> 1); i++) {
            adrs->setTreeHeight(height);
            adrs->setTreeIndex(i);

            h_hash(pub_seed, adrs, 
                   &tree_buffer[(2 * i) * 32], 
                   &tree_buffer[(2 * i + 1) * 32], 
                   (SHA256_CTX*)&tree_buffer[i * 32]);
        }

        if (len % 2 == 1) {
            for (int j = 0; j < 32; j++) {
                tree_buffer[(len >> 1) * 32 + j] = tree_buffer[(len - 1) * 32 + j];
            }
        }

        // Move up to the next layer
        // The parent of index 'X' is at index 'X / 2' (or X >> 1)
        current_target_idx >>= 1; 
        len = (len >> 1) + (len % 2);
        height++;
    }
}

__global__ void extract_auth_path_kernel(uint8_t* tree_buffer, uint32_t num_leaves, uint32_t leaf_index, const uint8_t* pub_seed, uint8_t* auth_path_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ADRS adrs;
        adrs.setLayerAddress(0);
        adrs.setTreeAddress(1);

        extract_auth_path(tree_buffer, num_leaves, leaf_index, pub_seed, &adrs, auth_path_out);
    }
}