#include "merkle.cuh"
#include "hash_wrappers.cuh"

__device__ void build_hash_tree(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, ADRS* adrs, uint8_t* root_out) {
    uint32_t len = num_leaves;
    uint32_t height = 0;

    // Set the address type to Hash Tree
    adrs->setType(XMSS_ADDR_TYPE_HASHTREE);

    // Compress layer by layer until 1 node remains (similar to L-Tree)
    while (len > 1) {                                     
        for (uint32_t i = 0; i < (len >> 1); i++) {         
            adrs->setTreeHeight(height);                    
            adrs->setTreeIndex(i);

            h_hash(pub_seed, adrs,                         
                   &tree_buffer[(2 * i) * 32],              
                   &tree_buffer[(2 * i + 1) * 32], 
                   (SHA256_CTX*)&tree_buffer[i * 32]);
        }

        // (should never reach here since Merkle Tree should be balanced => len % 2 == 0 immer)
        if (len % 2 == 1) {                                
            for (int j = 0; j < 32; j++) {                  
                tree_buffer[(len >> 1) * 32 + j] = tree_buffer[(len - 1) * 32 + j];
            }
        }

        len = (len >> 1) + (len % 2);
        height++;
    }

    for (int i = 0; i < 32; i++) {
        root_out[i] = tree_buffer[i];
    }
}