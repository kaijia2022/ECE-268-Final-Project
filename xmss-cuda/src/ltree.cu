#include "ltree.cuh"
#include "hash_wrappers.cuh"

__device__ void l_tree(const uint8_t* wots_pks, const uint8_t* pub_seed, ADRS* adrs, uint8_t* root_out) {
    uint8_t nodes[67 * 32];

    for (int i = 0; i < 67 * 32; i++) {
        nodes[i] = wots_pks[i];
    }

    uint32_t len = 67;
    uint32_t height = 0;

    // Set Type to L-Tree
    adrs->setType(XMSS_ADDR_TYPE_LTREE);

    // Compress until only 1 node is left
    while (len > 1) {                                       //e.g. len = 5
        for (uint32_t i = 0; i < (len >> 1); i++) {         // (len >> 1) = 5/2 = 2
            adrs->setTreeHeight(height);
            adrs->setTreeIndex(i);

            // Hash nodes[2i] and nodes[2i+1] together, and overwrite nodes[i] with the result
            // i = 0: hash index 0, 1 and update index 0
            // i = 1: hash index 2, 3 and update index 1
            h_hash(pub_seed, adrs, &nodes[(2 * i) * 32], &nodes[(2 * i + 1) * 32], (SHA256_CTX*)&nodes[i * 32]);
        }

        if (len % 2 == 1) {
            for (int j = 0; j < 32; j++) {                     
                nodes[(len >> 1) * 32 + j] = nodes[(len - 1) * 32 + j];    // index 4 is left out, must lift it
            }                                                              // update index 2 with index 4 for the next level's hash
        }

        // Calculate the number of nodes for the next level 
        len = (len >> 1) + (len % 2);
        height++;
    }

    // The final surviving node is the L-Tree root
    for (int i = 0; i < 32; i++) {
        root_out[i] = nodes[i];
    }
}