#ifndef ADRS_H
#define ADRS_H

#include <stdint.h>

#define XMSS_ADDR_TYPE_OTS       0
#define XMSS_ADDR_TYPE_LTREE     1
#define XMSS_ADDR_TYPE_HASHTREE  2

struct ADRS {
    uint32_t a[8]; // 8 * 4 bytes = 32 bytes

    __host__ __device__ ADRS() {
        for (int i = 0; i < 8; i++) a[i] = 0;
    }

    // Word 0: Layer Address
    __host__ __device__ void setLayerAddress(uint32_t layer) { a[0] = layer; }

    // Words 1-2: Tree Address (64-bit)
    __host__ __device__ void setTreeAddress(uint64_t tree) {
        a[1] = (uint32_t)(tree >> 32);
        a[2] = (uint32_t)tree;
    }

    // Word 3: Type
    __host__ __device__ void setType(uint32_t type) {
        a[3] = type;
        a[4] = 0; a[5] = 0; a[6] = 0; a[7] = 0;
    }

    // WOTS+ Specific Fields (Type = 0) 
    __host__ __device__ void setOTSAddress(uint32_t ots) { a[4] = ots; }
    __host__ __device__ void setChainAddress(uint32_t chain) { a[5] = chain; }
    __host__ __device__ void setHashAddress(uint32_t hash) { a[6] = hash; }
    __host__ __device__ void setKeyAndMask(uint32_t keyAndMask) { a[7] = keyAndMask; }

    // L-Tree/Merkle Tree Specific Fields (Types 1 & 2)
    
    // Word 4: L-Tree Address (Only used when Type = 1)
    __host__ __device__ void setLTreeAddress(uint32_t ltree) { a[4] = ltree; }
    
    // Word 5: Tree Height
    __host__ __device__ void setTreeHeight(uint32_t height) { a[5] = height; }
    
    // Word 6: Tree Index
    __host__ __device__ void setTreeIndex(uint32_t index) { a[6] = index; }
};

#endif // ADRS_H