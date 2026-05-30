#ifndef LTREE_CUH
#define LTREE_CUH

#include <stdint.h>
#include "adrs.h"

// Compresses an array of 67 WOTS+ public keys down to a single 32-byte leaf node
// wots_pks: input array of 67 * 32 bytes
// root_out: output array of 32 bytes
// l_tree is unbalanced
__device__ void l_tree(const uint8_t* wots_pks, const uint8_t* pub_seed, ADRS* adrs, uint8_t* root_out);

#endif // LTREE_CUH