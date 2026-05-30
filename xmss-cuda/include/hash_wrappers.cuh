#ifndef HASH_WRAPPERS_CUH
#define HASH_WRAPPERS_CUH

#include <stdint.h>
#include "sha256.cuh"
#include "adrs.h"

// PRF (Pseudorandom Function): Used to generate keys and bitmasks from a seed and an ADRS
// seed: 32 bytes, adrs: 32 bytes (struct), output: 32 bytes
__device__ void prf(const uint8_t* pub_seed, const ADRS* adrs, SHA256_CTX* output);

// F (Chaining Function): Used inside the WOTS+ loops to hash the chain forward
// key: 32 bytes, in_data: 32 bytes, output: 32 bytes
__device__ void f_chain(const uint8_t* key, const uint8_t* in_data, SHA256_CTX* output);

// H (Randomized Hash Function): Used in L-Tree and Main Tree to combine two nodes
// left: 32 bytes, right: 32 bytes, output: 32 bytes
__device__ void h_hash(const uint8_t* pub_seed, ADRS* adrs, const uint8_t* left, const uint8_t* right, SHA256_CTX* output);

#endif // HASH_WRAPPERS_CUH