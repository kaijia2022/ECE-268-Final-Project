#ifndef SHA256_CUH
#define SHA256_CUH

#include <stdint.h>


struct SHA256_CTX {
    uint8_t hash[32];
};

__device__ void sha256(const uint8_t* data, size_t len, SHA256_CTX* output);

#endif // SHA256_CUH