#include "hash_wrappers.cuh"

// toByte(0, 32) is used for the F function (leaf nodes in WOTS+);
// toByte(1, 32) is used for the H function (internal nodes in the Merkle/L-Tree);
// toByte(2, 32) is used for the H_msg function (hashing the actual message);
// toByte(3, 32) is used for the PRF function (generating keys and masks);

// Helper function to convert the 8x uint32_t ADRS struct into a 32-byte Big-Endian array.
// This is required by the XMSS specification before hashing the address.
__device__ void adrs_to_bytes_big_endian(const ADRS* adrs, uint8_t* out_bytes) {
    for (int i = 0; i < 8; i++) {
        out_bytes[i * 4 + 0] = (uint8_t)(adrs->a[i] >> 24);
        out_bytes[i * 4 + 1] = (uint8_t)(adrs->a[i] >> 16);
        out_bytes[i * 4 + 2] = (uint8_t)(adrs->a[i] >> 8);
        out_bytes[i * 4 + 3] = (uint8_t)(adrs->a[i]);
    }
}

// PRF(KEY, M) = SHA256(toByte(3, 32) || KEY || M)
__device__ void prf(const uint8_t* pub_seed, const ADRS* adrs, SHA256_CTX* output) {
    uint8_t buffer[96];
    
    // Prefix: toByte(3, 32) --> 32 Byte 0x3 in Big-Endien
    for (int i = 0; i < 31; i++) buffer[i] = 0x00;
    buffer[31] = 0x03;

    // KEY: The 32-byte public seed
    for (int i = 0; i < 32; i++) buffer[32 + i] = pub_seed[i];

    // M: The 32-byte ADRS (Converted to Big-Endian)
    adrs_to_bytes_big_endian(adrs, &buffer[64]);

    // Hash the resulting 96 bytes
    sha256(buffer, 96, output);
}

// F(KEY, M) = SHA256(toByte(0, 32) || KEY || M)
__device__ void f_chain(const uint8_t* key, const uint8_t* in_data, SHA256_CTX* output) {
    uint8_t buffer[96];

    // Prefix: toByte(0, 32) --> 32 Byte 0x0 in Big-Endien
    for (int i = 0; i < 32; i++) buffer[i] = 0x00;

    // KEY: The 32-byte step key
    for (int i = 0; i < 32; i++) buffer[32 + i] = key[i];

    // M: The 32-byte input data (the previous node in the chain)
    for (int i = 0; i < 32; i++) buffer[64 + i] = in_data[i];

    // Hash the resulting 96 bytes
    sha256(buffer, 96, output);
}

// H(KEY, M) = SHA256(toByte(1, 32) || KEY || M)
// Where M = (left ^ mask_left) || (right ^ mask_right)
__device__ void h_hash(const uint8_t* pub_seed, ADRS* adrs, const uint8_t* left, const uint8_t* right, SHA256_CTX* output) {
    uint8_t buffer[128]; // 32 prefix + 32 key + 64 data (left + right)

    // Prefix: toByte(1, 32) --> 32 Byte 0x1 in Big-Endien
    for (int i = 0; i < 31; i++) buffer[i] = 0x00;
    buffer[31] = 0x01;

    // Generate Key (KeyAndMask = 0)
    adrs->setKeyAndMask(0);
    uint8_t key[32];
    prf(pub_seed, adrs, (SHA256_CTX*)key);
    for (int i = 0; i < 32; i++) buffer[32 + i] = key[i];

    // left and right bitmasks is exactly what the "eXtended" in eXtended Merkle Signature Scheme stands for
    // Because the masks are derived from the pub_seed and the exact tree coordinates,
    // every single node in the tree is effectively using a mathematically distinct, customized hash function
    // provides Second-Preimage Resistance
    
    // Generate Left Mask (KeyAndMask = 1)
    adrs->setKeyAndMask(1);                                        
    uint8_t mask_left[32];
    prf(pub_seed, adrs, (SHA256_CTX*)mask_left);

    // Generate Right Mask (KeyAndMask = 2)
    adrs->setKeyAndMask(2);
    uint8_t mask_right[32];
    prf(pub_seed, adrs, (SHA256_CTX*)mask_right);

    // XOR the children with their masks and append to buffer
    for (int i = 0; i < 32; i++) {
        buffer[64 + i] = left[i] ^ mask_left[i];
        buffer[96 + i] = right[i] ^ mask_right[i];
    }

    sha256(buffer, 128, output);
}