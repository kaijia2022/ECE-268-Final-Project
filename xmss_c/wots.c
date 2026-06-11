// WOTS+ and the low level crypto it sits on
// This is a strict port of the xmss-cuda sha256 hash_wrappers and wots code
// Same ADRS layout same domain separators same chain with bitmasks
// Goal is a fair CPU baseline for the xmss-cuda GPU version
// We use a software SHA256 here just like the GPU kernels so the
// only real difference measured is the GPU parallelism

#include "wots.h"
#include <string.h>

// ---------- SHA256 ----------
// Plain software SHA256 same math as the GPU kernel

static const uint32_t Kc[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

static inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// compress one 64 byte block into the running state h
static void sha256_block(uint32_t h[8], const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i*4]   << 24) | ((uint32_t)blk[i*4+1] << 16) |
               ((uint32_t)blk[i*4+2] <<  8) | ((uint32_t)blk[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + Kc[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// running tally of how many times sha256 was called
// tests reset this then read it back to count hashes per op
uint64_t g_sha256_count = 0;

// hash an arbitrary length message and write 32 bytes out
// This mirrors the xmss-cuda sha256 byte for byte on purpose
// Watch the length field the kernel adds 512 per full block while streaming
// and then also adds the whole message length at the end
// That double counts the full blocks so for inputs of 64 bytes or more the
// length field is not standard sha256
// We copy the same quirk here so the C output matches the GPU output exactly
// otherwise the two would never agree on a single hash
void sha256(const uint8_t* msg, size_t len, uint8_t out[N]) {
    g_sha256_count++;
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint8_t buffer[64];
    uint32_t datalen = 0;
    uint64_t bitlen = 0;

    // stream the message a byte at a time and flush every full 64 byte block
    for (size_t i = 0; i < len; i++) {
        buffer[datalen++] = msg[i];
        if (datalen == 64) {
            sha256_block(h, buffer);
            bitlen += 512;
            datalen = 0;
        }
    }

    // append the 0x80 marker then pad if there is no room for the length
    buffer[datalen++] = 0x80;
    if (datalen > 56) {
        while (datalen < 64) buffer[datalen++] = 0x00;
        sha256_block(h, buffer);
        datalen = 0;
    }

    // zero up to byte 56 then write the length field big endian
    while (datalen < 56) buffer[datalen++] = 0x00;
    bitlen += (uint64_t)len * 8;
    for (int i = 63; i >= 56; i--) { buffer[i] = (uint8_t)bitlen; bitlen >>= 8; }
    sha256_block(h, buffer);

    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(h[i] >> 24);
        out[i*4+1] = (uint8_t)(h[i] >> 16);
        out[i*4+2] = (uint8_t)(h[i] >>  8);
        out[i*4+3] = (uint8_t)(h[i]);
    }
}

// ---------- hash wrappers ----------
// Domain separated hashes all built on SHA256
// toByte 0 32 for F the chain step toByte 3 32 for PRF

// turn the 8 words into 32 big endian bytes for hashing
static void adrs_to_bytes(const ADRS* x, uint8_t out[N]) {
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(x->a[i] >> 24);
        out[i*4+1] = (uint8_t)(x->a[i] >> 16);
        out[i*4+2] = (uint8_t)(x->a[i] >>  8);
        out[i*4+3] = (uint8_t)(x->a[i]);
    }
}

// PRF KEY M is SHA256 of toByte 3 32 then KEY then the 32 byte ADRS
void prf(const uint8_t key[N], const ADRS* adrs, uint8_t out[N]) {
    uint8_t buf[96];
    memset(buf, 0, 31);
    buf[31] = 0x03;
    memcpy(buf + 32, key, 32);
    adrs_to_bytes(adrs, buf + 64);
    sha256(buf, 96, out);
}

// F key in is SHA256 of toByte 0 32 then key then the previous chain node
static void f_chain(const uint8_t key[N], const uint8_t in_data[N], uint8_t out[N]) {
    uint8_t buf[96];
    memset(buf, 0, 32);
    memcpy(buf + 32, key, 32);
    memcpy(buf + 64, in_data, 32);
    sha256(buf, 96, out);
}

// ---------- WOTS+ ----------

// walk a chain forward starting from start_value for the given number of steps
// each step masks the current node then runs it through F
void wots_chain(const uint8_t* start_value, uint32_t start_idx, uint32_t steps,
                const uint8_t pub_seed[N], ADRS* adrs, uint8_t out[N]) {
    memcpy(out, start_value, N);
    for (uint32_t i = start_idx; i < start_idx + steps; i++) {
        adrs_set_hash(adrs, i);

        uint8_t bitmask[N], step_key[N];
        adrs_set_keymask(adrs, 1); prf(pub_seed, adrs, bitmask);
        adrs_set_keymask(adrs, 0); prf(pub_seed, adrs, step_key);

        uint8_t xored[N];
        for (int j = 0; j < 32; j++) xored[j] = out[j] ^ bitmask[j];
        f_chain(step_key, xored, out);
    }
}

// turn a 32 byte digest into the 67 chain lengths
// 64 nibbles from the digest then 3 nibbles of checksum
void get_chain_lengths(const uint8_t msg_hash[N], uint8_t lengths[WOTS_LEN]) {
    uint32_t csum = 0;
    for (int i = 0; i < 32; i++) {
        uint8_t byte = msg_hash[i];
        lengths[2*i]     = byte >> 4;
        lengths[2*i + 1] = byte & 0x0F;
        csum += (15 - lengths[2*i]);
        csum += (15 - lengths[2*i + 1]);
    }
    lengths[64] = (csum >> 8) & 0x0F;
    lengths[65] = (csum >> 4) & 0x0F;
    lengths[66] = csum & 0x0F;
}
