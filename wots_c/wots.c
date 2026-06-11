// WOTS in plain C
// This is a strict port of wots_baseline/wots.py
// Same params same F same H_msg same seeded keygen same checksum
// Goal is a fair CPU baseline for the wots_pycuda GPU version
// We use a software SHA256 here just like the GPU kernel so the
// only real difference measured is the GPU parallelism

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N        32          // n is 32 bytes
#define W        16          // Winternitz parameter
#define LEN1     64          // ceil 8n over log2 w
#define LEN2     3           // checksum digits
#define LEN      67          // total chains LEN1 plus LEN2

// one 32 byte atom
typedef struct { uint8_t b[N]; } atom;

// ---------- SHA256 ----------
// Plain software SHA256 same math as the GPU kernel
// We keep it simple and process the message in 64 byte blocks

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
    // load the 16 big endian words
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i*4]   << 24) | ((uint32_t)blk[i*4+1] << 16) |
               ((uint32_t)blk[i*4+2] <<  8) | ((uint32_t)blk[i*4+3]);
    }
    // extend the schedule
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

// hash an arbitrary length message and write 32 bytes out
static void sha256(const uint8_t* msg, size_t len, uint8_t out[N]) {
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    size_t full = len / 64;
    for (size_t i = 0; i < full; i++) sha256_block(h, msg + i*64);

    // build the final padded block or two
    uint8_t tail[128];
    size_t rem = len - full*64;
    memcpy(tail, msg + full*64, rem);
    tail[rem] = 0x80;
    size_t pad_len = (rem < 56) ? 64 : 128;
    memset(tail + rem + 1, 0, pad_len - rem - 1);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) tail[pad_len - 1 - i] = (uint8_t)(bits >> (8*i));
    sha256_block(h, tail);
    if (pad_len == 128) sha256_block(h, tail + 64);

    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(h[i] >> 24);
        out[i*4+1] = (uint8_t)(h[i] >> 16);
        out[i*4+2] = (uint8_t)(h[i] >>  8);
        out[i*4+3] = (uint8_t)(h[i]);
    }
}

// ---------- WOTS hash functions ----------

// F(M) = SHA256 of 32 zero bytes then M
// The zero prefix is the domain separator toByte 0 32
static void F(const uint8_t M[N], uint8_t out[N]) {
    uint8_t buf[2*N];
    memset(buf, 0, N);
    memcpy(buf + N, M, N);
    sha256(buf, 2*N, out);
}

// H_msg(M) = SHA256 of toByte 2 32 then M
// toByte 2 32 is the number 2 as 32 big endian bytes so it is 31 zeros then a 2
static void H_msg(const uint8_t* M, size_t len, uint8_t out[N]) {
    uint8_t* buf = (uint8_t*)malloc(N + len);
    memset(buf, 0, N);
    buf[N - 1] = 2;
    memcpy(buf + N, M, len);
    sha256(buf, N + len, out);
    free(buf);
}

// ---------- chain ----------

// apply s steps of F starting from node X
// each step is X = F(X)
static void chain(const uint8_t X[N], int s, uint8_t out[N]) {
    uint8_t tmp[N];
    memcpy(tmp, X, N);
    for (int j = 0; j < s; j++) {
        uint8_t next[N];
        F(tmp, next);
        memcpy(tmp, next, N);
    }
    memcpy(out, tmp, N);
}

// ---------- keygen ----------

// seeded private key atom is SHA256 of seed then i as 4 big endian bytes
static void prg(const uint8_t* seed, size_t slen, uint32_t i, uint8_t out[N]) {
    uint8_t* buf = (uint8_t*)malloc(slen + 4);
    memcpy(buf, seed, slen);
    buf[slen]   = (uint8_t)(i >> 24);
    buf[slen+1] = (uint8_t)(i >> 16);
    buf[slen+2] = (uint8_t)(i >>  8);
    buf[slen+3] = (uint8_t)(i);
    sha256(buf, slen + 4, out);
    free(buf);
}

// build sk and pk from a seed
// pk atom is the sk atom chained all the way to the chain end
static void key_gen(const uint8_t* seed, size_t slen, atom sk[LEN], atom pk[LEN]) {
    for (int i = 0; i < LEN; i++) prg(seed, slen, i, sk[i].b);
    for (int i = 0; i < LEN; i++) chain(sk[i].b, W - 1, pk[i].b);
}

// draw a fresh 32 byte seed from the OS random source
// this is the true random path
// save the seed and feed it back to key_gen to reproduce the same keys
static void random_seed(uint8_t out[N]) {
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f) { fprintf(stderr, "cannot open urandom\n"); exit(1); }
    if (fread(out, 1, N, f) != N) { fprintf(stderr, "urandom read failed\n"); exit(1); }
    fclose(f);
}

// keygen with a truly random seed
// the seed used is written to seed_out so the keys can be reproduced later
static void key_gen_random(uint8_t seed_out[N], atom sk[LEN], atom pk[LEN]) {
    random_seed(seed_out);
    key_gen(seed_out, N, sk, pk);
}

// ---------- base w and checksum ----------

// split bytes into base w digits
// for w 16 each byte gives two nibbles
static void base_w(const uint8_t* data, int out_len, int* out) {
    int idx = 0;
    int bits = 0;
    uint8_t total = 0;
    for (int i = 0; i < out_len; i++) {
        if (bits == 0) {
            total = data[idx++];
            bits = 8;
        }
        bits -= 4;          // log2 w is 4
        out[i] = (total >> bits) & (W - 1);
    }
}

// checksum digits over the LEN1 message digits
static void get_checksum(const int* M, int* csum_out) {
    int csum = 0;
    for (int i = 0; i < LEN1; i++) csum += (W - 1) - M[i];
    // shift so the digits sit at the top
    int csum_bits = LEN2 * 4;
    int shift = (8 - (csum_bits % 8)) % 8;
    csum <<= shift;
    // pack into 2 bytes big endian then split
    uint8_t csum_bytes[2];
    csum_bytes[0] = (uint8_t)(csum >> 8);
    csum_bytes[1] = (uint8_t)(csum);
    base_w(csum_bytes, LEN2, csum_out);
}

// full message digit vector B is the LEN1 message digits then LEN2 checksum
static void get_msg_digits(const uint8_t* msg, size_t len, int B[LEN]) {
    uint8_t digest[N];
    H_msg(msg, len, digest);
    base_w(digest, LEN1, B);
    get_checksum(B, B + LEN1);
}

// ---------- sign and verify ----------

// signature atom is sk atom chained forward by its message digit
static void sign(const atom sk[LEN], const uint8_t* msg, size_t len, atom sig[LEN]) {
    int B[LEN];
    get_msg_digits(msg, len, B);
    for (int i = 0; i < LEN; i++) chain(sk[i].b, B[i], sig[i].b);
}

// rebuild the pk candidate by finishing each chain and compare
static int verify(const atom pk[LEN], const uint8_t* msg, size_t len, const atom sig[LEN]) {
    int B[LEN];
    get_msg_digits(msg, len, B);
    atom cand[LEN];
    for (int i = 0; i < LEN; i++) chain(sig[i].b, W - 1 - B[i], cand[i].b);
    return memcmp(cand, pk, sizeof(atom) * LEN) == 0;
}

// ---------- helpers ----------

// read a whole file into a freshly allocated buffer
static uint8_t* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz);
    size_t got = fread(buf, 1, sz, f);
    fclose(f);
    *out_len = got;
    return buf;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------- tests and benchmark ----------

static void test_roundtrip(const uint8_t* msg, size_t len) {
    atom sk[LEN], pk[LEN], sig[LEN];
    key_gen((const uint8_t*)"test_seed", 9, sk, pk);
    sign(sk, msg, len, sig);
    if (!verify(pk, msg, len, sig)) { printf("roundtrip FAILED\n"); exit(1); }
    // flip one atom and expect failure
    atom bad[LEN];
    memcpy(bad, sig, sizeof(bad));
    memset(bad[0].b, 0, N);
    if (verify(pk, msg, len, bad)) { printf("tamper check FAILED\n"); exit(1); }
    printf("roundtrip test passed\n");
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "../wots_baseline/plaintext/short.txt";
    size_t len;
    uint8_t* msg = read_file(path, &len);

    test_roundtrip(msg, len);

    // show the two seeding modes
    // reproducible mode uses a fixed seed so the same keys come back
    // random mode draws a fresh seed from the OS so the keys differ each run
    {
        atom sk_a[LEN], pk_a[LEN], sk_b[LEN], pk_b[LEN];
        key_gen((const uint8_t*)"fixed_seed", 10, sk_a, pk_a);
        key_gen((const uint8_t*)"fixed_seed", 10, sk_b, pk_b);
        int same = memcmp(pk_a, pk_b, sizeof(atom) * LEN) == 0;
        printf("reproducible keygen same key %s\n", same ? "yes" : "no");

        uint8_t seed_r[N];
        atom sk_r[LEN], pk_r[LEN];
        key_gen_random(seed_r, sk_r, pk_r);
        // feed the saved seed back and check we rebuild the same key
        atom sk_c[LEN], pk_c[LEN];
        key_gen(seed_r, N, sk_c, pk_c);
        int rebuilt = memcmp(pk_r, pk_c, sizeof(atom) * LEN) == 0;
        printf("random keygen reproducible from saved seed %s\n", rebuilt ? "yes" : "no");
    }

    // benchmark keygen sign verify over many iterations
    int iters = (argc > 2) ? atoi(argv[2]) : 1000;
    atom sk[LEN], pk[LEN], sig[LEN];

    double t0 = now_sec();
    for (int it = 0; it < iters; it++) key_gen((const uint8_t*)"bench_seed", 10, sk, pk);
    double t1 = now_sec();
    for (int it = 0; it < iters; it++) sign(sk, msg, len, sig);
    double t2 = now_sec();
    int ok = 1;
    for (int it = 0; it < iters; it++) ok &= verify(pk, msg, len, sig);
    double t3 = now_sec();

    printf("iters %d  verify ok %d\n", iters, ok);
    printf("keygen  %.3f us per op\n", (t1 - t0) / iters * 1e6);
    printf("sign    %.3f us per op\n", (t2 - t1) / iters * 1e6);
    printf("verify  %.3f us per op\n", (t3 - t2) / iters * 1e6);

    // chain only timing
    // this times just the hash chains which is the exact work the cuda kernel does
    // it drops the seed derive and the H_msg so it is a fair compute vs compute number
    int B[LEN];
    get_msg_digits(msg, len, B);
    atom cand[LEN];

    double c0 = now_sec();
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < LEN; i++) chain(sk[i].b, W - 1, pk[i].b);
    double c1 = now_sec();
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < LEN; i++) chain(sk[i].b, B[i], sig[i].b);
    double c2 = now_sec();
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < LEN; i++) chain(sig[i].b, W - 1 - B[i], cand[i].b);
    double c3 = now_sec();

    printf("chain_keygen  %.3f us per op\n", (c1 - c0) / iters * 1e6);
    printf("chain_sign    %.3f us per op\n", (c2 - c1) / iters * 1e6);
    printf("chain_verify  %.3f us per op\n", (c3 - c2) / iters * 1e6);

    free(msg);
    return 0;
}
