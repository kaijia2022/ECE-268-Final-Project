// XMSS in plain C the merkle tree layer and the signature scheme
// This is a strict port of the xmss-cuda ltree merkle auth_path and xmss code
// The WOTS+ leaf layer and the crypto primitives live in wots.c
//
// Because every byte layout matches the kernels the keygen root here is
// identical to the one xmss-cuda prints for the same seeds so the two
// implementations cross check each other

#include "xmss.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------- node hash ----------

// H combines two child nodes after masking them
// key and the two masks all come from the pub_seed and the node address
// so every tree node hashes with its own key and masks
// H is SHA256 of toByte 1 32 then key then masked left then masked right
static void h_hash(const uint8_t pub_seed[N], ADRS* adrs,
                   const uint8_t left[N], const uint8_t right[N], uint8_t out[N]) {
    uint8_t buf[128];
    memset(buf, 0, 31);
    buf[31] = 0x01;

    uint8_t key[N], mask_left[N], mask_right[N];
    adrs_set_keymask(adrs, 0); prf(pub_seed, adrs, key);
    adrs_set_keymask(adrs, 1); prf(pub_seed, adrs, mask_left);
    adrs_set_keymask(adrs, 2); prf(pub_seed, adrs, mask_right);

    memcpy(buf + 32, key, 32);
    for (int i = 0; i < 32; i++) buf[64 + i] = left[i]  ^ mask_left[i];
    for (int i = 0; i < 32; i++) buf[96 + i] = right[i] ^ mask_right[i];
    sha256(buf, 128, out);
}

// ---------- L-Tree ----------

// fold the 67 wots public chains into one leaf node
// setting the type here clears the ltree address word so it ends up 0
// this matches the kernel exactly
static void l_tree(const uint8_t* wots_pks, const uint8_t pub_seed[N], ADRS* adrs, uint8_t root_out[N]) {
    uint8_t nodes[WOTS_LEN * N];
    memcpy(nodes, wots_pks, WOTS_LEN * N);

    uint32_t len = WOTS_LEN;
    uint32_t height = 0;
    adrs_set_type(adrs, ADDR_TYPE_LTREE);

    while (len > 1) {
        for (uint32_t i = 0; i < (len >> 1); i++) {
            adrs_set_height(adrs, height);
            adrs_set_index(adrs, i);
            h_hash(pub_seed, adrs, &nodes[(2*i)*N], &nodes[(2*i + 1)*N], &nodes[i*N]);
        }
        // odd node count means the last node has no partner so lift it up
        if (len % 2 == 1) {
            memcpy(&nodes[(len >> 1)*N], &nodes[(len - 1)*N], N);
        }
        len = (len >> 1) + (len % 2);
        height++;
    }
    memcpy(root_out, nodes, N);
}

// ---------- Merkle tree ----------

// climb the leaf nodes into the single merkle root
// this eats its buffer so always hand it a throwaway copy
static void build_hash_tree(uint8_t* tree, uint32_t num_leaves, const uint8_t pub_seed[N],
                            ADRS* adrs, uint8_t root_out[N]) {
    uint32_t len = num_leaves;
    uint32_t height = 0;
    adrs_set_type(adrs, ADDR_TYPE_HASHTREE);

    while (len > 1) {
        for (uint32_t i = 0; i < (len >> 1); i++) {
            adrs_set_height(adrs, height);
            adrs_set_index(adrs, i);
            h_hash(pub_seed, adrs, &tree[(2*i)*N], &tree[(2*i + 1)*N], &tree[i*N]);
        }
        if (len % 2 == 1) {
            memcpy(&tree[(len >> 1)*N], &tree[(len - 1)*N], N);
        }
        len = (len >> 1) + (len % 2);
        height++;
    }
    memcpy(root_out, tree, N);
}

// pull out the sibling at every level on the way up to the root
// the sibling is always index xor 1
// this also eats its buffer so hand it a throwaway copy
static void extract_auth_path(uint8_t* tree, uint32_t num_leaves, uint32_t leaf_index,
                              const uint8_t pub_seed[N], ADRS* adrs, uint8_t* auth_path_out) {
    uint32_t len = num_leaves;
    uint32_t height = 0;
    uint32_t target = leaf_index;
    adrs_set_type(adrs, ADDR_TYPE_HASHTREE);

    while (len > 1) {
        uint32_t sibling = target ^ 1;
        memcpy(&auth_path_out[height*N], &tree[sibling*N], N);

        for (uint32_t i = 0; i < (len >> 1); i++) {
            adrs_set_height(adrs, height);
            adrs_set_index(adrs, i);
            h_hash(pub_seed, adrs, &tree[(2*i)*N], &tree[(2*i + 1)*N], &tree[i*N]);
        }
        if (len % 2 == 1) {
            memcpy(&tree[(len >> 1)*N], &tree[(len - 1)*N], N);
        }
        target >>= 1;
        len = (len >> 1) + (len % 2);
        height++;
    }
}

// rebuild a root from a wots public key and an auth path
// fold the wots pk into a leaf then climb using the siblings
static void compute_root(const uint8_t* wots_pk, const uint8_t* auth_path, uint32_t leaf_idx,
                         uint32_t tree_height, const uint8_t pub_seed[N], uint8_t computed_root[N]) {
    ADRS adrs; adrs_init(&adrs);
    adrs_set_layer(&adrs, 0);
    adrs_set_tree(&adrs, 1);
    adrs_set_type(&adrs, ADDR_TYPE_LTREE);
    adrs_set_ltree(&adrs, leaf_idx);

    uint8_t node[N];
    l_tree(wots_pk, pub_seed, &adrs, node);

    uint32_t cur = leaf_idx;
    adrs_set_type(&adrs, ADDR_TYPE_HASHTREE);
    for (uint32_t h = 0; h < tree_height; h++) {
        adrs_set_height(&adrs, h);
        adrs_set_index(&adrs, cur >> 1);
        const uint8_t* sibling = &auth_path[h*N];
        if (cur % 2 == 0) h_hash(pub_seed, &adrs, node, sibling, node);
        else              h_hash(pub_seed, &adrs, sibling, node, node);
        cur >>= 1;
    }
    memcpy(computed_root, node, N);
}

// ---------- randomized message digest ----------

// message randomness r for one leaf index
// r is SHA256 of toByte 3 32 then sk_prf then toByte idx 32
// deterministic per index so the same idx always gives the same r
static void prf_msg(const uint8_t sk_prf[N], uint32_t idx, uint8_t r_out[N]) {
    uint8_t buf[96];
    memset(buf, 0, 31);
    buf[31] = 0x03;
    memcpy(buf + 32, sk_prf, 32);
    memset(buf + 64, 0, 28);
    buf[92] = (uint8_t)(idx >> 24);
    buf[93] = (uint8_t)(idx >> 16);
    buf[94] = (uint8_t)(idx >> 8);
    buf[95] = (uint8_t)(idx);
    sha256(buf, 96, r_out);
}

// the randomized digest both sign and verify feed into wots
// digest is SHA256 of toByte 2 32 then r then root then toByte idx 32 then msg
// binding root and idx in stops a signature from moving to another leaf or key
static void h_msg(const uint8_t r[N], const uint8_t root[N], uint32_t idx,
                  const uint8_t* msg, size_t msg_len, uint8_t digest_out[N]) {
    uint8_t* buf = (uint8_t*)malloc(128 + msg_len);
    memset(buf, 0, 31);
    buf[31] = 0x02;
    memcpy(buf + 32, r, 32);
    memcpy(buf + 64, root, 32);
    memset(buf + 96, 0, 28);
    buf[124] = (uint8_t)(idx >> 24);
    buf[125] = (uint8_t)(idx >> 16);
    buf[126] = (uint8_t)(idx >> 8);
    buf[127] = (uint8_t)(idx);
    memcpy(buf + 128, msg, msg_len);
    sha256(buf, 128 + msg_len, digest_out);
    free(buf);
}

// ---------- keygen sign verify ----------

// compute one real leaf
// derive the 67 wots secret seeds run each chain to its end then fold with l_tree
static void compute_leaf(uint32_t leaf_idx, const uint8_t sk_seed[N], const uint8_t pub_seed[N],
                         uint8_t leaf_out[N]) {
    uint8_t wots_pk[WOTS_LEN * N];
    for (int c = 0; c < WOTS_LEN; c++) {
        ADRS adrs; adrs_init(&adrs);
        adrs_set_layer(&adrs, 0);
        adrs_set_tree(&adrs, 1);
        adrs_set_type(&adrs, ADDR_TYPE_OTS);
        adrs_set_ots(&adrs, leaf_idx);
        adrs_set_chain(&adrs, c);

        uint8_t secret[N];
        prf(sk_seed, &adrs, secret);
        wots_chain(secret, 0, CHAIN_MAX, pub_seed, &adrs, &wots_pk[c*N]);
    }
    ADRS adrs; adrs_init(&adrs);
    adrs_set_layer(&adrs, 0);
    adrs_set_tree(&adrs, 1);
    adrs_set_type(&adrs, ADDR_TYPE_LTREE);
    adrs_set_ltree(&adrs, leaf_idx);
    l_tree(wots_pk, pub_seed, &adrs, leaf_out);
}

void xmss_keygen(XMSS_Keypair* kp, uint32_t tree_height,
                 const uint8_t sk_seed[N], const uint8_t sk_prf[N], const uint8_t pub_seed[N]) {
    uint32_t num_leaves = 1u << tree_height;
    kp->tree_height = tree_height;
    kp->num_leaves = num_leaves;
    kp->next_idx = 0;                       // fresh key starts at leaf 0
    memcpy(kp->sk_seed, sk_seed, N);
    memcpy(kp->sk_prf, sk_prf, N);
    memcpy(kp->pub_seed, pub_seed, N);
    kp->tree_buffer = (uint8_t*)malloc(num_leaves * N);

    for (uint32_t i = 0; i < num_leaves; i++)
        compute_leaf(i, sk_seed, pub_seed, &kp->tree_buffer[i*N]);

    // build the root from a copy because the build eats its buffer
    uint8_t* copy = (uint8_t*)malloc(num_leaves * N);
    memcpy(copy, kp->tree_buffer, num_leaves * N);
    ADRS adrs; adrs_init(&adrs);
    adrs_set_layer(&adrs, 0);
    adrs_set_tree(&adrs, 1);
    build_hash_tree(copy, num_leaves, pub_seed, &adrs, kp->root);
    free(copy);
}

void xmss_keypair_free(XMSS_Keypair* kp) {
    free(kp->tree_buffer);
    kp->tree_buffer = NULL;
}

bool xmss_sign(XMSS_Keypair* kp, const uint8_t* msg, size_t msg_len, XMSS_Signature* out_sig) {
    // state check refuse to sign once every leaf has been used
    if (kp->next_idx >= kp->num_leaves) return false;

    uint32_t leaf_idx = kp->next_idx;       // take the next free leaf
    out_sig->leaf_idx = leaf_idx;
    out_sig->auth_path_len = kp->tree_height * N;
    out_sig->auth_path = (uint8_t*)malloc(out_sig->auth_path_len);

    // real message hash randomized and bound to this leaf and this root
    uint8_t r[N], digest[N];
    prf_msg(kp->sk_prf, leaf_idx, r);
    h_msg(r, kp->root, leaf_idx, msg, msg_len, digest);
    memcpy(out_sig->msg_randomness, r, N);

    // derive this leaf private chains and sign the digest with them
    uint8_t lengths[WOTS_LEN];
    get_chain_lengths(digest, lengths);
    for (int c = 0; c < WOTS_LEN; c++) {
        ADRS adrs; adrs_init(&adrs);
        adrs_set_layer(&adrs, 0);
        adrs_set_tree(&adrs, 1);
        adrs_set_type(&adrs, ADDR_TYPE_OTS);
        adrs_set_ots(&adrs, leaf_idx);
        adrs_set_chain(&adrs, c);

        uint8_t secret[N];
        prf(kp->sk_seed, &adrs, secret);
        wots_chain(secret, 0, lengths[c], kp->pub_seed, &adrs, &out_sig->wots_sig[c*N]);
    }

    // pull the auth path out of a throwaway copy of the tree
    uint8_t* copy = (uint8_t*)malloc(kp->num_leaves * N);
    memcpy(copy, kp->tree_buffer, kp->num_leaves * N);
    ADRS adrs; adrs_init(&adrs);
    adrs_set_layer(&adrs, 0);
    adrs_set_tree(&adrs, 1);
    extract_auth_path(copy, kp->num_leaves, leaf_idx, kp->pub_seed, &adrs, out_sig->auth_path);
    free(copy);

    // state update this leaf is spent so move on to the next one
    kp->next_idx++;
    return true;
}

bool xmss_verify(const XMSS_Signature* sig, const uint8_t* msg, size_t msg_len,
                 const uint8_t expected_pub_key[N], uint32_t tree_height, const uint8_t pub_seed[N]) {
    // the verifier hashes the message the same way the signer did using the
    // trusted public key as the root and the r carried in the signature
    uint8_t digest[N];
    h_msg(sig->msg_randomness, expected_pub_key, sig->leaf_idx, msg, msg_len, digest);

    uint8_t lengths[WOTS_LEN];
    get_chain_lengths(digest, lengths);

    // rebuild the wots public key by finishing each chain
    uint8_t wots_pk[WOTS_LEN * N];
    for (int c = 0; c < WOTS_LEN; c++) {
        uint32_t start = lengths[c];
        uint32_t steps = CHAIN_MAX - start;
        ADRS adrs; adrs_init(&adrs);
        adrs_set_layer(&adrs, 0);
        adrs_set_tree(&adrs, 1);
        adrs_set_type(&adrs, ADDR_TYPE_OTS);
        adrs_set_ots(&adrs, sig->leaf_idx);
        adrs_set_chain(&adrs, c);
        wots_chain(&sig->wots_sig[c*N], start, steps, pub_seed, &adrs, &wots_pk[c*N]);
    }

    // fold and climb to a root then check it against the public key
    uint8_t computed_root[N];
    compute_root(wots_pk, sig->auth_path, sig->leaf_idx, tree_height, pub_seed, computed_root);
    return memcmp(computed_root, expected_pub_key, N) == 0;
}

// ---------- demo and benchmark ----------
// everything below is the standalone runner not the library
// tests compile this file with XMSS_NO_MAIN so they can bring their own main
#ifndef XMSS_NO_MAIN

// ---------- helpers ----------

// draw bytes from the OS random source
// this is the true random path save the seeds to reproduce a key later
static void random_bytes(uint8_t* out, size_t len) {
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f) { fprintf(stderr, "cannot open urandom\n"); exit(1); }
    if (fread(out, 1, len, f) != len) { fprintf(stderr, "urandom read failed\n"); exit(1); }
    fclose(f);
}

// keygen with three truly random seeds
// the seeds are written back so the same key can be rebuilt later
static void xmss_keygen_random(XMSS_Keypair* kp, uint32_t tree_height,
                               uint8_t sk_seed[N], uint8_t sk_prf[N], uint8_t pub_seed[N]) {
    random_bytes(sk_seed, N);
    random_bytes(sk_prf, N);
    random_bytes(pub_seed, N);
    xmss_keygen(kp, tree_height, sk_seed, sk_prf, pub_seed);
}

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

static void print_hex(const uint8_t* b, int n) {
    for (int i = 0; i < n; i++) printf("%02x", b[i]);
}

// ---------- lifecycle test ----------

// sign one message then verify it back and report
static bool sign_and_check(XMSS_Keypair* kp, const char* message) {
    size_t msg_len = strlen(message);
    XMSS_Signature sig;
    if (!xmss_sign(kp, (const uint8_t*)message, msg_len, &sig)) {
        printf("    no leaves left signing refused\n");
        return false;
    }
    printf("    signed '%s' with leaf %u auth path %u bytes\n", message, sig.leaf_idx, sig.auth_path_len);
    bool ok = xmss_verify(&sig, (const uint8_t*)message, msg_len, kp->root, kp->tree_height, kp->pub_seed);
    printf("    verify %s\n", ok ? "VALID" : "INVALID");
    free(sig.auth_path);
    return ok;
}

// the root xmss-cuda prints for sk_seed 0xBB sk_prf 0xCC pub_seed 0xAA height 3
// our C version must produce the exact same bytes
static const uint8_t CUDA_ROOT8[8] = {0x90,0x1f,0xf2,0xbf,0x90,0xc4,0x0b,0xa6};

static void lifecycle_test(void) {
    printf("=== XMSS lifecycle test keygen sign verify and state ===\n\n");

    uint32_t tree_height = 3;               // small tree 8 leaves
    uint8_t sk_seed[N]  = {0xBB};
    uint8_t sk_prf[N]   = {0xCC};
    uint8_t pub_seed[N] = {0xAA};

    printf("[*] keygen height %u building a full real tree\n", tree_height);
    XMSS_Keypair kp;
    xmss_keygen(&kp, tree_height, sk_seed, sk_prf, pub_seed);

    printf("    public key root ");
    print_hex(kp.root, 8);
    printf("...\n");
    bool match = memcmp(kp.root, CUDA_ROOT8, 8) == 0;
    printf("    matches xmss-cuda root %s\n", match ? "yes" : "no");
    printf("    leaves available %u next leaf %u\n\n", kp.num_leaves, kp.next_idx);

    printf("[*] signing three messages watch the leaf index advance\n");
    sign_and_check(&kp, "Launch codes validated.");
    sign_and_check(&kp, "Second message here.");
    sign_and_check(&kp, "Third and final note.");
    printf("    state after three signatures next leaf %u\n\n", kp.next_idx);

    printf("[*] tamper check on a fresh signature\n");
    const char* original = "Transfer 100 coins.";
    XMSS_Signature sig;
    xmss_sign(&kp, (const uint8_t*)original, strlen(original), &sig);
    const char* tampered = "Transfer 999 coins.";
    bool bad = xmss_verify(&sig, (const uint8_t*)tampered, strlen(tampered), kp.root, kp.tree_height, kp.pub_seed);
    printf("    tampered message verify %s\n\n", bad ? "VALID bad" : "rejected good");
    free(sig.auth_path);

    printf("[*] draining the rest of the tree to prove leaves are not reused\n");
    int signed_count = 4;                   // four signatures used so far
    while (xmss_sign(&kp, (const uint8_t*)"filler", 6, &sig)) {
        signed_count++;
        free(sig.auth_path);
    }
    printf("    total signatures made %d tree holds %u leaves\n", signed_count, kp.num_leaves);
    printf("    signing now refuses because every leaf is spent\n\n");

    xmss_keypair_free(&kp);

    // show the random seed path rebuilds the same key from saved seeds
    uint8_t rs[N], rp[N], rpub[N];
    XMSS_Keypair kr;
    xmss_keygen_random(&kr, tree_height, rs, rp, rpub);
    XMSS_Keypair kc;
    xmss_keygen(&kc, tree_height, rs, rp, rpub);
    int rebuilt = memcmp(kr.root, kc.root, N) == 0;
    printf("[*] random keygen reproducible from saved seeds %s\n\n", rebuilt ? "yes" : "no");
    xmss_keypair_free(&kr);
    xmss_keypair_free(&kc);
}

// ---------- benchmark ----------

// keygen builds the whole tree which is the work the gpu parallelizes
// sign and verify both run 67 short chains plus the tree climb
// sign resets next_idx each round so it always signs leaf 0 for a clean number
static void benchmark(uint32_t tree_height, int iters, const uint8_t* msg, size_t msg_len) {
    uint8_t sk_seed[N]  = {0x11};
    uint8_t sk_prf[N]   = {0x22};
    uint8_t pub_seed[N] = {0x33};

    printf("=== benchmark height %u iters %d ===\n", tree_height, iters);

    double t0 = now_sec();
    XMSS_Keypair kp;
    for (int it = 0; it < iters; it++) {
        xmss_keygen(&kp, tree_height, sk_seed, sk_prf, pub_seed);
        if (it < iters - 1) xmss_keypair_free(&kp);
    }
    double t1 = now_sec();

    XMSS_Signature sig;
    for (int it = 0; it < iters; it++) {
        kp.next_idx = 0;                    // reuse leaf 0 so the timing is clean
        xmss_sign(&kp, msg, msg_len, &sig);
        if (it < iters - 1) free(sig.auth_path);
    }
    double t2 = now_sec();

    int ok = 1;
    for (int it = 0; it < iters; it++)
        ok &= xmss_verify(&sig, msg, msg_len, kp.root, kp.tree_height, kp.pub_seed);
    double t3 = now_sec();

    free(sig.auth_path);
    xmss_keypair_free(&kp);

    printf("verify ok %d\n", ok);
    printf("keygen  %.3f us per op\n", (t1 - t0) / iters * 1e6);
    printf("sign    %.3f us per op\n", (t2 - t1) / iters * 1e6);
    printf("verify  %.3f us per op\n", (t3 - t2) / iters * 1e6);

    // per op sha256 counts for this exact message
    // reset the global counter run one clean op then read it back
    XMSS_Keypair kh;
    g_sha256_count = 0;
    xmss_keygen(&kh, tree_height, sk_seed, sk_prf, pub_seed);
    unsigned long long hc_keygen = (unsigned long long)g_sha256_count;

    XMSS_Signature sh;
    kh.next_idx = 0;
    g_sha256_count = 0;
    xmss_sign(&kh, msg, msg_len, &sh);
    unsigned long long hc_sign = (unsigned long long)g_sha256_count;

    g_sha256_count = 0;
    xmss_verify(&sh, msg, msg_len, kh.root, kh.tree_height, kh.pub_seed);
    unsigned long long hc_verify = (unsigned long long)g_sha256_count;

    free(sh.auth_path);
    xmss_keypair_free(&kh);

    printf("keygen_hashes %llu\n", hc_keygen);
    printf("sign_hashes %llu\n", hc_sign);
    printf("verify_hashes %llu\n", hc_verify);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "../wots_pycuda/plaintext/short.txt";
    uint32_t height  = (argc > 2) ? (uint32_t)atoi(argv[2]) : 3;
    int iters        = (argc > 3) ? atoi(argv[3]) : 100;

    size_t len;
    uint8_t* msg = read_file(path, &len);

    lifecycle_test();
    benchmark(height, iters, msg, len);

    free(msg);
    return 0;
}

#endif // XMSS_NO_MAIN
