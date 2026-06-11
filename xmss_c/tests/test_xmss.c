// Unit tests for the C XMSS baseline
// These check the four things project_requirements asks a stateful scheme for
// KeyGen Sign Verify and state management
// and they quantify signature size key size and per signature hash count
//
// Build and run with make test from the xmss_c folder

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../xmss.h"

// ---------- tiny test harness ----------

static int g_pass = 0;
static int g_fail = 0;

static void check(int cond, const char* name) {
    if (cond) { g_pass++; printf("  PASS  %s\n", name); }
    else      { g_fail++; printf("  FAIL  %s\n", name); }
}

// compare a byte buffer against a hex string
static int hex_eq(const uint8_t* buf, int n, const char* hex) {
    for (int i = 0; i < n; i++) {
        char b[3] = { hex[2*i], hex[2*i+1], 0 };
        unsigned v = (unsigned)strtoul(b, NULL, 16);
        if (buf[i] != (uint8_t)v) return 0;
    }
    return 1;
}

// the full 32 byte root xmss-cuda prints for these exact seeds at height 3
// captured from xmss-cuda tests test_cross_check.cu
// if the C keygen ever drifts from the GPU this constant catches it
static const char* CUDA_ROOT_HEX =
    "901ff2bf90c40ba67de9d63ff59bf37edcf6d872fbd5e4c2d47c5bb106101c4d";

// shared test seeds
static const uint8_t SK_SEED[N]  = {0xBB};
static const uint8_t SK_PRF[N]   = {0xCC};
static const uint8_t PUB_SEED[N] = {0xAA};

// ---------- KeyGen ----------

// keygen must be deterministic from the seeds and must match the GPU root
static void test_keygen(void) {
    printf("[KeyGen]\n");

    XMSS_Keypair a;
    xmss_keygen(&a, 3, SK_SEED, SK_PRF, PUB_SEED);
    check(a.num_leaves == 8, "height 3 builds 8 leaves");
    check(a.next_idx == 0, "fresh key starts at leaf 0");
    check(hex_eq(a.root, N, CUDA_ROOT_HEX), "root matches xmss-cuda byte for byte");

    // same seeds give the same root
    XMSS_Keypair b;
    xmss_keygen(&b, 3, SK_SEED, SK_PRF, PUB_SEED);
    check(memcmp(a.root, b.root, N) == 0, "same seeds give the same root");

    // one different seed byte gives a different root
    uint8_t other[N] = {0xBB}; other[1] = 0x01;
    XMSS_Keypair c;
    xmss_keygen(&c, 3, other, SK_PRF, PUB_SEED);
    check(memcmp(a.root, c.root, N) != 0, "different seed gives a different root");

    xmss_keypair_free(&a);
    xmss_keypair_free(&b);
    xmss_keypair_free(&c);
}

// ---------- Sign and Verify ----------

// a fresh signature must verify and any change must be rejected
static void test_sign_verify(void) {
    printf("[Sign and Verify]\n");

    XMSS_Keypair kp;
    xmss_keygen(&kp, 3, SK_SEED, SK_PRF, PUB_SEED);

    const char* msg = "the quick brown fox";
    size_t mlen = strlen(msg);
    XMSS_Signature sig;
    bool signed_ok = xmss_sign(&kp, (const uint8_t*)msg, mlen, &sig);
    check(signed_ok, "sign returns true on a fresh key");

    bool good = xmss_verify(&sig, (const uint8_t*)msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    check(good, "valid signature verifies");

    // tamper the message
    const char* bad_msg = "the quick brown box";
    bool t1 = xmss_verify(&sig, (const uint8_t*)bad_msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    check(!t1, "tampered message is rejected");

    // verify under the wrong public key
    uint8_t wrong_root[N];
    memcpy(wrong_root, kp.root, N);
    wrong_root[0] ^= 0xFF;
    bool t2 = xmss_verify(&sig, (const uint8_t*)msg, mlen, wrong_root, kp.tree_height, kp.pub_seed);
    check(!t2, "wrong public key is rejected");

    // tamper one byte of the wots signature
    XMSS_Signature s3 = sig;
    uint8_t wsig[WOTS_LEN * N];
    memcpy(wsig, sig.wots_sig, WOTS_LEN * N);
    wsig[0] ^= 0x01;
    memcpy(s3.wots_sig, wsig, WOTS_LEN * N);
    bool t3 = xmss_verify(&s3, (const uint8_t*)msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    check(!t3, "tampered wots signature is rejected");

    // tamper one byte of the auth path
    uint8_t* apath = (uint8_t*)malloc(sig.auth_path_len);
    memcpy(apath, sig.auth_path, sig.auth_path_len);
    apath[0] ^= 0x01;
    XMSS_Signature s4 = sig;
    s4.auth_path = apath;
    bool t4 = xmss_verify(&s4, (const uint8_t*)msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    check(!t4, "tampered auth path is rejected");
    free(apath);

    // claim a different leaf index than the one we signed with
    XMSS_Signature s5 = sig;
    s5.leaf_idx = 5;
    bool t5 = xmss_verify(&s5, (const uint8_t*)msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    check(!t5, "wrong leaf index is rejected");

    free(sig.auth_path);
    xmss_keypair_free(&kp);
}

// ---------- State management ----------

// every signature must use a fresh leaf and the tree must refuse once it is full
static void test_state(void) {
    printf("[State management]\n");

    XMSS_Keypair kp;
    xmss_keygen(&kp, 3, SK_SEED, SK_PRF, PUB_SEED);

    int seen[8] = {0};
    int distinct = 1;
    int all_verify = 1;
    int idx_in_order = 1;

    // sign with every leaf in the tree
    for (uint32_t i = 0; i < kp.num_leaves; i++) {
        char msg[32];
        snprintf(msg, sizeof(msg), "message number %u", i);
        XMSS_Signature sig;
        bool ok = xmss_sign(&kp, (const uint8_t*)msg, strlen(msg), &sig);
        if (!ok) { idx_in_order = 0; break; }

        if (sig.leaf_idx != i) idx_in_order = 0;     // leaves come out in order
        if (sig.leaf_idx < 8) {
            if (seen[sig.leaf_idx]) distinct = 0;     // no leaf is reused
            seen[sig.leaf_idx] = 1;
        }
        // each leaf signs a different message but all verify under the one root
        if (!xmss_verify(&sig, (const uint8_t*)msg, strlen(msg), kp.root, kp.tree_height, kp.pub_seed))
            all_verify = 0;
        free(sig.auth_path);
    }

    check(idx_in_order, "leaf index advances 0 1 2 up the tree");
    check(distinct, "no leaf is ever reused");
    check(all_verify, "all 8 leaves verify under the same root");
    check(kp.next_idx == kp.num_leaves, "state reaches the end after a full tree");

    // the tree is full now so the next sign must be refused
    XMSS_Signature extra;
    bool refused = !xmss_sign(&kp, (const uint8_t*)"one too many", 12, &extra);
    check(refused, "signing is refused once every leaf is spent");

    xmss_keypair_free(&kp);
}

// ---------- size and hash count quantification ----------

// measure the hashes one op costs by reading the global sha256 counter
static uint64_t count_keygen(uint32_t height) {
    XMSS_Keypair kp;
    g_sha256_count = 0;
    xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED);
    uint64_t c = g_sha256_count;
    xmss_keypair_free(&kp);
    return c;
}

// print the sizes and hash counts the report needs and assert the size formulas
static void test_sizes_and_hashcount(void) {
    printf("[Sizes and hash count]\n");

    uint32_t height = 3;
    XMSS_Keypair kp;
    xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED);

    const char* msg = "size probe message";
    size_t mlen = strlen(msg);

    XMSS_Signature sig;
    g_sha256_count = 0;
    xmss_sign(&kp, (const uint8_t*)msg, mlen, &sig);
    uint64_t sign_hashes = g_sha256_count;

    g_sha256_count = 0;
    bool ok = xmss_verify(&sig, (const uint8_t*)msg, mlen, kp.root, kp.tree_height, kp.pub_seed);
    uint64_t verify_hashes = g_sha256_count;
    uint64_t keygen_hashes = count_keygen(height);

    // signature pieces leaf_idx 4 bytes plus r 32 plus wots_sig plus auth path
    uint32_t wots_sig_bytes = WOTS_LEN * N;
    uint32_t auth_bytes = sig.auth_path_len;
    uint32_t sig_total = 4 + N + wots_sig_bytes + auth_bytes;
    // public key is the root, private key is the three seeds plus the state index
    uint32_t pubkey_bytes = N;
    uint32_t privkey_bytes = 3 * N + 4;

    check(ok, "size probe signature still verifies");
    check(wots_sig_bytes == 2144, "wots signature is 2144 bytes");
    check(auth_bytes == height * N, "auth path is height times 32 bytes");
    check(sig_total == 4 + 32 + 2144 + 96, "full signature is 2276 bytes at height 3");
    check(keygen_hashes > 0 && sign_hashes > 0 && verify_hashes > 0, "every op does real hashing");

    printf("    signature total      %u bytes\n", sig_total);
    printf("      leaf index         4 bytes\n");
    printf("      randomness r       %u bytes\n", N);
    printf("      wots signature     %u bytes\n", wots_sig_bytes);
    printf("      auth path          %u bytes\n", auth_bytes);
    printf("    public key root      %u bytes\n", pubkey_bytes);
    printf("    private key seeds    %u bytes\n", privkey_bytes);
    printf("    hashes per keygen    %llu sha256 calls\n", (unsigned long long)keygen_hashes);
    printf("    hashes per sign      %llu sha256 calls\n", (unsigned long long)sign_hashes);
    printf("    hashes per verify    %llu sha256 calls\n", (unsigned long long)verify_hashes);
    printf("    ECDSA P-256 compare lives in benchmark results and the report\n");

    free(sig.auth_path);
    xmss_keypair_free(&kp);
}

int main(void) {
    printf("=== xmss_c unit tests ===\n\n");

    test_keygen();
    test_sign_verify();
    test_state();
    test_sizes_and_hashcount();

    printf("\n=== %d passed %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
