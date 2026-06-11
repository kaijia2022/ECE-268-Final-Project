#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "xmss.cuh"

// Cross check vector dumper for the GPU side
// It keygens a fixed tree then signs one fixed message at leaf 0
// and prints the root and the whole signature as hex
// The C baseline in xmss_c prints the exact same lines
// so a plain diff proves the two implementations agree byte for byte

static void print_field(const char* name, const uint8_t* buf, int n) {
    printf("%s=", name);
    for (int i = 0; i < n; i++) printf("%02x", buf[i]);
    printf("\n");
}

int main() {
    uint32_t tree_height = 3;
    uint8_t sk_seed[32]  = {0xBB};
    uint8_t sk_prf[32]   = {0xCC};
    uint8_t pub_seed[32] = {0xAA};

    XMSS_Keypair kp;
    xmss_keygen(&kp, tree_height, sk_seed, sk_prf, pub_seed);
    print_field("root", kp.root, 32);

    // fixed message so both sides hash the exact same bytes at leaf 0
    const char* message = "xmss cross check vector";
    XMSS_Signature sig;
    xmss_sign(&kp, (const uint8_t*)message, strlen(message), &sig);

    printf("leaf_idx=%u\n", sig.leaf_idx);
    print_field("r", sig.msg_randomness, 32);
    print_field("wots_sig", sig.wots_sig, 67 * 32);
    print_field("auth_path", sig.auth_path, sig.auth_path_len);

    free(sig.auth_path);
    xmss_keypair_free(&kp);
    return 0;
}
