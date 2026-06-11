// Cross check vector dumper for the C side
// Same fixed seeds and message as xmss-cuda tests test_cross_check.cu
// It prints the root and the whole signature as hex in the same format
// so a plain diff against cuda_reference_vector.txt proves the C baseline
// and the GPU implementation agree byte for byte

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../xmss.h"

static void print_field(const char* name, const uint8_t* buf, int n) {
    printf("%s=", name);
    for (int i = 0; i < n; i++) printf("%02x", buf[i]);
    printf("\n");
}

int main(void) {
    uint32_t tree_height = 3;
    uint8_t sk_seed[N]  = {0xBB};
    uint8_t sk_prf[N]   = {0xCC};
    uint8_t pub_seed[N] = {0xAA};

    XMSS_Keypair kp;
    xmss_keygen(&kp, tree_height, sk_seed, sk_prf, pub_seed);
    print_field("root", kp.root, N);

    // fixed message so both sides hash the exact same bytes at leaf 0
    const char* message = "xmss cross check vector";
    XMSS_Signature sig;
    xmss_sign(&kp, (const uint8_t*)message, strlen(message), &sig);

    printf("leaf_idx=%u\n", sig.leaf_idx);
    print_field("r", sig.msg_randomness, N);
    print_field("wots_sig", sig.wots_sig, WOTS_LEN * N);
    print_field("auth_path", sig.auth_path, sig.auth_path_len);

    free(sig.auth_path);
    xmss_keypair_free(&kp);
    return 0;
}
