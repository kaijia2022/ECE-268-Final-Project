#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "xmss.cuh"

// Sign one message with the keypair then verify it back.
// Returns true if the signature verified, false if not.
static bool sign_and_check(XMSS_Keypair* kp, const char* message) {
    size_t msg_len = strlen(message);

    XMSS_Signature sig;
    if (!xmss_sign(kp, (const uint8_t*)message, msg_len, &sig)) {
        printf("    [INFO] No leaves left, signing refused.\n");
        return false;
    }

    printf("    Signed '%s' with leaf %u. Auth path %u bytes.\n", message, sig.leaf_idx, sig.auth_path_len);

    bool ok = xmss_verify(&sig, (const uint8_t*)message, msg_len, kp->root, kp->tree_height, kp->pub_seed);
    printf("    Verify: %s\n", ok ? "VALID" : "INVALID");

    free(sig.auth_path);
    return ok;
}

int main() {
    printf("=== XMSS FULL LIFECYCLE TEST keygen sign verify and state ===\n\n");

    uint32_t tree_height = 3;               // small tree, 8 leaves
    uint8_t sk_seed[32]  = {0xBB};
    uint8_t sk_prf[32]   = {0xCC};
    uint8_t pub_seed[32] = {0xAA};

    printf("[*] KeyGen height %u, building a full real tree...\n", tree_height);
    XMSS_Keypair kp;
    xmss_keygen(&kp, tree_height, sk_seed, sk_prf, pub_seed);

    printf("    Public Key root: ");
    for (int i = 0; i < 8; i++) printf("%02x", kp.root[i]);
    printf("...\n");
    printf("    Leaves available: %u, next leaf: %u\n\n", kp.num_leaves, kp.next_idx);

    // Sign a few messages, each one should grab the next leaf in order.
    printf("[*] Signing three messages, watch the leaf index advance...\n");
    sign_and_check(&kp, "Launch codes validated.");
    sign_and_check(&kp, "Second message here.");
    sign_and_check(&kp, "Third and final note.");
    printf("    State after three signatures, next leaf: %u\n\n", kp.next_idx);

    // Tamper check, verify must reject a changed message.
    printf("[*] Tamper check on a fresh signature...\n");
    const char* original = "Transfer 100 coins.";
    XMSS_Signature sig;
    xmss_sign(&kp, (const uint8_t*)original, strlen(original), &sig);
    const char* tampered = "Transfer 999 coins.";
    bool bad = xmss_verify(&sig, (const uint8_t*)tampered, strlen(tampered), kp.root, kp.tree_height, kp.pub_seed);
    printf("    Tampered message verify: %s\n\n", bad ? "VALID, BAD" : "rejected, good");
    free(sig.auth_path);

    // State check, keep signing until the tree runs out of leaves.
    printf("[*] Draining the rest of the tree to prove leaves are not reused...\n");
    int signed_count = 4; // four signatures used so far
    while (xmss_sign(&kp, (const uint8_t*)"filler", 6, &sig)) {
        signed_count++;
        free(sig.auth_path);
    }
    printf("    Total signatures made: %d, tree holds %u leaves.\n", signed_count, kp.num_leaves);
    printf("    Signing now refuses because every leaf is spent.\n");

    xmss_keypair_free(&kp);
    return 0;
}
