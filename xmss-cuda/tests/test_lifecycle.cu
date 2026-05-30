#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "xmss.cuh"
#include "wots.cuh"
#include "ltree.cuh"
#include "merkle.cuh"
#include "hash_wrappers.cuh"

// Generates the exact cryptographic leaf for the index we want to sign with
__global__ void compute_leaf_kernel(const uint8_t* secret_seeds, uint8_t* tree_buffer, uint32_t target_leaf_idx, const uint8_t* pub_seed) {
    int tid = threadIdx.x; 
    if (tid >= 67) return;

    // Allocate shared memory so all threads can pool their results
    __shared__ uint8_t shared_wots_pk[67 * 32];

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    adrs.setType(XMSS_ADDR_TYPE_OTS);
    adrs.setOTSAddress(target_leaf_idx);
    
    adrs.setChainAddress(tid);

    wots_chain(&secret_seeds[tid * 32], 0, 15, pub_seed, &adrs, &shared_wots_pk[tid * 32]);

    // Barrier here
    __syncthreads();

    if (tid == 0) {
        adrs.setType(XMSS_ADDR_TYPE_LTREE);
        adrs.setLTreeAddress(target_leaf_idx);
        l_tree(shared_wots_pk, pub_seed, &adrs, &tree_buffer[target_leaf_idx * 32]);
    }
}


// Safe wrapper to build the Merkle hash tree
__global__ void build_tree_safe_kernel(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, uint8_t* root_out) {
    if (threadIdx.x != 0) return;
    
    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    
    build_hash_tree(tree_buffer, num_leaves, pub_seed, &adrs, root_out);
}

// Private Key Generation: Derives the 67 WOTS+ secret seeds dynamically using the PRF and SK_SEED
__global__ void generate_secret_seeds_kernel(const uint8_t* sk_seed, uint32_t target_leaf_idx, uint8_t* secret_seeds_out) {
    int chain_index = threadIdx.x;
    if (chain_index >= 67) return;

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    adrs.setType(XMSS_ADDR_TYPE_OTS);
    adrs.setOTSAddress(target_leaf_idx);
    adrs.setChainAddress(chain_index);

    prf(sk_seed, &adrs, (SHA256_CTX*)&secret_seeds_out[chain_index * 32]);
}

void xmss_keygen(uint32_t num_leaves, uint32_t target_leaf_idx, uint8_t* out_secret_seeds, uint8_t* out_tree_buffer, uint8_t* out_pub_root) {
    uint8_t h_pub_seed[32] = {0xAA}; // Public Seed
    uint8_t h_sk_seed[32]  = {0xBB}; // Master Secret Key Seed (SK_SEED)
    
    // We  mock the OTHER leaves just so the Merkle tree has a full dataset to hash,
    // but the target leaf we are actually using will be cryptographically real.
    for (int i = 0; i < num_leaves * 32; i++) out_tree_buffer[i] = (uint8_t)(i / 32);

    uint8_t *d_sk_seed, *d_secret_seeds, *d_tree_buffer, *d_pub_seed, *d_root;
    cudaMalloc(&d_sk_seed, 32);
    cudaMalloc(&d_secret_seeds, 67 * 32);
    cudaMalloc(&d_tree_buffer, num_leaves * 32);
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_root, 32);

    cudaMemcpy(d_sk_seed, h_sk_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_buffer, out_tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);

    // Derive the 67 private keys using the PRF 
    generate_secret_seeds_kernel<<<1, 67>>>(d_sk_seed, target_leaf_idx, d_secret_seeds);

    // Compute the cryptographic leaf from those private keys
    compute_leaf_kernel<<<1, 67>>>(d_secret_seeds, d_tree_buffer, target_leaf_idx, d_pub_seed);
    
    uint8_t* d_temp_tree_buffer;
    cudaMalloc(&d_temp_tree_buffer, num_leaves * 32);
    cudaMemcpy(d_temp_tree_buffer, d_tree_buffer, num_leaves * 32, cudaMemcpyDeviceToDevice);

    //Build the tree safely using a temp buffer
    build_tree_safe_kernel<<<1, 1>>>(d_temp_tree_buffer, num_leaves, d_pub_seed, d_root);
    
    cudaDeviceSynchronize();


    cudaMemcpy(out_pub_root, d_root, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_tree_buffer, d_tree_buffer, num_leaves * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_secret_seeds, d_secret_seeds, 67 * 32, cudaMemcpyDeviceToHost); // Bring derived keys back to host for signing
    
    cudaFree(d_sk_seed); cudaFree(d_secret_seeds); cudaFree(d_tree_buffer); 
    cudaFree(d_pub_seed); cudaFree(d_root); cudaFree(d_temp_tree_buffer);
}

int main() {
    printf("=== XMSS FULL LIFECYCLE TEST (signing and verifying a single leaf index) ===\n\n");

    uint32_t tree_height = 3;               
    uint32_t num_leaves = 1 << tree_height; // num_leaves = 6
    uint8_t pub_seed[32] = {0xAA};
    uint32_t leaf_idx_to_use = 2; // Sign with the 3rd key

    printf("[*] Generating Keys (Height %u)...\n", tree_height);
    uint8_t* secret_seeds = (uint8_t*)malloc(67 * 32);
    uint8_t* tree_buffer = (uint8_t*)malloc(num_leaves * 32);
    uint8_t public_key_root[32];
    
    xmss_keygen(num_leaves, leaf_idx_to_use, secret_seeds, tree_buffer, public_key_root);
    
    printf("    Public Key Generated: ");
    for(int i=0; i<8; i++) printf("%02x", public_key_root[i]);
    printf("...\n\n");

    const char* message = "Launch codes validated.";
    size_t msg_len = strlen(message);
    
    printf("[*] Signing Message: '%s' (Using Leaf %u)\n", message, leaf_idx_to_use);
    XMSS_Signature sig;
    xmss_sign((const uint8_t*)message, msg_len, leaf_idx_to_use, tree_height, secret_seeds, pub_seed, tree_buffer, &sig);
    printf("    Signature generated successfully. Auth path size: %u bytes.\n\n", sig.auth_path_len);

    printf("[*] Verifying Signature against Public Key...\n");
    bool is_valid = xmss_verify(&sig, (const uint8_t*)message, msg_len, public_key_root, tree_height, pub_seed);

    if (is_valid) {
        printf("    [SUCCESS] Signature is VALID!\n");
    } else {
        printf("    [FAILED] Signature is INVALID!\n");
    }

    printf("\n[*] Tampering with message and verifying again...\n");
    const char* tampered_message = "Launch codes destroyed.";
    bool is_tampered_valid = xmss_verify(&sig, (const uint8_t*)tampered_message, strlen(tampered_message), public_key_root, tree_height, pub_seed);
    
    if (!is_tampered_valid) {
        printf("    [SUCCESS] System correctly rejected the tampered signature!\n");
    }

    free(secret_seeds);
    free(tree_buffer);
    free(sig.auth_path);

    return 0;
}