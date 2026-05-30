#include "xmss.cuh"
#include "wots.cuh"
#include "ltree.cuh"
#include "auth_path.cuh"
#include "hash_wrappers.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void compute_root_kernel(uint8_t* wots_pk, const uint8_t* auth_path, uint32_t leaf_idx, uint32_t tree_height, const uint8_t* pub_seed, uint8_t* computed_root) {
    if (threadIdx.x != 0) return; 

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    
    adrs.setType(XMSS_ADDR_TYPE_LTREE);
    adrs.setLTreeAddress(leaf_idx);

    uint8_t leaf[32];
    l_tree(wots_pk, pub_seed, &adrs, leaf);

    // Climb the Auth Path
    uint32_t current_idx = leaf_idx;
    uint8_t current_node[32];
    for(int i = 0; i < 32; i++) current_node[i] = leaf[i];

    adrs.setType(XMSS_ADDR_TYPE_HASHTREE);
    for(uint32_t h = 0; h < tree_height; h++) {
        adrs.setTreeHeight(h);
        adrs.setTreeIndex(current_idx >> 1);

        const uint8_t* sibling = &auth_path[h * 32];
        
        if (current_idx % 2 == 0) {
            h_hash(pub_seed, &adrs, current_node, sibling, (SHA256_CTX*)current_node);
        } else {
            h_hash(pub_seed, &adrs, sibling, current_node, (SHA256_CTX*)current_node);
        }
        current_idx >>= 1; 
    }

    for(int i = 0; i < 32; i++) computed_root[i] = current_node[i];
}


// FRONTEND API for xmss signing
void xmss_sign(const uint8_t* msg, size_t msg_len, uint32_t leaf_idx, uint32_t tree_height, const uint8_t* secret_seeds, const uint8_t* pub_seed, uint8_t* precomputed_tree_buffer, XMSS_Signature* out_sig) {
    
    // Hash the message (Mocked for simplicity)
    uint8_t h_msg_hash[32];
    for(int i = 0; i < 32; i++) h_msg_hash[i] = msg[i % msg_len];

    out_sig->leaf_idx = leaf_idx;
    for(int i = 0; i < 32; i++) out_sig->msg_randomness[i] = 0xAA;
    out_sig->auth_path_len = tree_height * 32;
    out_sig->auth_path = (uint8_t*)malloc(out_sig->auth_path_len);

    uint32_t num_leaves = 1 << tree_height;

    uint8_t *d_msg_hash, *d_secret_seeds, *d_pub_seed, *d_wots_sig, *d_tree_buffer, *d_auth_path;
    cudaMalloc(&d_msg_hash, 32); cudaMalloc(&d_secret_seeds, 67 * 32);
    cudaMalloc(&d_pub_seed, 32); cudaMalloc(&d_wots_sig, 67 * 32);
    cudaMalloc(&d_tree_buffer, num_leaves * 32); cudaMalloc(&d_auth_path, tree_height * 32);

    cudaMemcpy(d_msg_hash, h_msg_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_seeds, secret_seeds, 67 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_buffer, precomputed_tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);


    wots_sign_kernel<<<1, 67>>>(d_msg_hash, d_secret_seeds, d_pub_seed, leaf_idx, d_wots_sig);
    extract_auth_path_kernel<<<1, 1>>>(d_tree_buffer, num_leaves, leaf_idx, d_pub_seed, d_auth_path);
    cudaDeviceSynchronize();

    cudaMemcpy(out_sig->wots_sig, d_wots_sig, 67 * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_sig->auth_path, d_auth_path, tree_height * 32, cudaMemcpyDeviceToHost);

    cudaFree(d_msg_hash); cudaFree(d_secret_seeds); cudaFree(d_pub_seed);
    cudaFree(d_wots_sig); cudaFree(d_tree_buffer); cudaFree(d_auth_path);
}

// FRONTEND API for xmss verifying
bool xmss_verify(const XMSS_Signature* sig, const uint8_t* msg, size_t msg_len, const uint8_t* expected_pub_key, uint32_t tree_height, const uint8_t* pub_seed) {
    uint8_t h_msg_hash[32];
    for(int i = 0; i < 32; i++) h_msg_hash[i] = msg[i % msg_len]; // Match signature hash logic

    uint8_t *d_msg_hash, *d_wots_sig, *d_pub_seed, *d_wots_pk, *d_auth_path, *d_computed_root;
    cudaMalloc(&d_msg_hash, 32);
    cudaMalloc(&d_wots_sig, 67 * 32);
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_wots_pk, 67 * 32);
    cudaMalloc(&d_auth_path, tree_height * 32);
    cudaMalloc(&d_computed_root, 32);

    cudaMemcpy(d_msg_hash, h_msg_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wots_sig, sig->wots_sig, 67 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_auth_path, sig->auth_path, tree_height * 32, cudaMemcpyHostToDevice);

    // Reconstruct WOTS+ PK (67 threads)
    wots_verify_kernel<<<1, 67>>>(d_msg_hash, d_wots_sig, d_pub_seed, sig->leaf_idx, d_wots_pk);
    
    // Compress & Climb Tree (1 thread)
    compute_root_kernel<<<1, 1>>>(d_wots_pk, d_auth_path, sig->leaf_idx, tree_height, d_pub_seed, d_computed_root);
    
    cudaDeviceSynchronize();

    uint8_t h_computed_root[32];
    cudaMemcpy(h_computed_root, d_computed_root, 32, cudaMemcpyDeviceToHost);

    cudaFree(d_msg_hash); cudaFree(d_wots_sig); cudaFree(d_pub_seed);
    cudaFree(d_wots_pk); cudaFree(d_auth_path); cudaFree(d_computed_root);

    // Validation
    return memcmp(h_computed_root, expected_pub_key, 32) == 0;
}