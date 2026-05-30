#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "merkle.cuh"
#include "adrs.h"

__global__ void test_merkle_kernel(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, uint8_t* final_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ADRS adrs;
        adrs.setLayerAddress(0);
        adrs.setTreeAddress(1);

        build_hash_tree(tree_buffer, num_leaves, pub_seed, &adrs, final_root);
    }
}

int main() {
    uint32_t num_leaves = 8; // A tree of height 3
    printf("Starting Main Hash Tree test with %u leaves...\n\n", num_leaves);

    uint8_t h_pub_seed[32] = {0xCC};
    uint8_t h_final_root[32] = {0};
    uint8_t* h_tree_buffer = (uint8_t*)malloc(num_leaves * 32);

    for (int i = 0; i < num_leaves; i++) {
        for (int j = 0; j < 32; j++) {
            h_tree_buffer[i * 32 + j] = (uint8_t)(i + j);
        }
    }

    uint8_t *d_pub_seed, *d_tree_buffer, *d_final_root;
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_tree_buffer, num_leaves * 32);
    cudaMalloc(&d_final_root, 32);

    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_buffer, h_tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);

    test_merkle_kernel<<<1, 1>>>(d_tree_buffer, num_leaves, d_pub_seed, d_final_root);
    cudaDeviceSynchronize();

    cudaMemcpy(h_final_root, d_final_root, 32, cudaMemcpyDeviceToHost);

    printf("Final XMSS Public Key (Merkle Root):\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_final_root[i]);
    }
    printf("\n");

    cudaFree(d_pub_seed); cudaFree(d_tree_buffer); cudaFree(d_final_root);
    free(h_tree_buffer);

    return 0;
}