#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "ltree.cuh"
#include "adrs.h"

__global__ void test_ltree_kernel(const uint8_t* wots_pks, const uint8_t* pub_seed, uint8_t* ltree_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ADRS adrs;
        adrs.setLayerAddress(0);
        adrs.setTreeAddress(1);
        
        // Compressing the leaves of WOTS+ signature index 0
        adrs.setLTreeAddress(0);

        l_tree(wots_pks, pub_seed, &adrs, ltree_root);
    }
}

int main() {
    printf("Starting L-Tree Compression test...\n\n");

    uint8_t h_pub_seed[32] = {0xBB}; // Dummy public seed
    uint8_t* h_wots_pks = (uint8_t*)malloc(67 * 32);
    uint8_t h_ltree_root[32] = {0};

    // Fill the 67 "keys" with sequential dummy data
    for (int i = 0; i < 67 * 32; i++) {
        h_wots_pks[i] = (uint8_t)(i % 256);
    }

    uint8_t *d_pub_seed, *d_wots_pks, *d_ltree_root;
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_wots_pks, 67 * 32);
    cudaMalloc(&d_ltree_root, 32);

    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wots_pks, h_wots_pks, 67 * 32, cudaMemcpyHostToDevice);

    test_ltree_kernel<<<1, 1>>>(d_wots_pks, d_pub_seed, d_ltree_root);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ltree_root, d_ltree_root, 32, cudaMemcpyDeviceToHost);

    printf("L-Tree Root Node:\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_ltree_root[i]);
    }
    printf("\n");

    cudaFree(d_pub_seed); cudaFree(d_wots_pks); cudaFree(d_ltree_root);
    free(h_wots_pks);

    return 0;
}