#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "auth_path.cuh"
#include "adrs.h"

int main() {
    uint32_t num_leaves = 8; // Tree height = 3
    uint32_t leaf_index = 2; // Signing with the 3rd leaf
    uint32_t tree_height = 3;

    printf("Extracting Auth Path for Leaf %u (Tree Height %u)...\n\n", leaf_index, tree_height);

    uint8_t h_pub_seed[32] = {0xDD};
    uint8_t* h_tree_buffer = (uint8_t*)malloc(num_leaves * 32);
    uint8_t* h_auth_path = (uint8_t*)malloc(tree_height * 32);

    // Create a dummy tree
    for (int i = 0; i < num_leaves; i++) {
        for (int j = 0; j < 32; j++) h_tree_buffer[i * 32 + j] = (uint8_t)i; 
    }


    uint8_t *d_pub_seed, *d_tree_buffer, *d_auth_path;
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_tree_buffer, num_leaves * 32);
    cudaMalloc(&d_auth_path, tree_height * 32);

    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_buffer, h_tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);

    extract_auth_path_kernel<<<1, 1>>>(d_tree_buffer, num_leaves, leaf_index, d_pub_seed, d_auth_path);

    cudaDeviceSynchronize();

    cudaMemcpy(h_auth_path, d_auth_path, tree_height * 32, cudaMemcpyDeviceToHost);

    printf("Authentication Path:\n");
    for (int h = 0; h < tree_height; h++) {
        printf("Layer %d Sibling: ", h);
        for (int i = 0; i < 32; i++) printf("%02x", h_auth_path[h * 32 + i]);
        printf("\n");
    }

    cudaFree(d_pub_seed); cudaFree(d_tree_buffer); cudaFree(d_auth_path);
    free(h_tree_buffer); free(h_auth_path);

    return 0;
}