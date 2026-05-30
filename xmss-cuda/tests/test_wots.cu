#include <stdio.h>
#include <stdint.h>
#include "wots.cuh"
#include "adrs.h"

// WOTS+ Kernel: Launches 67 threads; Each thread executes 1 chain
__global__ void test_wots_kernel(const uint8_t* pub_seed, const uint8_t* secret_seeds, uint8_t* public_keys) {
    int chain_index = threadIdx.x; // Will run from 0 to 66
    
    if (chain_index >= 67) return;

    ADRS thread_adrs;
    thread_adrs.setLayerAddress(0);
    thread_adrs.setTreeAddress(1);
    thread_adrs.setType(XMSS_ADDR_TYPE_OTS);
    thread_adrs.setOTSAddress(0);
    thread_adrs.setChainAddress(chain_index); 

    // Get the specific secret seed for this chain (32 bytes per thread)
    const uint8_t* my_secret_seed = &secret_seeds[chain_index * 32];
    
    uint8_t* my_public_key = &public_keys[chain_index * 32];

    // Execute the 15-step chain
    wots_chain(my_secret_seed, 0, 15, pub_seed, &thread_adrs, my_public_key);
}

int main() {
    printf("Starting 67-thread WOTS+ Multi-Chain execution...\n\n");

    uint8_t h_pub_seed[32] = {0xAA}; // Dummy public seed
    uint8_t* h_secret_seeds = (uint8_t*)malloc(67 * 32);
    uint8_t* h_public_keys = (uint8_t*)malloc(67 * 32);

    for (int i = 0; i < 67 * 32; i++) {
        h_secret_seeds[i] = (uint8_t)(i % 256); 
    }

    uint8_t *d_pub_seed, *d_secret_seeds, *d_public_keys;
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_secret_seeds, 67 * 32);
    cudaMalloc(&d_public_keys, 67 * 32);

    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_seeds, h_secret_seeds, 67 * 32, cudaMemcpyHostToDevice);

    test_wots_kernel<<<1, 67>>>(d_pub_seed, d_secret_seeds, d_public_keys);
    cudaDeviceSynchronize();

    cudaMemcpy(h_public_keys, d_public_keys, 67 * 32, cudaMemcpyDeviceToHost);

    for (int c = 0; c < 3; c++) {
        printf("Chain %d Final PK: ", c);
        for (int i = 0; i < 32; i++) printf("%02x", h_public_keys[c * 32 + i]);
        printf("\n");
    }

    cudaFree(d_pub_seed); cudaFree(d_secret_seeds); cudaFree(d_public_keys);
    free(h_secret_seeds); free(h_public_keys);

    return 0;
}