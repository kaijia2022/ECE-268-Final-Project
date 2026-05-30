#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "wots.cuh"


int main() {
    printf("Starting WOTS+ Signature Generation...\n\n");

    uint8_t h_msg_hash[32];
    uint8_t h_pub_seed[32] = {0xAA};
    uint8_t* h_secret_seeds = (uint8_t*)malloc(67 * 32);
    uint8_t* h_wots_signature = (uint8_t*)malloc(67 * 32);

    for (int i = 0; i < 32; i++) h_msg_hash[i] = 0xAB; 
    for (int i = 0; i < 67 * 32; i++) h_secret_seeds[i] = (uint8_t)(i % 256);

    uint8_t *d_msg_hash, *d_secret_seeds, *d_pub_seed, *d_wots_signature;
    cudaMalloc(&d_msg_hash, 32);
    cudaMalloc(&d_secret_seeds, 67 * 32);
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_wots_signature, 67 * 32);

    cudaMemcpy(d_msg_hash, h_msg_hash, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_seeds, h_secret_seeds, 67 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, h_pub_seed, 32, cudaMemcpyHostToDevice);

    wots_sign_kernel<<<1, 67>>>(d_msg_hash, d_secret_seeds, d_pub_seed, d_wots_signature);
    cudaDeviceSynchronize();

    cudaMemcpy(h_wots_signature, d_wots_signature, 67 * 32, cudaMemcpyDeviceToHost);

    printf("WOTS+ Signature generated successfully! (67 blocks of 32 bytes)\n");
    for (int c = 0; c < 3; c++) {
        printf("Signature Block %d: ", c);
        for (int i = 0; i < 32; i++) {
            printf("%02x", h_wots_signature[c * 32 + i]);
        }
        printf("\n");
    }

    cudaFree(d_msg_hash); cudaFree(d_secret_seeds); cudaFree(d_pub_seed); cudaFree(d_wots_signature);
    free(h_secret_seeds); free(h_wots_signature);

    return 0;
}