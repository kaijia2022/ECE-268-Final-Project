#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "adrs.h"
#include "hash_wrappers.cuh"
#include "sha256.cuh" 


__global__ void test_wrappers_kernel(const uint8_t* seed, const uint8_t* key, const uint8_t* in_data, uint8_t* prf_out, uint8_t* f_out) {

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(12345);
    adrs.setType(XMSS_ADDR_TYPE_OTS);
    adrs.setOTSAddress(1);
    adrs.setChainAddress(2);
    adrs.setHashAddress(3);

    prf(seed, &adrs, (SHA256_CTX*)prf_out);

    f_chain(key, in_data, (SHA256_CTX*)f_out);
}

void print_hash(const char* label, uint8_t* hash) {
    printf("%s: ", label);
    for (int i = 0; i < 32; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

int main() {
    printf("Starting PRF and F wrapper tests...\n\n");

    uint8_t h_seed[32], h_key[32], h_data[32];
    uint8_t h_prf_out[32] = {0};
    uint8_t h_f_out[32] = {0};

    for (int i = 0; i < 32; i++) {
        h_seed[i] = i;        // 0x00, 0x01, 0x02 ...
        h_key[i]  = 255 - i;  // 0xff, 0xfe, 0xfd ...
        h_data[i] = 0xAA;     // 0xaa, 0xaa, 0xaa ...
    }

    uint8_t *d_seed, *d_key, *d_data, *d_prf_out, *d_f_out;
    cudaMalloc((void**)&d_seed, 32);
    cudaMalloc((void**)&d_key, 32);
    cudaMalloc((void**)&d_data, 32);
    cudaMalloc((void**)&d_prf_out, sizeof(SHA256_CTX));
    cudaMalloc((void**)&d_f_out, sizeof(SHA256_CTX));

    cudaMemcpy(d_seed, h_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, 32, cudaMemcpyHostToDevice);

    test_wrappers_kernel<<<1, 1>>>(d_seed, d_key, d_data, d_prf_out, d_f_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_prf_out, d_prf_out, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_out, d_f_out, 32, cudaMemcpyDeviceToHost);

    print_hash("PRF Result", h_prf_out);
    print_hash("F   Result", h_f_out);

    cudaFree(d_seed);
    cudaFree(d_key);
    cudaFree(d_data);
    cudaFree(d_prf_out);
    cudaFree(d_f_out);

    return 0;
}