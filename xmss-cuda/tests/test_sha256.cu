#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "sha256.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void testHashKernel(uint8_t* d_data, size_t len, SHA256_CTX* d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        sha256(d_data, len, d_output);
    }
}

int main() {

    const char* message = "abc"; 
    size_t len = 3;             
    
    uint8_t* d_data;
    SHA256_CTX* d_output;
    CUDA_CHECK(cudaMalloc(&d_data, len));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(SHA256_CTX)));

    CUDA_CHECK(cudaMemcpy(d_data, message, len, cudaMemcpyHostToDevice));

    testHashKernel<<<1, 1>>>(d_data, len, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    SHA256_CTX h_output;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(SHA256_CTX), cudaMemcpyDeviceToHost));

    std::cout << "Input:  \"" << message << "\"\n";
    std::cout << "SHA256: ";
    for (int i = 0; i < 32; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)h_output.hash[i];
    }
    std::cout << std::dec << std::endl; 

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}