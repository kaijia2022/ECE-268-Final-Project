#include "wots.cuh"
#include "hash_wrappers.cuh"

__device__ void wots_chain(const uint8_t* start_value, uint32_t start_idx, uint32_t steps, 
                           const uint8_t* pub_seed, ADRS* adrs, uint8_t* output) {
    
    // Initialize the output buffer with our starting value
    for (int i = 0; i < 32; i++) {
        output[i] = start_value[i];
    }

    for (uint32_t i = start_idx; i < start_idx + steps; i++) {
        
        // Update the address for this specific link in the chain
        adrs->setHashAddress(i);

        // Generate the Bitmask (KeyAndMask = 1)
        adrs->setKeyAndMask(1);
        uint8_t bitmask[32];
        prf(pub_seed, adrs, (SHA256_CTX*)bitmask);

        // Generate the Step Key (KeyAndMask = 0)
        adrs->setKeyAndMask(0);
        uint8_t step_key[32];
        prf(pub_seed, adrs, (SHA256_CTX*)step_key);

        // XOR the current hash value with the bitmask
        uint8_t xored_data[32];
        for (int j = 0; j < 32; j++) {
            xored_data[j] = output[j] ^ bitmask[j];
        }

        // Hash the XORed data and step_key together using F
        // The output of F becomes the new output for the next iteration
        f_chain(step_key, xored_data, (SHA256_CTX*)output);
    }
}

__host__ __device__ void get_chain_lengths(const uint8_t* msg_hash, uint8_t* lengths) {
    uint32_t csum = 0;

    // Base-W Conversion (The Message)
    // A 32-byte hash yields 64 nibbles (since w = 16)
    for (int i = 0; i < 32; i++) {
        uint8_t byte = msg_hash[i];
        
        // High nibble (first 4 bits)
        lengths[2 * i] = byte >> 4;           
        // Low nibble (last 4 bits)
        lengths[2 * i + 1] = byte & 0x0F;     

        // Checksum Calculation
        // Subtract each value from (w-1), w=16
        csum += (15 - lengths[2 * i]);
        csum += (15 - lengths[2 * i + 1]);
    }

    // Base-W Conversion (The Checksum)
    // The max possible checksum is 64*15 = 960 (0x3C0) => requires 10 bits (or 3 base-16 nibbles)
    // pad it into lengths[64], lengths[65], and lengths[66]
    lengths[64] = (csum >> 8) & 0x0F;
    lengths[65] = (csum >> 4) & 0x0F;
    lengths[66] = csum & 0x0F;
}

__global__ void wots_sign_kernel(const uint8_t* msg_hash, const uint8_t* secret_seeds, const uint8_t* pub_seed, uint32_t leaf_idx, uint8_t* wots_signature) {
    int chain_index = threadIdx.x;
    if (chain_index >= 67) return;

    uint8_t lengths[67];
    get_chain_lengths(msg_hash, lengths);
    uint32_t step_count = lengths[chain_index];

    ADRS thread_adrs;
    thread_adrs.setLayerAddress(0);
    thread_adrs.setTreeAddress(1);
    thread_adrs.setType(XMSS_ADDR_TYPE_OTS);
    
    thread_adrs.setOTSAddress(leaf_idx); 
    thread_adrs.setChainAddress(chain_index);

    const uint8_t* my_secret_seed = &secret_seeds[chain_index * 32];
    uint8_t* my_sig_block = &wots_signature[chain_index * 32];

    wots_chain(my_secret_seed, 0, step_count, pub_seed, &thread_adrs, my_sig_block);
}

__global__ void wots_verify_kernel(const uint8_t* msg_hash, const uint8_t* wots_sig, const uint8_t* pub_seed, uint32_t leaf_idx, uint8_t* wots_pk_out) {
    int chain_index = threadIdx.x;
    if (chain_index >= 67) return;

    uint8_t lengths[67];
    get_chain_lengths(msg_hash, lengths);
    
    uint32_t start_step = lengths[chain_index];
    uint32_t steps_to_run = 15 - start_step; 

    ADRS thread_adrs;
    thread_adrs.setLayerAddress(0);
    thread_adrs.setTreeAddress(1);
    thread_adrs.setType(XMSS_ADDR_TYPE_OTS);
    
    thread_adrs.setOTSAddress(leaf_idx); 
    thread_adrs.setChainAddress(chain_index);

    wots_chain(&wots_sig[chain_index * 32], start_step, steps_to_run, pub_seed, &thread_adrs, &wots_pk_out[chain_index * 32]);
}