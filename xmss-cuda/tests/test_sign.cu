#include <stdio.h>
#include <stdint.h>
#include "wots.cuh"

int main() {
    printf("Starting Message Digest to Base-W Checksum test...\n\n");

    // Mock a 32-byte Message Digest
    uint8_t msg_hash[32];
    for (int i = 0; i < 32; i++) {
        msg_hash[i] = 0xAB; // 0xAB: nibbles are 10 (A) and 11 (B)
    }

    uint8_t lengths[67] = {0};

    get_chain_lengths(msg_hash, lengths);

    printf("First 4 Message Chain Lengths (Expected: 10, 11, 10, 11):\n");
    for (int i = 0; i < 4; i++) {
        printf("Chain %d: %u\n", i, lengths[i]);
    }

    // We got 64 nibbles alternating between 0xA and 0xB (32 nibbles each)
    // => 0xF - 0xA = 5; 32 * 5 = 160
    // => 0xF - 0xB = 4; 32 * 4 = 128
    // Total Expected Checksum = 160 + 128 = 288 = 0x120
    // Final 3 nibbles should be 1, 2, 0
    
    printf("\nFinal 3 Checksum Chain Lengths (Expected: 1, 2, 0):\n");
    printf("Chain 64: %u\n", lengths[64]);
    printf("Chain 65: %u\n", lengths[65]);
    printf("Chain 66: %u\n", lengths[66]);

    return 0;
}