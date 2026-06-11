#include "xmss.cuh"
#include "wots.cuh"
#include "ltree.cuh"
#include "merkle.cuh"
#include "auth_path.cuh"
#include "hash_wrappers.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Derive the 67 wots secret seeds for one leaf and hash each chain to the end
// then squeeze the 67 public chain ends down into a single leaf node.
// One block does one leaf so launching with many blocks fills the whole tree.
__global__ void keygen_leaves_kernel(const uint8_t* sk_seed, const uint8_t* pub_seed, uint8_t* tree_buffer) {
    uint32_t leaf_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= 67) return;

    // Shared so all 67 chains can pool their public ends for the l_tree step.
    __shared__ uint8_t shared_wots_pk[67 * 32];

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    adrs.setType(XMSS_ADDR_TYPE_OTS);
    adrs.setOTSAddress(leaf_idx);
    adrs.setChainAddress(tid);

    // Secret seed for this chain comes from the master sk_seed through the prf.
    uint8_t secret_seed[32];
    prf(sk_seed, &adrs, (SHA256_CTX*)secret_seed);

    // Walk the chain all 15 steps to reach the public end of this chain.
    wots_chain(secret_seed, 0, 15, pub_seed, &adrs, &shared_wots_pk[tid * 32]);

    __syncthreads();

    // One thread folds the 67 ends into the leaf node for this index.
    if (tid == 0) {
        adrs.setType(XMSS_ADDR_TYPE_LTREE);
        adrs.setLTreeAddress(leaf_idx);
        l_tree(shared_wots_pk, pub_seed, &adrs, &tree_buffer[leaf_idx * 32]);
    }
}

// Derive the 67 wots secret seeds for the one leaf we are about to sign with.
// Same derivation as keygen so the signature lines up with the tree.
__global__ void gen_leaf_secret_seeds_kernel(const uint8_t* sk_seed, uint32_t leaf_idx, uint8_t* secret_seeds_out) {
    int chain_index = threadIdx.x;
    if (chain_index >= 67) return;

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);
    adrs.setType(XMSS_ADDR_TYPE_OTS);
    adrs.setOTSAddress(leaf_idx);
    adrs.setChainAddress(chain_index);

    prf(sk_seed, &adrs, (SHA256_CTX*)&secret_seeds_out[chain_index * 32]);
}

// Climb the leaf nodes layer by layer into the single merkle root.
// build_hash_tree eats its buffer so we always hand it a throwaway copy.
__global__ void build_root_kernel(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, uint8_t* root_out) {
    if (threadIdx.x != 0) return;

    ADRS adrs;
    adrs.setLayerAddress(0);
    adrs.setTreeAddress(1);

    build_hash_tree(tree_buffer, num_leaves, pub_seed, &adrs, root_out);
}

// Message randomness r for one leaf index.
// r = SHA256 toByte(3,32) || sk_prf || toByte(idx,32). It is deterministic per
// index so the same idx always gives the same r but every leaf gets its own.
__global__ void prf_msg_kernel(const uint8_t* sk_prf, uint32_t idx, uint8_t* r_out) {
    if (threadIdx.x != 0) return;

    uint8_t buffer[96];
    for (int i = 0; i < 31; i++) buffer[i] = 0x00;
    buffer[31] = 0x03;
    for (int i = 0; i < 32; i++) buffer[32 + i] = sk_prf[i];
    for (int i = 0; i < 28; i++) buffer[64 + i] = 0x00;
    buffer[92] = (uint8_t)(idx >> 24);
    buffer[93] = (uint8_t)(idx >> 16);
    buffer[94] = (uint8_t)(idx >> 8);
    buffer[95] = (uint8_t)(idx);

    sha256(buffer, 96, (SHA256_CTX*)r_out);
}

// The randomized message digest both sign and verify feed into wots.
// digest = SHA256 toByte(2,32) || r || root || toByte(idx,32) || msg.
// Binding root and idx into the hash stops a signature from being moved to
// another leaf or another key. scratch holds the bytes we are about to hash.
__global__ void h_msg_kernel(const uint8_t* r, const uint8_t* root, uint32_t idx,
                             const uint8_t* msg, size_t msg_len,
                             uint8_t* scratch, uint8_t* digest_out) {
    if (threadIdx.x != 0) return;

    for (int i = 0; i < 31; i++) scratch[i] = 0x00;
    scratch[31] = 0x02;
    for (int i = 0; i < 32; i++) scratch[32 + i] = r[i];
    for (int i = 0; i < 32; i++) scratch[64 + i] = root[i];
    for (int i = 0; i < 28; i++) scratch[96 + i] = 0x00;
    scratch[124] = (uint8_t)(idx >> 24);
    scratch[125] = (uint8_t)(idx >> 16);
    scratch[126] = (uint8_t)(idx >> 8);
    scratch[127] = (uint8_t)(idx);
    for (size_t i = 0; i < msg_len; i++) scratch[128 + i] = msg[i];

    sha256(scratch, 128 + msg_len, (SHA256_CTX*)digest_out);
}


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


// FRONTEND API for xmss key generation
void xmss_keygen(XMSS_Keypair* kp, uint32_t tree_height, const uint8_t* sk_seed, const uint8_t* sk_prf, const uint8_t* pub_seed) {
    uint32_t num_leaves = 1 << tree_height;

    kp->tree_height = tree_height;
    kp->num_leaves = num_leaves;
    kp->next_idx = 0;                       // fresh key starts at leaf 0
    for (int i = 0; i < 32; i++) kp->sk_seed[i] = sk_seed[i];
    for (int i = 0; i < 32; i++) kp->sk_prf[i] = sk_prf[i];
    for (int i = 0; i < 32; i++) kp->pub_seed[i] = pub_seed[i];
    kp->tree_buffer = (uint8_t*)malloc(num_leaves * 32);

    uint8_t *d_sk_seed, *d_pub_seed, *d_tree, *d_tree_copy, *d_root;
    cudaMalloc(&d_sk_seed, 32);
    cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_tree, num_leaves * 32);
    cudaMalloc(&d_tree_copy, num_leaves * 32);
    cudaMalloc(&d_root, 32);

    cudaMemcpy(d_sk_seed, sk_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, pub_seed, 32, cudaMemcpyHostToDevice);

    // Every leaf is computed for real, one block per leaf.
    keygen_leaves_kernel<<<num_leaves, 67>>>(d_sk_seed, d_pub_seed, d_tree);

    // Build the root from a copy because the build eats its buffer.
    cudaMemcpy(d_tree_copy, d_tree, num_leaves * 32, cudaMemcpyDeviceToDevice);
    build_root_kernel<<<1, 1>>>(d_tree_copy, num_leaves, d_pub_seed, d_root);
    cudaDeviceSynchronize();

    cudaMemcpy(kp->root, d_root, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(kp->tree_buffer, d_tree, num_leaves * 32, cudaMemcpyDeviceToHost);

    cudaFree(d_sk_seed); cudaFree(d_pub_seed);
    cudaFree(d_tree); cudaFree(d_tree_copy); cudaFree(d_root);
}

void xmss_keypair_free(XMSS_Keypair* kp) {
    free(kp->tree_buffer);
    kp->tree_buffer = NULL;
}

// FRONTEND API for xmss signing
bool xmss_sign(XMSS_Keypair* kp, const uint8_t* msg, size_t msg_len, XMSS_Signature* out_sig) {
    // State check, refuse to sign once every leaf has been used.
    if (kp->next_idx >= kp->num_leaves) {
        return false;
    }

    uint32_t leaf_idx = kp->next_idx;       // take the next free leaf
    uint32_t tree_height = kp->tree_height;
    uint32_t num_leaves = kp->num_leaves;

    out_sig->leaf_idx = leaf_idx;
    out_sig->auth_path_len = tree_height * 32;
    out_sig->auth_path = (uint8_t*)malloc(out_sig->auth_path_len);

    uint8_t *d_sk_seed, *d_sk_prf, *d_pub_seed, *d_root, *d_secret_seeds;
    uint8_t *d_r, *d_digest, *d_scratch, *d_msg, *d_wots_sig, *d_tree_buffer, *d_auth_path;
    cudaMalloc(&d_sk_seed, 32); cudaMalloc(&d_sk_prf, 32); cudaMalloc(&d_pub_seed, 32);
    cudaMalloc(&d_root, 32); cudaMalloc(&d_secret_seeds, 67 * 32);
    cudaMalloc(&d_r, 32); cudaMalloc(&d_digest, 32); cudaMalloc(&d_scratch, 128 + msg_len);
    cudaMalloc(&d_msg, msg_len); cudaMalloc(&d_wots_sig, 67 * 32);
    cudaMalloc(&d_tree_buffer, num_leaves * 32); cudaMalloc(&d_auth_path, tree_height * 32);

    cudaMemcpy(d_sk_seed, kp->sk_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk_prf, kp->sk_prf, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, kp->pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_root, kp->root, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg, msg_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_buffer, kp->tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);

    // Real message hash, randomized and bound to this leaf and this root.
    prf_msg_kernel<<<1, 1>>>(d_sk_prf, leaf_idx, d_r);
    h_msg_kernel<<<1, 1>>>(d_r, d_root, leaf_idx, d_msg, msg_len, d_scratch, d_digest);

    // Derive this leaf private key then sign the digest with it.
    gen_leaf_secret_seeds_kernel<<<1, 67>>>(d_sk_seed, leaf_idx, d_secret_seeds);
    wots_sign_kernel<<<1, 67>>>(d_digest, d_secret_seeds, d_pub_seed, leaf_idx, d_wots_sig);

    // Pull the auth path out of a throwaway copy of the tree.
    extract_auth_path_kernel<<<1, 1>>>(d_tree_buffer, num_leaves, leaf_idx, d_pub_seed, d_auth_path);
    cudaDeviceSynchronize();

    cudaMemcpy(out_sig->msg_randomness, d_r, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_sig->wots_sig, d_wots_sig, 67 * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_sig->auth_path, d_auth_path, tree_height * 32, cudaMemcpyDeviceToHost);

    cudaFree(d_sk_seed); cudaFree(d_sk_prf); cudaFree(d_pub_seed); cudaFree(d_root);
    cudaFree(d_secret_seeds); cudaFree(d_r); cudaFree(d_digest); cudaFree(d_scratch);
    cudaFree(d_msg); cudaFree(d_wots_sig); cudaFree(d_tree_buffer); cudaFree(d_auth_path);

    // State update, this leaf is spent so move on to the next one.
    kp->next_idx++;
    return true;
}

// FRONTEND API for xmss verifying
bool xmss_verify(const XMSS_Signature* sig, const uint8_t* msg, size_t msg_len, const uint8_t* expected_pub_key, uint32_t tree_height, const uint8_t* pub_seed) {
    uint8_t *d_r, *d_root, *d_digest, *d_scratch, *d_msg;
    uint8_t *d_wots_sig, *d_pub_seed, *d_wots_pk, *d_auth_path, *d_computed_root;
    cudaMalloc(&d_r, 32); cudaMalloc(&d_root, 32); cudaMalloc(&d_digest, 32);
    cudaMalloc(&d_scratch, 128 + msg_len); cudaMalloc(&d_msg, msg_len);
    cudaMalloc(&d_wots_sig, 67 * 32); cudaMalloc(&d_pub_seed, 32); cudaMalloc(&d_wots_pk, 67 * 32);
    cudaMalloc(&d_auth_path, tree_height * 32); cudaMalloc(&d_computed_root, 32);

    // The verifier hashes the message the same way the signer did.
    // It uses the trusted public key as the root and the r from the signature.
    cudaMemcpy(d_r, sig->msg_randomness, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_root, expected_pub_key, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg, msg_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wots_sig, sig->wots_sig, 67 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub_seed, pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_auth_path, sig->auth_path, tree_height * 32, cudaMemcpyHostToDevice);

    h_msg_kernel<<<1, 1>>>(d_r, d_root, sig->leaf_idx, d_msg, msg_len, d_scratch, d_digest);

    // Rebuild the wots public key from the signature then fold and climb to a root.
    wots_verify_kernel<<<1, 67>>>(d_digest, d_wots_sig, d_pub_seed, sig->leaf_idx, d_wots_pk);
    compute_root_kernel<<<1, 1>>>(d_wots_pk, d_auth_path, sig->leaf_idx, tree_height, d_pub_seed, d_computed_root);
    cudaDeviceSynchronize();

    uint8_t h_computed_root[32];
    cudaMemcpy(h_computed_root, d_computed_root, 32, cudaMemcpyDeviceToHost);

    cudaFree(d_r); cudaFree(d_root); cudaFree(d_digest); cudaFree(d_scratch); cudaFree(d_msg);
    cudaFree(d_wots_sig); cudaFree(d_pub_seed); cudaFree(d_wots_pk);
    cudaFree(d_auth_path); cudaFree(d_computed_root);

    // The rebuilt root must match the public key for the signature to be good.
    return memcmp(h_computed_root, expected_pub_key, 32) == 0;
}
