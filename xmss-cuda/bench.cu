// GPU benchmark for XMSS
// Times keygen sign and verify two ways so the report can tell host cost apart
// from compute cost
//   kernel only  cuda events around just the kernels with the device buffers
//                set up once so no malloc and no host device copy is counted
//   end to end   wall clock around the real host api which also pays the malloc
//                the copies and the free every call
// Then a keygen scaling sweep over tree heights. Keygen is where the gpu should
// win because a tree of 2^h leaves times 67 chains are all independent so a
// taller tree feeds the gpu more parallel work.
//
// Output is plain key value lines so the python runner can parse it the same
// way the cpu runner parses the C baseline.

#include "xmss.cuh"
#include "wots.cuh"
#include "auth_path.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// kernels live in xmss.cu with no header so forward declare the ones we drive
// directly for the kernel only timing. signatures must match xmss.cu exactly.
__global__ void keygen_leaves_kernel(const uint8_t* sk_seed, const uint8_t* pub_seed, uint8_t* tree_buffer);
__global__ void gen_leaf_secret_seeds_kernel(const uint8_t* sk_seed, uint32_t leaf_idx, uint8_t* secret_seeds_out);
__global__ void build_root_kernel(uint8_t* tree_buffer, uint32_t num_leaves, const uint8_t* pub_seed, uint8_t* root_out);
__global__ void prf_msg_kernel(const uint8_t* sk_prf, uint32_t idx, uint8_t* r_out);
__global__ void h_msg_kernel(const uint8_t* r, const uint8_t* root, uint32_t idx,
                             const uint8_t* msg, size_t msg_len, uint8_t* scratch, uint8_t* digest_out);
__global__ void compute_root_kernel(uint8_t* wots_pk, const uint8_t* auth_path, uint32_t leaf_idx,
                                    uint32_t tree_height, const uint8_t* pub_seed, uint8_t* computed_root);

// fixed seeds so the run is reproducible same as the cpu benchmark
static const uint8_t SK_SEED[32]  = {0x11};
static const uint8_t SK_PRF[32]   = {0x22};
static const uint8_t PUB_SEED[32] = {0x33};

// ---------- timers ----------

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// run a closure of kernel launches iters times and return microseconds per call
// cuda events sit on the default stream so they only see the kernel work
// body is a lambda so it can hold commas the launches need
template <typename F>
static double time_kernel(int iters, F body) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    body();                            // warm up one launch
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) body();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return (double)ms / iters * 1000.0;
}

// ---------- read the message ----------

static uint8_t* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz);
    size_t got = fread(buf, 1, sz, f);
    fclose(f);
    *out_len = got;
    return buf;
}

// ---------- single signature latency ----------

// time keygen sign and verify at one height
// kernel only pre allocates the device buffers and loops only the kernels
// end to end calls the real host api so it pays every malloc and copy
static void bench_single(uint32_t height, int iters, const uint8_t* msg, size_t msg_len) {
    uint32_t num_leaves = 1u << height;

    // build one real keypair on the host so sign and verify have a real tree
    XMSS_Keypair kp;
    xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED);

    // ----- keygen kernel only -----
    uint8_t *d_sk, *d_pub, *d_tree, *d_tree_copy, *d_root;
    cudaMalloc(&d_sk, 32); cudaMalloc(&d_pub, 32);
    cudaMalloc(&d_tree, num_leaves * 32); cudaMalloc(&d_tree_copy, num_leaves * 32);
    cudaMalloc(&d_root, 32);
    cudaMemcpy(d_sk, SK_SEED, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub, PUB_SEED, 32, cudaMemcpyHostToDevice);

    double k_keygen = time_kernel(iters, [&]() {
        keygen_leaves_kernel<<<num_leaves, 67>>>(d_sk, d_pub, d_tree);
        cudaMemcpy(d_tree_copy, d_tree, num_leaves * 32, cudaMemcpyDeviceToDevice);
        build_root_kernel<<<1, 1>>>(d_tree_copy, num_leaves, d_pub, d_root);
    });

    cudaFree(d_sk); cudaFree(d_pub); cudaFree(d_tree); cudaFree(d_tree_copy); cudaFree(d_root);

    // ----- sign kernel only -----
    // a real sign runs five kernels. extract_auth_path eats the tree so we keep
    // a master copy and refill a work buffer each round.
    uint8_t *s_sk, *s_prf, *s_pub, *s_root, *s_seeds, *s_r, *s_dig, *s_scratch, *s_msg;
    uint8_t *s_sig, *s_tree_master, *s_tree_work, *s_auth;
    cudaMalloc(&s_sk, 32); cudaMalloc(&s_prf, 32); cudaMalloc(&s_pub, 32); cudaMalloc(&s_root, 32);
    cudaMalloc(&s_seeds, 67 * 32); cudaMalloc(&s_r, 32); cudaMalloc(&s_dig, 32);
    cudaMalloc(&s_scratch, 128 + msg_len); cudaMalloc(&s_msg, msg_len); cudaMalloc(&s_sig, 67 * 32);
    cudaMalloc(&s_tree_master, num_leaves * 32); cudaMalloc(&s_tree_work, num_leaves * 32);
    cudaMalloc(&s_auth, height * 32);
    cudaMemcpy(s_sk, kp.sk_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(s_prf, kp.sk_prf, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(s_pub, kp.pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(s_root, kp.root, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(s_msg, msg, msg_len, cudaMemcpyHostToDevice);
    cudaMemcpy(s_tree_master, kp.tree_buffer, num_leaves * 32, cudaMemcpyHostToDevice);

    double k_sign = time_kernel(iters, [&]() {
        prf_msg_kernel<<<1, 1>>>(s_prf, 0, s_r);
        h_msg_kernel<<<1, 1>>>(s_r, s_root, 0, s_msg, msg_len, s_scratch, s_dig);
        gen_leaf_secret_seeds_kernel<<<1, 67>>>(s_sk, 0, s_seeds);
        wots_sign_kernel<<<1, 67>>>(s_dig, s_seeds, s_pub, 0, s_sig);
        cudaMemcpy(s_tree_work, s_tree_master, num_leaves * 32, cudaMemcpyDeviceToDevice);
        extract_auth_path_kernel<<<1, 1>>>(s_tree_work, num_leaves, 0, s_pub, s_auth);
    });

    cudaFree(s_sk); cudaFree(s_prf); cudaFree(s_pub); cudaFree(s_root); cudaFree(s_seeds);
    cudaFree(s_r); cudaFree(s_dig); cudaFree(s_scratch); cudaFree(s_msg); cudaFree(s_sig);
    cudaFree(s_tree_master); cudaFree(s_tree_work); cudaFree(s_auth);

    // ----- verify kernel only -----
    // make one real signature so verify has real inputs to chew on
    XMSS_Signature sig;
    kp.next_idx = 0;
    xmss_sign(&kp, msg, msg_len, &sig);

    uint8_t *v_r, *v_root, *v_dig, *v_scratch, *v_msg, *v_sig, *v_pub, *v_pk, *v_auth, *v_croot;
    cudaMalloc(&v_r, 32); cudaMalloc(&v_root, 32); cudaMalloc(&v_dig, 32);
    cudaMalloc(&v_scratch, 128 + msg_len); cudaMalloc(&v_msg, msg_len); cudaMalloc(&v_sig, 67 * 32);
    cudaMalloc(&v_pub, 32); cudaMalloc(&v_pk, 67 * 32); cudaMalloc(&v_auth, height * 32);
    cudaMalloc(&v_croot, 32);
    cudaMemcpy(v_r, sig.msg_randomness, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(v_root, kp.root, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(v_msg, msg, msg_len, cudaMemcpyHostToDevice);
    cudaMemcpy(v_sig, sig.wots_sig, 67 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(v_pub, kp.pub_seed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(v_auth, sig.auth_path, height * 32, cudaMemcpyHostToDevice);

    double k_verify = time_kernel(iters, [&]() {
        h_msg_kernel<<<1, 1>>>(v_r, v_root, sig.leaf_idx, v_msg, msg_len, v_scratch, v_dig);
        wots_verify_kernel<<<1, 67>>>(v_dig, v_sig, v_pub, sig.leaf_idx, v_pk);
        compute_root_kernel<<<1, 1>>>(v_pk, v_auth, sig.leaf_idx, height, v_pub, v_croot);
    });

    cudaFree(v_r); cudaFree(v_root); cudaFree(v_dig); cudaFree(v_scratch); cudaFree(v_msg);
    cudaFree(v_sig); cudaFree(v_pub); cudaFree(v_pk); cudaFree(v_auth); cudaFree(v_croot);
    free(sig.auth_path);

    // ----- end to end through the real host api -----
    int e2e_iters = iters < 200 ? iters : 200;
    double t0 = now_sec();
    for (int i = 0; i < e2e_iters; i++) { xmss_keypair_free(&kp); xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED); }
    double e_keygen = (now_sec() - t0) / e2e_iters * 1e6;

    t0 = now_sec();
    for (int i = 0; i < e2e_iters; i++) {
        XMSS_Signature s; kp.next_idx = 0;
        xmss_sign(&kp, msg, msg_len, &s);
        free(s.auth_path);
    }
    double e_sign = (now_sec() - t0) / e2e_iters * 1e6;

    kp.next_idx = 0;
    XMSS_Signature vs; xmss_sign(&kp, msg, msg_len, &vs);
    t0 = now_sec();
    for (int i = 0; i < e2e_iters; i++)
        xmss_verify(&vs, msg, msg_len, kp.root, kp.tree_height, kp.pub_seed);
    double e_verify = (now_sec() - t0) / e2e_iters * 1e6;
    free(vs.auth_path);

    xmss_keypair_free(&kp);

    printf("SINGLE height %u iters %d\n", height, iters);
    printf("kernel_keygen_us %.3f\n", k_keygen);
    printf("kernel_sign_us %.3f\n", k_sign);
    printf("kernel_verify_us %.3f\n", k_verify);
    printf("e2e_keygen_us %.3f\n", e_keygen);
    printf("e2e_sign_us %.3f\n", e_sign);
    printf("e2e_verify_us %.3f\n", e_verify);
}

// ---------- keygen scaling sweep ----------

// for one height time keygen kernel only split into the parallel leaf pass and
// the serial root climb plus the full end to end keygen
// the split shows the leaf pass scales with the gpu while the single thread
// root build does not so it eventually limits the speedup
static void bench_scale_one(uint32_t height) {
    uint32_t num_leaves = 1u << height;
    int iters = (int)(40000 / num_leaves);
    if (iters < 5) iters = 5;
    if (iters > 100) iters = 100;

    uint8_t *d_sk, *d_pub, *d_tree, *d_tree_copy, *d_root;
    cudaMalloc(&d_sk, 32); cudaMalloc(&d_pub, 32);
    cudaMalloc(&d_tree, num_leaves * 32); cudaMalloc(&d_tree_copy, num_leaves * 32);
    cudaMalloc(&d_root, 32);
    cudaMemcpy(d_sk, SK_SEED, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pub, PUB_SEED, 32, cudaMemcpyHostToDevice);

    // parallel leaf pass on its own
    double t_leaves = time_kernel(iters, [&]() {
        keygen_leaves_kernel<<<num_leaves, 67>>>(d_sk, d_pub, d_tree);
    });

    // populate the tree once then time only the serial root climb
    keygen_leaves_kernel<<<num_leaves, 67>>>(d_sk, d_pub, d_tree);
    cudaDeviceSynchronize();
    double t_root = time_kernel(iters, [&]() {
        cudaMemcpy(d_tree_copy, d_tree, num_leaves * 32, cudaMemcpyDeviceToDevice);
        build_root_kernel<<<1, 1>>>(d_tree_copy, num_leaves, d_pub, d_root);
    });

    double t_total = t_leaves + t_root;

    cudaFree(d_sk); cudaFree(d_pub); cudaFree(d_tree); cudaFree(d_tree_copy); cudaFree(d_root);

    // end to end keygen through the host api
    int e2e_iters = iters < 50 ? iters : 50;
    XMSS_Keypair kp;
    xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED);   // warm up
    double t0 = now_sec();
    for (int i = 0; i < e2e_iters; i++) { xmss_keypair_free(&kp); xmss_keygen(&kp, height, SK_SEED, SK_PRF, PUB_SEED); }
    double e2e = (now_sec() - t0) / e2e_iters * 1e6;
    xmss_keypair_free(&kp);

    double leaves_per_sec = num_leaves / (t_total * 1e-6);

    printf("SCALE height %u leaves %u kernel_leaves_us %.3f kernel_root_us %.3f kernel_total_us %.3f e2e_us %.3f leaves_per_sec %.1f\n",
           height, num_leaves, t_leaves, t_root, t_total, e2e, leaves_per_sec);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "../wots_pycuda/plaintext/short.txt";
    uint32_t single_height = (argc > 2) ? (uint32_t)atoi(argv[2]) : 3;
    int single_iters = (argc > 3) ? atoi(argv[3]) : 200;
    const char* scale_csv = (argc > 4) ? argv[4] : "3,5,7,10,12";

    size_t msg_len;
    uint8_t* msg = read_file(path, &msg_len);

    bench_single(single_height, single_iters, msg, msg_len);

    // walk the comma list of heights for the scaling sweep
    char buf[256];
    strncpy(buf, scale_csv, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char* tok = strtok(buf, ","); tok; tok = strtok(NULL, ","))
        bench_scale_one((uint32_t)atoi(tok));

    free(msg);
    return 0;
}
