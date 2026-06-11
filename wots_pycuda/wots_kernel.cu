__constant__ unsigned int K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n){
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_32(const unsigned int in[8], unsigned int out[8]){
    unsigned int w[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = in[i];
    w[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0u;
    w[15] = 256u;
    #pragma unroll
    for (int i = 16; i < 64; i++){
        unsigned int s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    unsigned int a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au,
                 e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    #pragma unroll
    for (int i = 0; i < 64; i++){
        unsigned int S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int t1 = h + S1 + ch + K[i] + w[i];
        unsigned int S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667u; out[1]=b+0xbb67ae85u; out[2]=c+0x3c6ef372u; out[3]=d+0xa54ff53au;
    out[4]=e+0x510e527fu; out[5]=f+0x9b05688cu; out[6]=g+0x1f83d9abu; out[7]=h+0x5be0cd19u;
}


__device__ void sha256_64(const unsigned int in[16], unsigned int out[8]){
    unsigned int w[64];

    /* block 1 message schedule: the 64-byte input */
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = in[i];
    #pragma unroll
    for (int i = 16; i < 64; i++){
        unsigned int s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    unsigned int a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au,
                 e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    #pragma unroll
    for (int i = 0; i < 64; i++){
        unsigned int S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int t1 = h + S1 + ch + K[i] + w[i];
        unsigned int S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    //intermediate hash H1 = IV + compress(IV, block1)
    unsigned int h0=a+0x6a09e667u, h1=b+0xbb67ae85u, h2=c+0x3c6ef372u, h3=d+0xa54ff53au,
                 h4=e+0x510e527fu, h5=f+0x9b05688cu, h6=g+0x1f83d9abu, h7=h+0x5be0cd19u;

    //SHA-256 padding for a 512-bit message
    w[0] = 0x80000000u;
    #pragma unroll
    for (int i = 1; i < 14; i++) w[i] = 0u;
    w[14] = 0u;
    w[15] = 512u;
    #pragma unroll
    for (int i = 16; i < 64; i++){
        unsigned int s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // compress block 2 from H1
    a=h0; b=h1; c=h2; d=h3; e=h4; f=h5; g=h6; h=h7;
    #pragma unroll
    for (int i = 0; i < 64; i++){
        unsigned int S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int t1 = h + S1 + ch + K[i] + w[i];
        unsigned int S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    out[0]=a+h0; out[1]=b+h1; out[2]=c+h2; out[3]=d+h3;
    out[4]=e+h4; out[5]=f+h5; out[6]=g+h6; out[7]=h+h7;
}

extern "C" __global__
void wots_chain(const unsigned int* __restrict__ start,
                const int* __restrict__ steps,
                unsigned int* __restrict__ out, int Nc){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nc) return;
    unsigned int state[8], tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = start[tid*8 + i];
    int s = steps[tid];
    for (int j = 0; j < s; j++){
        /* F(M) = SHA256(0x00*32 || M): prepend 32 zero bytes as domain separator */
        unsigned int in64[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) in64[i]   = 0u;
        #pragma unroll
        for (int i = 0; i < 8; i++) in64[8+i] = state[i];
        sha256_64(in64, tmp);
        #pragma unroll
        for (int i = 0; i < 8; i++) state[i] = tmp[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) out[tid*8 + i] = state[i];
}
