#include "poseidon.h"
#include <stdio.h>

// Definitions
__device__ __constant__ uint64_t C_CONSTANTS[TOTAL_RC * 4];
__device__ __constant__ uint64_t M_MATRIX[T * T * 4];

// Helper to safely fetch a uint256 from the constant array
__device__ __forceinline__ uint256 load_const_uint256(const uint64_t* src, int index) {
    uint256 res;
    res.limbs[0] = src[index * 4 + 0];
    res.limbs[1] = src[index * 4 + 1];
    res.limbs[2] = src[index * 4 + 2];
    res.limbs[3] = src[index * 4 + 3];
    return res;
}

__device__ void sbox(uint256 &x) {
    uint256 x2 = mod_mul(x, x);
    uint256 x4 = mod_mul(x2, x2);
    x = mod_mul(x4, x);
    //x = mod_mul(x, x);
}

__device__ void mix_layer(uint256* state) {
    uint256 next[T];
    for (int i = 0; i < T; i++) {
        // Start with 0
        next[i].limbs[0] = 0; next[i].limbs[1] = 0; 
        next[i].limbs[2] = 0; next[i].limbs[3] = 0;

        for (int j = 0; j < T; j++) {
            uint256 mat_val = load_const_uint256(M_MATRIX, i * T + j);
            uint256 term = mod_mul(state[j], mat_val);
            next[i] = mod_add(next[i], term);
        }
    }
    for (int i = 0; i < T; i++) state[i] = next[i];
}

__device__ void run_poseidon_permutation(uint256 state[T]) {
    int rc_idx = 0;
    /*for (int r = 0; r < 1; r++) {
        for (int i = 0; i < T; i++) {
            uint256 c = load_const_uint256(C_CONSTANTS, rc_idx++);
            state[i] = mod_add(state[i], c);
            printf("Pre mult hash: 0x%016llx%016llx%016llx%016llx\n", 
            state[1].limbs[3], state[1].limbs[2], state[1].limbs[1], state[1].limbs[0]);
            sbox(state[i]);
        }
        mix_layer(state);
    }*/
    // Half-full rounds
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < T; i++) {
            uint256 c = load_const_uint256(C_CONSTANTS, rc_idx++);
            //printf("c limbs are: %lld, %lld, %lld, %lld,", c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3]);
            state[i] = mod_add(state[i], c);
            sbox(state[i]);
        }
        mix_layer(state);
    }
    // Partial rounds
    for (int r = 0; r < RP; r++) {
        for (int i = 0; i < T; i++) {
            uint256 c = load_const_uint256(C_CONSTANTS, rc_idx++);
            state[i] = mod_add(state[i], c);
        }
        sbox(state[0]);
        mix_layer(state);
    }
    // Half-full rounds
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < T; i++) {
            uint256 c = load_const_uint256(C_CONSTANTS, rc_idx++);
            state[i] = mod_add(state[i], c);
            sbox(state[i]);
        }
        mix_layer(state);
    }
}

__global__ void poseidon_sponge_kernel(uint8_t* d_input, uint256* d_output, int total_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // (Optional: remove the printf once you've confirmed it's running)
    printf("Kernel Started!\n");
    uint256 state[T];
    for(int i=0; i<T; i++) { 
        state[i].limbs[0]=0; state[i].limbs[1]=0; 
        state[i].limbs[2]=0; state[i].limbs[3]=0; 
    }

    for (int i = 0; i < total_len; i += 62) {
        int len1 = (i + 31 <= total_len) ? 31 : (total_len - i);
        uint256 m1 = bytes_to_uint256(d_input + i, len1);
        
        uint256 m2 = {0,0,0,0};
        if (i + 31 < total_len) {
            int len2 = (i + 62 <= total_len) ? 31 : (total_len - (i + 31));
            m2 = bytes_to_uint256(d_input + i + 31, len2);
        }

        state[1] = mod_add(state[1], m1);
        state[2] = mod_add(state[2], m2);
        
        run_poseidon_permutation(state);
    }
    d_output[tid] = state[1];
}

// Keep your existing wrapper functions (launch_poseidon_kernel and load_constants_to_gpu)

// WRAPPER FUNCTIONS
void launch_poseidon_kernel(uint8_t* d_in, uint256* d_out, size_t total_len) {
    poseidon_sponge_kernel<<<1, 1>>>(d_in, d_out, (int)total_len);
}


void load_constants_to_gpu(const uint64_t* h_rc, size_t rc_count, const uint64_t* h_mds, size_t mds_count) {
    cudaError_t err;

    // Use sizeof() on the actual symbol to ensure perfect alignment
    err = cudaMemcpyToSymbol(C_CONSTANTS, h_rc, rc_count * sizeof(uint64_t));
    if (err != cudaSuccess) {
        printf("RC Copy Failed: %s (Expected %zu bytes)\n", cudaGetErrorString(err), rc_count * sizeof(uint64_t));
    }

    err = cudaMemcpyToSymbol(M_MATRIX, h_mds, mds_count * sizeof(uint64_t));
    if (err != cudaSuccess) {
        printf("MDS Copy Failed: %s (Expected %zu bytes)\n", cudaGetErrorString(err), mds_count * sizeof(uint64_t));
    }
}