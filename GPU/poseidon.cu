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
}

__device__ void mix_layer(uint256* state) {
    int i = threadIdx.x;
    uint256 res;
    
    // 1. Initialize local accumulator to zero
    res.limbs[0] = 0; res.limbs[1] = 0; 
    res.limbs[2] = 0; res.limbs[3] = 0;

    // 2. Compute the dot product for this thread's row
    // Each thread 'i' computes: res = Sum(M[i][j] * state[j])
    if (i < T) {
        for (int j = 0; j < T; j++) {
            uint256 mat_val = load_const_uint256(M_MATRIX, i * T + j);
            uint256 term = mod_mul(state[j], mat_val);
            res = mod_add(res, term);
        }
    }

    // 3. FIRST BARRIER: Ensure all threads have finished reading from 'state'
    __syncthreads();

    // 4. Update shared memory with the new result
    if (i < T) {
        state[i] = res;
    }

    // 5. SECOND BARRIER: Ensure all 'state' updates are visible before moving to the next round
    __syncthreads();
}

__device__ void run_poseidon_permutation_parallel(uint256 state[T]) {
    int rc_idx = 0;
    // Half-full rounds
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < T; i++) {
            uint256 c = load_const_uint256(C_CONSTANTS, rc_idx++);
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

__global__ void poseidon_sponge_kernel(uint8_t* d_input, int* d_offsets, int* d_lengths, uint256* d_output) {
    // Each block gets its own private 'state' in shared memory
    // Since each block only handles ONE hash, the size is exactly T.
    __shared__ uint256 state[T];
    
    int hash_idx = blockIdx.x; // Block 0 handles Line 0, Block 1 handles Line 1...
    int tid = threadIdx.x;     
    
    // Pointers to this block's specific input data
    uint8_t* my_input = d_input + d_offsets[hash_idx];
    int my_len = d_lengths[hash_idx];

    // 1. Initialization (Parallel)
    // Only the first T threads do work; others sit idle but stay in sync.
    if (tid < T) {
        state[tid].limbs[0] = 0; state[tid].limbs[1] = 0;
        state[tid].limbs[2] = 0; state[tid].limbs[3] = 0;
    }
    
    // Ensure the zero-initialization is finished
    __syncthreads();

    // 2. Absorption Phase
    for (int i = 0; i < my_len; i += 62) {
        // Only thread 0 loads the data into the state
        if (tid == 0) {
            int len1 = (i + 31 <= my_len) ? 31 : (my_len - i);
            uint256 m1 = bytes_to_uint256(my_input + i, len1);
            uint256 m2 = {0,0,0,0};

            if (i + 31 < my_len) {
                int len2 = (i + 62 <= my_len) ? 31 : (my_len - (i + 31));
                m2 = bytes_to_uint256(my_input + i + 31, len2);
            }

            uint256 m1_mont, m2_mont;
            m1 = to_montgomery(m1);
            m2 = to_montgomery(m2);
            state[1] = mod_add(state[1], m1);
            state[2] = mod_add(state[2], m2);
        }
        
        // Wait for thread 0 to finish the additions
        __syncthreads();
        
        // 3. Run the Permutation
        // This function uses threads 0, 1, and 2 in parallel for the mix layer.
        run_poseidon_permutation_parallel(state); 
        
        // run_poseidon_permutation_parallel should have a __syncthreads() 
        // at its very end to ensure it's ready for the next iteration.
    }

    // 4. Output result
    // Usually state[1] is the designated output element for Poseidon T=3
    if (tid == 1) {
        state[1] = from_montgomery(state[1]);
        d_output[hash_idx] = state[1];
    }
}

void launch_poseidon_kernel(uint8_t* d_in, int* d_offsets, int* d_lengths, uint256* d_out, int num_hashes) {
    // We launch 32 threads even though we only use 3.
    // This aligns with the GPU's "Warp" size for better scheduling.
    int threads_per_block = 3; 
    
    // Shared memory is only needed for ONE state array (T elements) per block.
    size_t shared_mem_size = T * sizeof(uint256);

    poseidon_sponge_kernel<<<num_hashes, threads_per_block>>>(
        d_in, d_offsets, d_lengths, d_out
    );
}