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

/*__device__ void mix_layer(uint256* state) {
    uint256 next[T];
    for (int i = 0; i < T; i++) {
        // Start with 0
        next[i].limbs[0] = 0; next[i].limbs[1] = 0; 
        next[i].limbs[2] = 0; next[i].limbs[3] = 0;

        mod_mul[]
        for (int j = 0; j < T; j++) {
            uint256 mat_val = load_const_uint256(M_MATRIX, i * T + j);
            uint256 term = mod_mul(state[j], mat_val);
            next[i] = mod_add(next[i], term);
        }
    }
    for (int i = 0; i < T; i++) state[i] = next[i];
}*/

__device__ void mix_layer/*_parallel*/(uint256* state) {
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

/*__global__ void poseidon_sponge_kernel(uint8_t* d_input, uint256* d_output, int total_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // (Optional: remove the printf once you've confirmed it's running)
    //printf("Kernel Started!\n");
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
}*/

__global__ void poseidon_sponge_kernel(uint8_t* d_input, uint256* d_output, int total_len) {
    // 1. Define shared memory. 
    // SPONGES_PER_BLOCK = blockDim.x / T
    __shared__ uint256 shared_states[MAX_PARALLEL_SPONGES*T]; 

    // 2. Identify which sponge this thread belongs to, and which element (i) it owns
    int sponge_idx = threadIdx.x / T; 
    int element_i  = threadIdx.x % T;
    
    // Global ID for the overall hash result
    int global_sponge_id = (blockIdx.x * (blockDim.x / T)) + sponge_idx;
    
    // Pointer to this specific sponge's state in shared memory
    uint256* state = &shared_states[sponge_idx * T];

    // 3. Initialization (Parallelized: Each thread zeros out its own element_i)
    if (element_i < T) {
        state[element_i].limbs[0] = 0; state[element_i].limbs[1] = 0;
        state[element_i].limbs[2] = 0; state[element_i].limbs[3] = 0;
    }
    __syncthreads();

    // 4. Absorption Phase
    for (int i = 0; i < total_len; i += 62) {
        // We only need one thread per sponge to handle the data loading
        if (element_i == 0) {
            int len1 = (i + 31 <= total_len) ? 31 : (total_len - i);
            uint256 m1 = bytes_to_uint256(d_input + i, len1); // Note: Adjust d_input offset for global_sponge_id if inputs vary
            
            uint256 m2 = {0,0,0,0};
            if (i + 31 < total_len) {
                int len2 = (i + 62 <= total_len) ? 31 : (total_len - (i + 31));
                m2 = bytes_to_uint256(d_input + i + 31, len2);
            }

            state[1] = mod_add(state[1], m1);
            state[2] = mod_add(state[2], m2);
        }
        
        // Ensure the additions are finished before starting the permutation
        __syncthreads();
        
        // 5. Run Permutation (Parallelized version of the function we discussed)
        run_poseidon_permutation_parallel(state);
        
        // Permutation function already has a __syncthreads at the end
    }

    // 6. Output
    if (element_i == 1) {
        d_output[global_sponge_id] = state[1];
    }
}

// Keep your existing wrapper functions (launch_poseidon_kernel and load_constants_to_gpu)

// WRAPPER FUNCTIONS
void launch_poseidon_kernel(uint8_t* d_in, uint256* d_out, size_t total_len) {
    poseidon_sponge_kernel<<<1, 3>>>(d_in, d_out, (int)total_len);
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