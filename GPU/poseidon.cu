#include "poseidon.h"
#include "poseidon_device.cuh"
#include <stdio.h>

// Private constant memory for this translation unit
__device__ __constant__ uint64_t C_CONSTANTS[TOTAL_RC * 4];
__device__ __constant__ uint64_t M_MATRIX[T * T * 4];

void load_constants_to_gpu(const uint64_t* h_rc, size_t rc_count,
                            const uint64_t* h_mds, size_t mds_count)
{
    cudaError_t err;

    err = cudaMemcpyToSymbol(C_CONSTANTS, h_rc, rc_count * sizeof(uint64_t));
    if (err != cudaSuccess)
        printf("RC Copy Failed: %s (Expected %zu bytes)\n",
               cudaGetErrorString(err), rc_count * sizeof(uint64_t));

    err = cudaMemcpyToSymbol(M_MATRIX, h_mds, mds_count * sizeof(uint64_t));
    if (err != cudaSuccess)
        printf("MDS Copy Failed: %s (Expected %zu bytes)\n",
               cudaGetErrorString(err), mds_count * sizeof(uint64_t));
}

// One block per hash, T=3 threads per block.
// Thread 0 absorbs; all threads cooperate on poseidon_permutation via
// poseidon_mix_layer (which parallelises the MDS matrix-vector product).
__global__ void poseidon_sponge_kernel(uint8_t*  d_input,
                                        int*      d_offsets,
                                        int*      d_lengths,
                                        uint256*  d_output)
{
    __shared__ uint256 state[T];

    int hash_idx = blockIdx.x;
    int tid      = threadIdx.x;

    uint8_t* my_input = d_input + d_offsets[hash_idx];
    int      my_len   = d_lengths[hash_idx];

    if (tid < T) {
        state[tid].limbs[0] = 0; state[tid].limbs[1] = 0;
        state[tid].limbs[2] = 0; state[tid].limbs[3] = 0;
    }
    __syncthreads();

    // Absorption phase — thread 0 converts bytes to field elements
    for (int i = 0; i < my_len; i += 62) {
        if (tid == 0) {
            int len1 = (i + 31 <= my_len) ? 31 : (my_len - i);
            uint256 m1 = bytes_to_uint256(my_input + i, len1);
            uint256 m2 = {0, 0, 0, 0};
            if (i + 31 < my_len) {
                int len2 = (i + 62 <= my_len) ? 31 : (my_len - (i + 31));
                m2 = bytes_to_uint256(my_input + i + 31, len2);
            }
            state[1] = mod_add(state[1], to_montgomery(m1));
            state[2] = mod_add(state[2], to_montgomery(m2));
        }
        __syncthreads();
        poseidon_permutation(state, C_CONSTANTS, M_MATRIX);
    }

    // Output state[1] as a uint256 (caller serialises as needed)
    if (tid == 1) {
        state[1] = from_montgomery(state[1]);
        d_output[hash_idx] = state[1];
    }
}

void launch_poseidon_kernel(uint8_t*  d_in,
                             int*      d_offsets,
                             int*      d_lengths,
                             uint256*  d_out,
                             int       num_hashes)
{
    poseidon_sponge_kernel<<<num_hashes, T>>>(d_in, d_offsets, d_lengths, d_out);
}
