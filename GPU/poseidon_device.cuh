#pragma once

#include "poseidon.h"

// =============================================================================
// Shared device-callable Poseidon primitives.
//
// Both poseidon.cu and MerkleTree_GPU_v2.cu include this header.
// Each caller supplies its own device-side rc / mds pointers so they can use
// their own __device__ __constant__ symbols without -rdc=true.
//
// Kernels that call poseidon_mix_layer or poseidon_permutation must launch
// with exactly T threads per block; those functions use threadIdx.x and
// __syncthreads() to parallelise the MDS matrix-vector product.
// =============================================================================

// ---------------------------------------------------------------------------
// S-box: x -> x^5 mod p  (alpha = 5)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void poseidon_sbox(uint256& x)
{
    uint256 x2 = mod_mul(x, x);
    uint256 x4 = mod_mul(x2, x2);
    x = mod_mul(x4, x);
}

// ---------------------------------------------------------------------------
// Load one uint256 from a flat uint64_t array (4 limbs per entry).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint256 poseidon_load_elem(const uint64_t* src, int idx)
{
    uint256 r;
    r.limbs[0] = src[idx * 4 + 0];
    r.limbs[1] = src[idx * 4 + 1];
    r.limbs[2] = src[idx * 4 + 2];
    r.limbs[3] = src[idx * 4 + 3];
    return r;
}

// ---------------------------------------------------------------------------
// MDS matrix multiplication.
// Thread i computes row i of  state = MDS * state.
// state[] must live in shared memory; all T threads must call this together.
// ---------------------------------------------------------------------------
__device__ void poseidon_mix_layer(uint256* state, const uint64_t* mds)
{
    int i = threadIdx.x;
    uint256 res;
    res.limbs[0] = 0; res.limbs[1] = 0;
    res.limbs[2] = 0; res.limbs[3] = 0;

    if (i < T) {
        for (int j = 0; j < T; j++)
            res = mod_add(res, mod_mul(state[j], poseidon_load_elem(mds, i * T + j)));
    }
    __syncthreads();
    if (i < T) state[i] = res;
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Full Poseidon permutation: RF/2 full + RP partial + RF/2 full rounds.
// state[] must be in shared memory; all T threads must call this together.
// rc  — pointer to round-constant array  (TOTAL_RC * 4 uint64_t values)
// mds — pointer to MDS matrix array      (T * T * 4 uint64_t values)
// ---------------------------------------------------------------------------
__device__ void poseidon_permutation(uint256*        state,
                                      const uint64_t* rc,
                                      const uint64_t* mds)
{
    int idx = 0;

    // First RF/2 full rounds
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < T; i++) {
            state[i] = mod_add(state[i], poseidon_load_elem(rc, idx++));
            poseidon_sbox(state[i]);
        }
        poseidon_mix_layer(state, mds);
    }
    // RP partial rounds (S-box on state[0] only)
    for (int r = 0; r < RP; r++) {
        for (int i = 0; i < T; i++)
            state[i] = mod_add(state[i], poseidon_load_elem(rc, idx++));
        poseidon_sbox(state[0]);
        poseidon_mix_layer(state, mds);
    }
    // Last RF/2 full rounds
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < T; i++) {
            state[i] = mod_add(state[i], poseidon_load_elem(rc, idx++));
            poseidon_sbox(state[i]);
        }
        poseidon_mix_layer(state, mds);
    }
}

// ---------------------------------------------------------------------------
// Complete Poseidon sponge: absorb `len` raw bytes, write 32 bytes to `out`.
//
// Must be called by exactly T threads in the same block.
// rc / mds are device-side pointers (typically into __constant__ memory).
//
// Absorption: 62 bytes per step (two 31-byte field elements).
// Output: state[1] after the last permutation, converted from Montgomery form
//         and serialised as 4 little-endian uint64 limbs = 32 bytes.
// ---------------------------------------------------------------------------
__device__ void poseidon_hash_device(const uint8_t*  input,
                                      int             len,
                                      uint8_t*        out,
                                      const uint64_t* rc,
                                      const uint64_t* mds)
{
    __shared__ uint256 state[T];
    int tid = threadIdx.x;

    if (tid < T) {
        state[tid].limbs[0] = 0; state[tid].limbs[1] = 0;
        state[tid].limbs[2] = 0; state[tid].limbs[3] = 0;
    }
    __syncthreads();

    // Absorb in 62-byte chunks (two 31-byte field elements per step)
    for (int k = 0; k < len; k += 62) {
        if (tid == 0) {
            int len1 = (k + 31 <= len) ? 31 : (len - k);
            uint256 m1 = bytes_to_uint256(input + k, len1);
            uint256 m2 = {0, 0, 0, 0};
            if (k + 31 < len) {
                int len2 = (k + 62 <= len) ? 31 : (len - (k + 31));
                m2 = bytes_to_uint256(input + k + 31, len2);
            }
            state[1] = mod_add(state[1], to_montgomery(m1));
            state[2] = mod_add(state[2], to_montgomery(m2));
        }
        __syncthreads();
        poseidon_permutation(state, rc, mds);
        // poseidon_permutation ends with __syncthreads() inside poseidon_mix_layer
    }

    // Squeeze: thread 0 serialises state[1] to 32 little-endian bytes
    if (tid == 0) {
        uint256 result = from_montgomery(state[1]);
        for (int b = 0; b < 4; b++) {
            uint64_t limb = result.limbs[b];
            for (int j = 0; j < 8; j++)
                out[b * 8 + j] = static_cast<uint8_t>((limb >> (j * 8)) & 0xFF);
        }
    }
}
