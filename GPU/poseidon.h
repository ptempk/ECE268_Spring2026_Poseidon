#ifndef POSEIDON_H
#define POSEIDON_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define T 3
#define RF 8
#define RP 56
#define TOTAL_RC ((RF + RP) * T)

#define MAX_PARALLEL_SPONGES 10

struct uint256 {
    uint64_t limbs[4];
};

// Montgomery Constant: -1/P mod 2^64 for BLS12-381 scalar field
__device__ __constant__ uint64_t P_INV = 0xfffffffeffffffff;

__device__ __constant__ uint64_t P_LIMBS[4] = {
    0xffffffff00000001, // Limb 0
    0x53bda402fffe5bfe, // Limb 1
    0x3339d80809a1d805, // Limb 2
    0x73eda753299d7d48  // Limb 3
};

__device__ __constant__ uint64_t R2_MOD_P[4] = {
    0xc999e990f3f29c6d, // Limb 0
    0x2b6cedcb87925c23, // Limb 1
    0x05d314967254398f, // Limb 2
    0x0748d9d99f59ff11  // Limb 3
};

// Wrapper declarations
void launch_poseidon_kernel(uint8_t* d_in, int* d_offsets, int* d_lengths, uint256* d_out, int num_hashes);
void load_constants_to_gpu(const uint64_t* h_rc, size_t rc_bytes, const uint64_t* h_mds, size_t mds_bytes);

// Function declarations so main.cu knows they exist
__device__ __forceinline__ uint256 mod_add(uint256 a, uint256 b);
__device__ __forceinline__ uint256 mod_mul(uint256 a, uint256 b);
__device__ __forceinline__ uint256 bytes_to_uint256(const uint8_t* b, int len);

__device__ __forceinline__ uint256 bytes_to_uint256(const uint8_t* b, int len) {
    uint256 res = {0, 0, 0, 0};
    // Pack up to 31 bytes to stay within BLS12-381 scalar field safely
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            int byte_idx = i * 8 + j;
            if (byte_idx < 31 && byte_idx < len) {
                res.limbs[i] |= ((uint64_t)b[byte_idx]) << (j * 8);
            }
        }
    }
    return res;
}

__device__ __forceinline__ uint256 mod_add(uint256 a, uint256 b) {
    uint256 res;

    //Perform 256-bit addition in ONE block to protect the carry chain
    unsigned long long carry;
    asm(
        "add.cc.u64    %0, %5, %9; \n\t"
        "addc.cc.u64   %1, %6, %10;\n\t"
        "addc.cc.u64   %2, %7, %11;\n\t"
        "addc.cc.u64   %3, %8, %12;\n\t"
        "addc.u64      %4, 0, 0;   \n\t"
        : "=l"(res.limbs[0]), "=l"(res.limbs[1]), "=l"(res.limbs[2]), "=l"(res.limbs[3]), "=l"(carry)
        : "l"(a.limbs[0]), "l"(a.limbs[1]), "l"(a.limbs[2]), "l"(a.limbs[3]),
          "l"(b.limbs[0]), "l"(b.limbs[1]), "l"(b.limbs[2]), "l"(b.limbs[3])
    );

    //Check if (res >= P) or if we had a carry
    bool overflow = (carry != 0);
    if (!overflow) {
        for (int i = 3; i >= 0; i--) {
            if (res.limbs[i] > P_LIMBS[i]) {
                overflow = true;
                break;
            }
            if (res.limbs[i] < P_LIMBS[i]) break;
            if (i == 0) overflow = true; // res == P
        }
    }

    //Conditional Subtraction in ONE block
    if (overflow) {
        asm(
            "sub.cc.u64  %0, %0, %4;\n\t"
            "subc.cc.u64 %1, %1, %5;\n\t"
            "subc.cc.u64 %2, %2, %6;\n\t"
            "subc.u64    %3, %3, %7;\n\t"
            : "+l"(res.limbs[0]), "+l"(res.limbs[1]), "+l"(res.limbs[2]), "+l"(res.limbs[3])
            : "l"(P_LIMBS[0]), "l"(P_LIMBS[1]), "l"(P_LIMBS[2]), "l"(P_LIMBS[3])
        );
    }

    return res;
}

__device__ __forceinline__ uint256 to_montgomery(uint256 x) {
    uint256 r2;
    for(int i=0; i<4; i++) r2.limbs[i] = R2_MOD_P[i];
    return mod_mul(x, r2);
}

__device__ __forceinline__ uint256 from_montgomery(uint256 x_bar) {
    uint256 one = {1, 0, 0, 0};
    return mod_mul(x_bar, one);
}

__device__ __forceinline__ uint256 montgomery_reduce(uint64_t* t) {
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * P_INV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * P_LIMBS[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        
        // Propagate the carry through the remaining upper limbs
        for (int j = i + 4; j < 8; j++) {
            unsigned __int128 sum = (unsigned __int128)t[j] + carry;
            t[j] = (uint64_t)sum;
            carry = (uint64_t)(sum >> 64);
        }
    }

    // The result is now in the upper 4 limbs (t[4] to t[7])
    uint256 res;
    for (int i = 0; i < 4; i++) res.limbs[i] = t[i + 4];

    // Final conditional subtraction: if res >= P, res -= P
    bool geq = true;
    for (int i = 3; i >= 0; i--) {
        if (res.limbs[i] < P_LIMBS[i]) { geq = false; break; }
        if (res.limbs[i] > P_LIMBS[i]) break;
    }
    if (geq) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t prev = res.limbs[i];
            res.limbs[i] = prev - P_LIMBS[i] - borrow;
            borrow = (prev < P_LIMBS[i] || (prev == P_LIMBS[i] && borrow)) ? 1 : 0;
        }
    }
    return res;
}

__device__ __forceinline__ uint256 mod_mul(uint256 a, uint256 b) {
    uint64_t t[8] = {0};

    //256x256 multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry     = (uint64_t)(prod >> 64);
        }
        // Accumulate carry into the next upper limb
        unsigned __int128 sum = (unsigned __int128)t[i + 4] + carry;
        t[i + 4] = (uint64_t)sum;
        if ((uint64_t)(sum >> 64) && (i + 5 < 8))
            t[i + 5] += (uint64_t)(sum >> 64);
    }

    return montgomery_reduce(t);
}

#endif