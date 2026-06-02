#ifndef POSEIDON_H
#define POSEIDON_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- 1. PARAMETERS & TYPES ---
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

// Wrapper declarations
void launch_poseidon_kernel(uint8_t* d_in, uint256* d_out, size_t total_len);
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

    // 1. Perform 256-bit addition in ONE block to protect the carry chain
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

    // 2. Check if (res >= P) or if we had a carry
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

    // 3. Conditional Subtraction in ONE block
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

/*__device__ __forceinline__ uint256 mod_mul(uint256 a, uint256 b) {
    uint64_t t[8] = {0}; // 512-bit accumulator for (a * b)

    // --- PHASE 1: 256x256 -> 512 bit Multiplication (Schoolbook) ---
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i + 4] = carry;
    }

    // --- PHASE 2: Montgomery Reduction (CIOS Method) ---
    // this reduces t mod P by zeroing out the lower 4 limbs
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * P_INV; // the multiplier to make t[i] zero
        uint64_t carry = 0;
        
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * P_LIMBS[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod; // this results in t[i] becoming 0 in the first iteration
            carry = (uint64_t)(prod >> 64);
        }
        
        // Propagate carry through the upper limbs of t
        int k = i + 4;
        while (carry > 0 && k < 8) {
            unsigned __int128 sum = (unsigned __int128)t[k] + carry;
            t[k] = (uint64_t)sum;
            carry = (uint64_t)(sum >> 64);
            k++;
        }
    }

    // --- PHASE 3: Final Selection ---
    // the result is stored in the upper 4 limbs of t
    uint256 res;
    res.limbs[0] = t[4];
    res.limbs[1] = t[5];
    res.limbs[2] = t[6];
    res.limbs[3] = t[7];

    // Final check: if res >= P, return res - P. 
    bool geq = true;
    for(int i = 3; i >= 0; i--) {
        if(res.limbs[i] < P_LIMBS[i]) { geq = false; break; }
        if(res.limbs[i] > P_LIMBS[i]) { geq = true; break; }
    }

    if (geq) {
        unsigned char c = 0;
        for(int i = 0; i < 4; i++) {
            // Manual borrow-subtraction logic
            uint64_t old = res.limbs[i];
            res.limbs[i] -= (P_LIMBS[i] + c);
            c = (old < (P_LIMBS[i] + c)) ? 1 : 0;
        }
    }

    return res;
}*/


__device__ __forceinline__ uint256 slow_plain_reduction(uint64_t* t) {
    for (int i = 256; i >= 0; i--) {
        uint64_t shifted_P[8] = {0};
        int limb_shift = i / 64;
        int bit_shift  = i % 64;

        if (bit_shift == 0) {
            for (int j = 0; j < 4; j++) {
                if (j + limb_shift < 8) shifted_P[j + limb_shift] = P_LIMBS[j];
            }
        } else {
            uint64_t carry = 0;
            for (int j = 0; j < 4; j++) {
                uint64_t lo = (P_LIMBS[j] << bit_shift) | carry;
                carry = P_LIMBS[j] >> (64 - bit_shift);
                if (j + limb_shift < 8)
                    shifted_P[j + limb_shift] = lo;
            }
            if (4 + limb_shift < 8)
                shifted_P[4 + limb_shift] = carry;
        }

        bool geq = true;
        for (int j = 7; j >= 0; j--) {
            if (t[j] < shifted_P[j]) { geq = false; break; }
            if (t[j] > shifted_P[j]) break;
        }

        if (geq) {
            uint64_t borrow = 0;
            for (int j = 0; j < 8; j++) {
                uint64_t t_val = t[j];
                uint64_t s     = shifted_P[j];
                uint64_t tmp   = t_val - borrow;
                borrow  = (t_val < borrow) ? 1 : 0;
                borrow += (tmp   < s)      ? 1 : 0;
                t[j] = tmp - s;
            }
        }
    }

    uint256 res;
    for (int i = 0; i < 4; i++) res.limbs[i] = t[i];
    return res;
}

__device__ __forceinline__ uint256 mod_mul(uint256 a, uint256 b) {
    uint64_t t[8] = {0};

    // Schoolbook 256x256 -> 512-bit multiplication
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

    return slow_plain_reduction(t);
}

#endif