# Poseidon Hash — CPU & GPU Implementations

ECE 268 Spring 2026 Final Project. Implements the [Poseidon](https://eprint.iacr.org/2019/458.pdf) cryptographic hash function optimized for zero-knowledge proof systems, with a Python CPU reference and a parallel CUDA GPU implementation.

> **This repository is used as a Git submodule** inside a larger Merkle tree project. The GPU device interface in `GPU/poseidon_device.cuh` is designed to be included directly by the parent repo's Merkle tree CUDA kernel (`MerkleTree_GPU_v2.cu`).

## Overview

Poseidon is a ZK-friendly hash function that operates over prime fields. This project targets the **BLS12-381 scalar field** (255-bit prime `p = 0x73EDA753...00000001`) with parameters:

| Parameter | Value |
|-----------|-------|
| State width (`t`) | 3 |
| S-box exponent (`α`) | 5 |
| Full rounds (`RF`) | 8 |
| Partial rounds (`RP`) | 56 |
| Security level | 128-bit |

The sponge absorbs input in **62-byte chunks** (two 31-byte field elements per step) and outputs `state[1]` after the final permutation.

## Repository Structure

```
.
├── CPU/
│   ├── main.py          # Sponge construction and timing
│   ├── poseidon.py      # Full and partial round functions
│   ├── helper.py        # S-box, add_round_constants, mix_layer
│   └── parameters.py    # Prime, round constants, MDS matrix
├── GPU/
│   ├── poseidon.h           # uint256 type, mod_add, mod_mul (Montgomery), field constants
│   ├── poseidon_device.cuh  # Device-side sponge + permutation (included by Merkle tree kernel)
│   ├── poseidon.cu          # Standalone kernel: poseidon_sponge_kernel
│   ├── main.cpp             # Host: parameter loading, GPU dispatch, timing, output
│   ├── Makefile             # Builds poseidon_app
│   └── montgomery_constants.txt  # Round constants and MDS matrix in Montgomery form
├── misc/
│   ├── convert_params.py              # Converts standard params to Montgomery form
│   ├── poseidon_params_n255_t3_alpha5_M128.txt  # Standard round constants / MDS
│   └── montgomery_constants.txt       # Pre-converted Montgomery parameters
└── input.txt               # One string per line; each line is hashed independently
```

## Submodule Integration

The key integration file is `GPU/poseidon_device.cuh`. It exposes two device-callable interfaces:

- **`poseidon_hash_device(input, len, out, rc, mds)`** — complete sponge: absorbs `len` raw bytes and writes a 32-byte little-endian hash to `out`. This is what the parent Merkle tree kernel calls.
- **`poseidon_permutation(state, rc, mds)`** — raw permutation on a shared-memory state array, for callers that manage absorption themselves.

Each caller (this repo's `poseidon.cu` and the parent's `MerkleTree_GPU_v2.cu`) supplies its own `__device__ __constant__` symbols for `rc` and `mds`, avoiding the need for `-rdc=true` separate compilation.

Any kernel using these functions **must launch with exactly T = 3 threads per block**; `poseidon_mix_layer` uses `threadIdx.x` and `__syncthreads()` to parallelize the MDS matrix-vector product across threads.

## GPU Implementation Details

The standalone kernel (`poseidon_sponge_kernel`) launches **one block per hash** with **T = 3 threads per block**:

- Thread 0 handles byte absorption (converting raw bytes to field elements)
- All 3 threads cooperate on the MDS matrix multiply — each thread computes one row of `state = MDS × state`
- Round constants and the MDS matrix are stored in `__constant__` memory
- All arithmetic uses **Montgomery multiplication** for efficiency
- 256-bit addition uses inline PTX (`add.cc.u64` / `addc.cc.u64`) to preserve the carry chain

## Building the GPU Standalone Binary

Requires CUDA Toolkit and a C++11-capable `g++`.

```bash
cd GPU
make        # builds poseidon_app
make clean  # removes object files and binary
```

## Running

Both implementations read from `../input.txt` (one string per line) and hash each line independently.

**CPU (Python):**
```bash
cd CPU
python main.py
```

**GPU:**
```bash
cd GPU
./poseidon_app
```

Output format:
```
Line 0 Final hash: 0x<64 hex chars>
...
------------------------------------
Processed N hashes.
Total GPU Execution Time: X.XX ms
GPU operation Time: Y.YY ms
Average time per hash: Z.ZZ ms
------------------------------------
```

## Parameter Conversion

To regenerate `montgomery_constants.txt` from the standard parameter file:

```bash
cd misc
python convert_params.py
```

## Dependencies

- **CPU**: Python 3 (standard library only)
- **GPU**: CUDA Toolkit, g++ with C++11, a CUDA-capable GPU
