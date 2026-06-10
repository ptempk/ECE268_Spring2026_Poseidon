# Poseidon Hash — CPU & GPU Implementations

ECE 268 Spring 2026 Final Project. Implements the [Poseidon](https://eprint.iacr.org/2019/458.pdf) cryptographic hash function optimized for zero-knowledge proof systems, with a Python CPU reference and a parallel CUDA GPU implementation.

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
│   ├── main.py          # Sponge construction and benchmarking
│   ├── poseidon.py      # Full and partial round functions
│   ├── helper.py        # S-box, add_round_constants, mix_layer
│   └── parameters.py    # Prime, round constants, MDS matrix
├── GPU/
│   ├── poseidon.h       # uint256 type, mod_add, mod_mul (both paths), Montgomery constants
│   ├── poseidon_device.cuh  # Device-side sponge, permutation, mix_layer (shared header)
│   ├── poseidon.cu      # Kernel entry point: poseidon_sponge_kernel
│   ├── main.cpp         # Host: parameter loading, GPU dispatch, timing, output
│   ├── Makefile         # Builds poseidon_app_mont and poseidon_app_div
│   ├── poseidon_params_n255_t3_alpha5_M128.txt  # Standard round constants / MDS
│   └── montgomery_constants.txt                  # Parameters pre-converted to Montgomery form
├── misc/
│   └── convert_params.py   # Converts standard params to Montgomery representation
├── test/
│   └── test.py             # Correctness + benchmark suite (CPU vs GPU-Mont vs GPU-Div)
└── input.txt               # One string per line; each line is hashed independently
```

## GPU Implementation Details

The CUDA kernel launches **one block per hash**, with **T = 3 threads per block**. Thread cooperation is used to parallelize the MDS matrix-vector product: each thread computes one row of `state = MDS × state`.

Two modular multiplication strategies are compiled as separate binaries:

| Binary | Strategy | Flag |
|--------|----------|------|
| `poseidon_app_mont` | Montgomery multiplication | `-DUSE_MONTGOMERY` |
| `poseidon_app_div` | Naive 512-bit binary long division | *(none)* |

Round constants and the MDS matrix are stored in `__constant__` memory. 256-bit addition uses inline PTX (`add.cc.u64` / `addc.cc.u64`) to preserve the carry chain.

## Building the GPU Code

Requires CUDA Toolkit (tested with NVCC) and a C++11-capable `g++`.

```bash
cd GPU
make            # builds both poseidon_app_mont and poseidon_app_div
make clean      # removes object files and binaries
```

To build only one target:

```bash
make poseidon_app_mont
make poseidon_app_div
```

## Running

Both implementations read from `../input.txt` (one string per line). Each line is hashed independently.

**CPU (Python):**
```bash
cd CPU
python main.py
```

**GPU (from the `GPU/` directory):**
```bash
./poseidon_app_mont   # Montgomery path
./poseidon_app_div    # Naive division path
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

## Testing and Benchmarking

`test/test.py` generates random inputs, runs all three implementations, compares hashes for correctness, and prints a timing table.

```bash
cd test
python test.py
```

Sample output:
```
 Strings  ByteLen |     CPU (ms) |  GPU-Mont (ms)  GPU-Div (ms) |  Match Mont  Match Div
----------------------------------------------------------------------------------------------------
       1      255 |         X.XX |           X.XX          X.XX |          OK         OK
     255      510 |         X.XX |           X.XX          X.XX |          OK         OK
    ...
All correctness checks passed!
```

Test cases cover `{1, 255, 510, 1020}` strings × `{255, 510, 1020}` bytes per string.

## Parameter Conversion

To regenerate `montgomery_constants.txt` from the standard parameter file:

```bash
cd misc
python convert_params.py
```

## Dependencies

- **CPU**: Python 3 (standard library only)
- **GPU**: CUDA Toolkit, g++ with C++11, a CUDA-capable GPU
- **Tests**: Python 3 standard library
