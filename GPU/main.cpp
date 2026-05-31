#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iomanip>
#include "poseidon.h"

// --- 6. HOST PARSER & MAIN ---

void hex_to_limbs(std::string hex, uint64_t* limbs) {
    if (hex.substr(0, 2) == "0x") hex = hex.substr(2);
    while(hex.length() < 64) hex = "0" + hex;
    for (int i = 0; i < 4; i++) {
        limbs[i] = std::stoull(hex.substr(64 - (i + 1) * 16, 16), nullptr, 16);
    }
}

void load_parameters(const std::string& filename) {
    std::vector<uint64_t> rc, mds;
    std::ifstream file(filename.c_str());
    std::string line;
    bool reading_rc = false, reading_mds = false;
    while (std::getline(file, line)) {
        if (line.find("Round constants") != std::string::npos) { reading_rc = true; reading_mds = false; continue; }
        if (line.find("MDS matrix") != std::string::npos) { reading_mds = true; reading_rc = false; continue; }
        size_t pos = 0;
        while ((pos = line.find("'0x", pos)) != std::string::npos) {
            size_t end = line.find("'", pos + 1);
            uint64_t limbs[4];
            hex_to_limbs(line.substr(pos + 1, end - pos - 1), limbs);
            if (reading_rc) for(int j=0; j<4; j++) rc.push_back(limbs[j]);
            else if (reading_mds) for(int j=0; j<4; j++) mds.push_back(limbs[j]);
            pos = end;
        }
    }
    // Safely route the copy through the nvcc-compiled wrapper
    std::cout << "--- Vector Verification ---" << std::endl;
    std::cout << "RC size: " << rc.size() << " elements." << std::endl;
    if (!rc.empty()) {
        std::cout << "First RC (limb 0): 0x" << std::hex << rc[0] << std::dec << std::endl;
        std::cout << "Second RC (limb 1): 0x" << std::hex << rc[1] << std::dec << std::endl;
        std::cout << "Third RC (limb 2): 0x" << std::hex << rc[2] << std::dec << std::endl;
        std::cout << "Fourth RC (limb 3): 0x" << std::hex << rc[3] << std::dec << std::endl;
    }

    std::cout << "MDS size: " << mds.size() << " elements." << std::endl;
    if (!mds.empty()) {
        std::cout << "First MDS (limb 0): 0x" << std::hex << mds[0] << std::dec << std::endl;
    }
    load_constants_to_gpu(rc.data(), rc.size(), mds.data(), mds.size());
    std::cout << "Constants loaded to GPU memory." << std::endl;
}

int main() {
    load_parameters("poseidon_params_n255_t3_alpha5_M128.txt");

    // --- Read ../input.txt as raw bytes ---
    std::ifstream input_file("../input.txt", std::ios::binary);
    if (!input_file) {
        std::cerr << "Error: could not open ../input.txt" << std::endl;
        return 1;
    }
    std::vector<uint8_t> h_in(
        (std::istreambuf_iterator<char>(input_file)),
        std::istreambuf_iterator<char>()
    );
    input_file.close();

    int total_len = (int)h_in.size();
    std::cout << "Read " << total_len << " bytes from ../input.txt" << std::endl;

    if (total_len == 0) {
        std::cerr << "Error: input.txt is empty." << std::endl;
        return 1;
    }

    // --- GPU setup ---
    uint8_t  *d_in;
    uint256  *d_out;
    cudaMalloc(&d_in,  total_len);
    cudaMalloc(&d_out, sizeof(uint256));
    cudaMemcpy(d_in, h_in.data(), total_len, cudaMemcpyHostToDevice);

    // Single thread: one input, one hash
    launch_poseidon_kernel(d_in, d_out, total_len);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "GPU Execution Error: " << cudaGetErrorString(err) << std::endl;
    }

    // --- Copy result and print as hex ---
    uint256 h_out;
    cudaMemcpy(&h_out, d_out, sizeof(uint256), cudaMemcpyDeviceToHost);

    // --- Print Input Bytes (h_in) ---
    /*std::cout << "Input data (hex): ";
    for (int i = 0; i < total_len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)h_in[i];
    }
    std::cout << std::dec << "\n" << std::endl;*/
    
    std::cout << "Poseidon hash: 0x";
    for (int i = 3; i >= 0; i--) {
        // Print each 64-bit limb as 16 hex digits, big-endian
        for (int b = 7; b >= 0; b--) {
            uint8_t byte = (h_out.limbs[i] >> (b * 8)) & 0xff;
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
        }
    }
    std::cout << std::dec << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}