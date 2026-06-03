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
    load_constants_to_gpu(rc.data(), rc.size(), mds.data(), mds.size());
    
}

int main() {
    load_parameters("montgomery_constants.txt");//("poseidon_params_n255_t3_alpha5_M128.txt");

    std::ifstream input_file("../input.txt");
    if (!input_file) {
        std::cerr << "Error: could not open ../input.txt" << std::endl;
        return 1;
    }

    std::vector<uint8_t> h_all_data;
    std::vector<int> h_offsets;
    std::vector<int> h_lengths;
    std::string line;

    while (std::getline(input_file, line)) {
        if (line.empty()) continue; 
        
        h_offsets.push_back((int)h_all_data.size());
        h_lengths.push_back((int)line.size());
        
        // Append line bytes to the flat vector
        for (char c : line) {
            h_all_data.push_back(static_cast<uint8_t>(c));
        }
    }
    input_file.close();

    int num_hashes = (int)h_offsets.size();
    if (num_hashes == 0) {
        std::cerr << "Error: No data found in input.txt" << std::endl;
        return 1;
    }

    // --- GPU setup ---
    uint8_t *d_in;
    int *d_offsets, *d_lengths;
    uint256 *d_out;

    // --- Performance Measurement Setup ---
    cudaEvent_t start, stop, proctimestart, proctimestop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&proctimestart);
    cudaEventCreate(&proctimestop);

    // Record the start event
    cudaEventRecord(start);

    cudaMalloc(&d_in, h_all_data.size());
    cudaMalloc(&d_offsets, num_hashes * sizeof(int));
    cudaMalloc(&d_lengths, num_hashes * sizeof(int));
    cudaMalloc(&d_out, num_hashes * sizeof(uint256));

    cudaMemcpy(d_in, h_all_data.data(), h_all_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), num_hashes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths.data(), num_hashes * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventRecord(proctimestart);
    // Launch with one block per hash
    // We pass num_hashes to set the grid size
    launch_poseidon_kernel(d_in, d_offsets, d_lengths, d_out, num_hashes);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "GPU Execution Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(proctimestop);
    cudaEventSynchronize(proctimestop);
    // --- Copy results back ---
    std::vector<uint256> h_results(num_hashes);
    cudaMemcpy(h_results.data(), d_out, num_hashes * sizeof(uint256), cudaMemcpyDeviceToHost);
    
    // Record the stop event
    cudaEventRecord(stop);
    
    // Wait for the GPU to finish all work before calculating time
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    float procmilliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventElapsedTime(&procmilliseconds, proctimestart, proctimestop);

    // --- Cleanup Performance Events ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(proctimestart);
    cudaEventDestroy(proctimestop);

    // --- Output Results ---
    std::cout << "\n------------------------------------" << std::endl;
    std::cout << "Processed " << num_hashes << " hashes." << std::endl;
    std::cout << "Total GPU Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "GPU operation Time: " << procmilliseconds << " ms" << std::endl;
    if (num_hashes > 0) {
        std::cout << "Average time per hash: " << (milliseconds / num_hashes) << " ms" << std::endl;
    }
    std::cout << "------------------------------------\n" << std::endl;

    // Print all hashes
    for (int n = 0; n < num_hashes; n++) {
        std::cout << "Line " << n << " hash: 0x";
        for (int i = 3; i >= 0; i--) {
            for (int b = 7; b >= 0; b--) {
                uint8_t byte = (h_results[n].limbs[i] >> (b * 8)) & 0xff;
                std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
            }
        }
        std::cout << std::dec << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_out);
    return 0;
}