# Define the path to the CUDA directory
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Flags
CXXFLAGS = -std=c++11 -I$(CUDA_PATH)/include
CUDAFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

all:
	$(NVCC) -c poseidon.cu -o poseidon_gpu.o
	$(CXX) $(CXXFLAGS) -c main.cpp -o main_cpu.o
	$(CXX) main_cpu.o poseidon_gpu.o $(CUDAFLAGS) -o poseidon_app

clean:
	rm -f *.o poseidon_app