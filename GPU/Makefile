CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

CXXFLAGS = -std=c++11 -I$(CUDA_PATH)/include
CUDAFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

all: poseidon_app_mont poseidon_app_div

# Montgomery-optimized build
poseidon_app_mont: poseidon_gpu_mont.o main_cpu_mont.o
	$(CXX) main_cpu_mont.o poseidon_gpu_mont.o $(CUDAFLAGS) -o poseidon_app_mont

poseidon_gpu_mont.o: poseidon.cu
	$(NVCC) -DUSE_MONTGOMERY -c poseidon.cu -o poseidon_gpu_mont.o

main_cpu_mont.o: main.cpp
	$(CXX) $(CXXFLAGS) -DUSE_MONTGOMERY -c main.cpp -o main_cpu_mont.o

# Naive division build
poseidon_app_div: poseidon_gpu_div.o main_cpu_div.o
	$(CXX) main_cpu_div.o poseidon_gpu_div.o $(CUDAFLAGS) -o poseidon_app_div

poseidon_gpu_div.o: poseidon.cu
	$(NVCC) -c poseidon.cu -o poseidon_gpu_div.o

main_cpu_div.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main_cpu_div.o

clean:
	rm -f *.o poseidon_app_mont poseidon_app_div
