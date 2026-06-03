import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
from poseidon import poseidon_permutation
from parameters import prime_255, matrix_n255_t3, round_constants_n255_t3

def read_input_file(filename="input.txt"):
    """Reads the entire content of a text file as bytes."""
    if not os.path.exists("../" + filename):
        with open(filename, "wb") as f:
            # Writing 100 bytes to ensure we have more than one "t-1" batch
            f.write(b"This is a longer string to demonstrate how Poseidon absorbs multiple chunks of data step-by-step.")
    
    with open("../" + filename, "rb") as f:
        return f.read()

def get_chunks(data, chunk_size=31):
    """Splits bytes into 31-byte integers."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(int.from_bytes(chunk, 'little'))
    return chunks


'''def main():
    for i in range(3):
        print(f"RC[{i}] = {round_constants_n255_t3[i]}")
    data = read_input_file()
    all_chunks = get_chunks(data)
    
    # Initialize state [Capacity, Rate1, Rate2]
    state = [0, 0, 0]
    t = 3
    rate = t - 1 # For t=3, we absorb 2 chunks at a time

    print(f"Total chunks to process: {len(all_chunks)}")

    # Absorb Phase
    for i in range(0, len(all_chunks), rate):  
        print(f"\n--- Processing Batch starting at chunk {i} ---")
        
        # Get the next two chunks (padding with 0 if we reach the end)
        for j in range(rate):
            chunk_idx = i + j
            if chunk_idx < len(all_chunks):
                val = all_chunks[chunk_idx]
                # In a sponge, we add the new data to the existing rate state
                state[j + 1] = (state[j + 1] + val) % prime_255
            else:
                # Padding logic if needed (e.g., adding a 1 bit or just 0s)
                pass
        
        # After absorbing a batch, run the permutation
        state = poseidon_permutation(state) 
        
        print(f"State after permutation: {[hex(s)[:10] + '...' for s in state]}")

    print("\nFinal hash (squeezed state[1]):", hex(state[1]))
'''

def main():
    # Load parameters once
    # (Assuming round_constants_n255_t3 and prime_255 are already loaded globally)
    
    try:
        with open("../input.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: could not open ../input.txt")
        return
    
    # Start the timer for the entire hashing process
    process_start_time = time.perf_counter()

    t = 3
    rate = t - 1  # For t=3, we absorb 2 chunks at a time

    num_hashes = 0
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        num_hashes += 1
            
        #print(f"\n========== Processing Line {idx} ==========")
        
        # 1. Convert line to bytes then to field elements (chunks)
        # This matches your bytes_to_uint256 logic on the GPU
        line_bytes = line.encode('utf-8')
        all_chunks = get_chunks(line_bytes)
        
        # 2. IMPORTANT: Reset state for EVERY NEW LINE
        # [Capacity, Rate1, Rate2]
        state = [0, 0, 0]
        
        #print(f"Line length: {len(line)} bytes, Chunks: {len(all_chunks)}")

        # 3. Absorb Phase
        for i in range(0, len(all_chunks), rate):
            # Add new data to the rate portion of the state
            for j in range(rate):
                chunk_idx = i + j
                if chunk_idx < len(all_chunks):
                    val = all_chunks[chunk_idx]
                    state[j + 1] = (state[j + 1] + val) % prime_255
            
            # Run the permutation after each absorption step
            state = poseidon_permutation(state)
        
        # 4. Output Result (matches d_output[hash_idx] = state[1])
        print(f"Line {idx} Final hash: {hex(state[1])}")
    
    # 5. Report time
    process_end_time = time.perf_counter()
    total_duration_ms = (process_end_time - process_start_time) * 1000
    print("\n" + "-"*40)
    print(f"Processed {num_hashes} hashes.")
    print(f"Total Python Execution Time: {total_duration_ms:.3f} ms")
    if num_hashes > 0:
        print(f"Average time per hash: {total_duration_ms / num_hashes:.3f} ms")
    print("-"*40)

if __name__ == "__main__":
    main()