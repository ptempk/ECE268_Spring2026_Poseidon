import os
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


def main():
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

if __name__ == "__main__":
    main()