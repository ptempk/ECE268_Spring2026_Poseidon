from helper import add_round_constants, mix_layer, s_box, FULL_ROUNDS, PARTIAL_ROUNDS, T
from parameters import prime_255


def poseidon_full_round(state, current_round):
    # --- Full Round ---
        # Step A: Add Round Constants
        state = add_round_constants(state, current_round)
        
        # Step B: S-Box (Full means apply to all elements)
        #print("\nHash before mul (squeezed state[1]):", hex(state[1]))
        #state[1] = pow(state[1], 2, prime_255)
        #print("\nHash after mul (squeezed state[1]):", hex(state[1]))
        for i in range(T):
            state[i] = s_box(state[i])
            
        # Step C: MixLayer (We will implement this next)
        state = mix_layer(state)
        
        #print(f"Completed Full Round {current_round}")
        return state
    
def poseidon_partial_round(state, current_round):
    # --- Partial Round ---
    # Step A: Add Round Constants
    state = add_round_constants(state, current_round)
    
    # Step B: S-Box (Partial means apply to only one state)
    state[0] = s_box(state[0])
        
    # Step C: MixLayer (We will implement this next)
    state = mix_layer(state)

    #print(f"Completed Partial Round {current_round}")
    return state

def poseidon_permutation(state):
    current_round = 0
    #print("  [Permutation] Running rounds on current state...")
    for i in range(FULL_ROUNDS//2):
        state = poseidon_full_round(state, current_round)
        current_round += 1  
    for i in range(PARTIAL_ROUNDS):
        state = poseidon_partial_round(state, current_round)
        current_round += 1
    for i in range(FULL_ROUNDS//2):
        state = poseidon_full_round(state, current_round)
        current_round += 1
    # Logic for Ark, S-box, and Mix will go here
    return state

'''def poseidon_permutation(state):
    
    current_round = 0
    print("  [Permutation] Running rounds on current state...")
    state = poseidon_full_round(state, current_round)
    return state'''