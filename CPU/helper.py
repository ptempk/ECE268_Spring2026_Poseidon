from parameters import prime_255, matrix_n255_t3, round_constants_n255_t3

ALPHA = 5
T = 3
FULL_ROUNDS = 8
PARTIAL_ROUNDS = 56

def s_box(element):
    """Applies the power function x^5 % prime."""
    return pow(element, ALPHA, prime_255)

def add_round_constants(state, round_idx):
    """Adds T constants from the parameter list to the state."""
    for i in range(T):
        # The '16' handles strings like '0xabc...' from parameters.py
        constant = int(round_constants_n255_t3[round_idx * T + i], 16)
        state[i] = (state[i] + constant) % prime_255
    return state

def mix_layer(state):
    """Matrix multiplication: state = MDS_matrix * state."""
    new_state = [0] * T
    for i in range(T):
        temp_sum = 0
        for j in range(T):
            matrix_val = int(matrix_n255_t3[i][j], 16)
            temp_sum += matrix_val * state[j]
        new_state[i] = temp_sum % prime_255
    return new_state