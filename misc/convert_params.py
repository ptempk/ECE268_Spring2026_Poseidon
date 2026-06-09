import re

def to_montgomery(hex_str, modulus):
    # Convert hex string to integer
    val = int(hex_str.strip("'"), 16)
    # Montgomery R = 2^256
    r = 1 << 256
    return (val * r) % modulus

def main():
    input_file = 'poseidon_params_n255_t3_alpha5_M128.txt'
    output_file = 'montgomery_constants.txt'
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Find the modulus first
    modulus = None
    for line in lines:
        if "Modulus =" in line:
            modulus = int(line.split('=')[1].strip())
            break
    
    if modulus is None:
        print("Error: Modulus not found.")
        return

    new_lines = []
    
    # Flags to track which section we are in
    in_rc_section = False
    in_mds_section = False

    for line in lines:
        # Pass through the header information as is
        if "Round constants for GF(p):" in line:
            new_lines.append(line)
            in_rc_section = True
            continue
        if "MDS matrix:" in line:
            new_lines.append(line)
            in_mds_section = True
            in_rc_section = False
            continue
        
        # Process hex values if inside a target section
        if in_rc_section or in_mds_section:
            # Find all hex strings '0x...'
            hex_finds = re.findall(r"'0x[0-9a-fA-F]+'", line)
            if hex_finds:
                current_line = line
                for h in hex_finds:
                    mont_val = to_montgomery(h, modulus)
                    # Replace the old hex with the new Montgomery hex
                    current_line = current_line.replace(h, f"'{hex(mont_val)}'")
                new_lines.append(current_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    main()