import os
import sys
import subprocess
import time
import re
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
INPUT_FILE   = os.path.join(ROOT_DIR, "input.txt")
GPU_DIR      = os.path.join(ROOT_DIR, "GPU")
CPU_DIR      = os.path.join(ROOT_DIR, "CPU")
GPU_MONT     = os.path.join(GPU_DIR, "poseidon_app_mont")
GPU_DIV      = os.path.join(GPU_DIR, "poseidon_app_div")
CPU_MAIN     = os.path.join(CPU_DIR, "main.py")

# ---------------------------------------------------------------------------
# Generate a hex bytestring of exactly `byte_len` hex characters (each pair
# represents one byte, so the string encodes byte_len/2 bytes — but we match
# the GPU which treats each character as one byte of ASCII input).
# ---------------------------------------------------------------------------
def generate_input(num_strings, string_byte_len):
    lines = []
    for _ in range(num_strings):
        # Each "byte" is one hex character; generate string_byte_len hex chars.
        s = ''.join(random.choices('0123456789abcdef', k=string_byte_len))
        lines.append(s)
    with open(INPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# Parse hashes printed by the GPU binary.
# Lines look like: "Line 0 Final hash: 0x<64 hex chars>"
# ---------------------------------------------------------------------------
def parse_gpu_hashes(output: str) -> dict:
    hashes = {}
    for line in output.splitlines():
        m = re.match(r'Line\s+(\d+)\s+Final hash:\s+0x([0-9a-fA-F]+)', line)
        if m:
            hashes[int(m.group(1))] = m.group(2).lower()
    return hashes

# Parse timing from GPU output
def parse_gpu_time(output: str) -> float:
    m = re.search(r'Total GPU Execution Time:\s+([\d.]+)\s*ms', output)
    return float(m.group(1)) if m else float('nan')

# ---------------------------------------------------------------------------
# Parse hashes printed by the CPU (Python) script.
# Lines look like: "Line 0 Final hash: 0x<hex>"
# ---------------------------------------------------------------------------
def parse_cpu_hashes(output: str) -> dict:
    hashes = {}
    for line in output.splitlines():
        m = re.match(r'Line\s+(\d+)\s+Final hash:\s+0x([0-9a-fA-F]+)', line)
        if m:
            hashes[int(m.group(1))] = m.group(2).lower()
    return hashes

def parse_cpu_time(output: str) -> float:
    m = re.search(r'Total Python Execution Time:\s+([\d.]+)\s*ms', output)
    return float(m.group(1)) if m else float('nan')

# ---------------------------------------------------------------------------
# Run a subprocess and return (stdout, elapsed_wall_ms)
# ---------------------------------------------------------------------------
def run(cmd, cwd=None):
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    elapsed = (time.perf_counter() - t0) * 1000
    if result.returncode != 0:
        print(f"  [ERROR] {' '.join(cmd)} exited {result.returncode}")
        print(result.stderr[:500])
    return result.stdout, result.stderr, elapsed

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def main():
    random.seed(42)

    num_strings_cases  = [1, 255, 510, 1020]
    string_byte_cases  = [255, 510, 1020]

    print(f"{'Strings':>8} {'ByteLen':>8} | {'CPU (ms)':>12} | {'GPU-Mont (ms)':>14} {'GPU-Div (ms)':>13} | {'Match Mont':>11} {'Match Div':>10}")
    print("-" * 100)

    all_pass = True

    for n_str in num_strings_cases:
        for byte_len in string_byte_cases:
            generate_input(n_str, byte_len)

            # --- CPU ---
            cpu_out, cpu_err, _ = run([sys.executable, CPU_MAIN], cwd=CPU_DIR)
            cpu_ms     = parse_cpu_time(cpu_out)
            cpu_hashes = parse_cpu_hashes(cpu_out)

            # --- GPU Montgomery ---
            mont_out, mont_err, _ = run([GPU_MONT], cwd=GPU_DIR)
            mont_ms     = parse_gpu_time(mont_out)
            mont_hashes = parse_gpu_hashes(mont_out)

            # --- GPU Naive division ---
            div_out, div_err, _ = run([GPU_DIV], cwd=GPU_DIR)
            div_ms     = parse_gpu_time(div_out)
            div_hashes = parse_gpu_hashes(div_out)

            # --- Correctness check ---
            match_mont = True
            match_div  = True
            mismatches_mont = []
            mismatches_div  = []

            for idx in range(n_str):
                cpu_h  = cpu_hashes.get(idx, "MISSING")
                mont_h = mont_hashes.get(idx, "MISSING")
                div_h  = div_hashes.get(idx, "MISSING")

                # CPU emits shorter hex (no leading zeros); pad to 64 chars.
                cpu_h_padded  = cpu_h.lstrip('0x').zfill(64) if cpu_h != "MISSING" else "MISSING"
                mont_h_padded = mont_h.zfill(64) if mont_h != "MISSING" else "MISSING"
                div_h_padded  = div_h.zfill(64) if div_h != "MISSING" else "MISSING"

                if cpu_h_padded != mont_h_padded:
                    match_mont = False
                    mismatches_mont.append((idx, cpu_h_padded, mont_h_padded))
                if cpu_h_padded != div_h_padded:
                    match_div = False
                    mismatches_div.append((idx, cpu_h_padded, div_h_padded))

            if not match_mont or not match_div:
                all_pass = False

            mont_ok = "OK" if match_mont else f"FAIL({len(mismatches_mont)})"
            div_ok  = "OK" if match_div  else f"FAIL({len(mismatches_div)})"

            print(f"{n_str:>8} {byte_len:>8} | {cpu_ms:>12.2f} | {mont_ms:>14.2f} {div_ms:>13.2f} | {mont_ok:>11} {div_ok:>10}")

            # Print details of first mismatch if any
            if mismatches_mont:
                idx, c, g = mismatches_mont[0]
                print(f"           Mont mismatch line {idx}: CPU={c}  GPU={g}")
            if mismatches_div:
                idx, c, g = mismatches_div[0]
                print(f"           Div  mismatch line {idx}: CPU={c}  GPU={g}")

    print("-" * 100)
    print("All correctness checks passed!" if all_pass else "SOME CORRECTNESS CHECKS FAILED.")

if __name__ == "__main__":
    main()
