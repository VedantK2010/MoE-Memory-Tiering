"""
Scaled-Down, Burst-Level DRAMSim3 Trace
==========================================
Our original dramsim3_trace.txt issued ONE trace line per expert access,
which DRAMSim3 correctly read as one small burst-sized command (e.g. 64
bytes) -- NOT a full 16MB expert transfer. That's why the earlier
DRAMSim3 latency numbers (~59ns) weren't directly comparable to our
Python model's per-access time (~19,681ns): they were measuring
different amounts of data moved per "access."

To properly simulate a real expert weight FETCH as multiple sequential
burst commands, we'd need ~262,144 commands per access (16MB / 64B) --
at 100,000 accesses, that's over 26 BILLION trace lines. Not runnable.

This script instead generates an HONEST, DISCLOSED scaled-down version:
  - Transfer unit reduced from 16MB to 4KB (a standard memory "page"
    size -- a legitimate, explainable simplification, not an arbitrary
    shrink)
  - Uses a SUBSAMPLE of accesses (first N tokens) rather than the full
    50,000-token trace, to keep the file a runnable size

This keeps the file small enough to actually simulate, while still
representing each expert access as a real sequence of burst-level
commands (not a single command), which is the methodologically correct
way to feed DRAMSim3 a "bulk transfer."
"""

import pandas as pd

# --- Configuration ---
NUM_TOKENS_TO_SAMPLE = 2000     # subsample size (out of the original 50,000)
TRANSFER_CHUNK_BYTES = 4096     # scaled-down "page size" per expert access
                                 # (was 16 MiB in the original trace)
BURST_SIZE_BYTES = 64           # standard DDR4 burst/command size --
                                 # VERIFY this against your specific
                                 # DRAMSim3 config file if precision matters:
                                 #   grep -i "burst" configs/DDR4_8Gb_x8_3200.ini
CYCLES_BETWEEN_ACCESSES = 100   # matches the spacing used in the original trace
BASE_ADDRESS = 0x00000000

COMMANDS_PER_ACCESS = TRANSFER_CHUNK_BYTES // BURST_SIZE_BYTES
# Give each expert a large enough address region that its commands never
# overlap with another expert's, regardless of chunk size chosen above.
EXPERT_ADDRESS_STRIDE = 0x10000000  # 256 MiB apart, plenty of headroom


def generate_burst_trace(trace_df, num_tokens, out_path):
    sampled = trace_df.head(num_tokens)
    expert_cols = [c for c in sampled.columns if c.startswith("expert_")]

    lines = []
    cycle = 0
    for _, row in sampled.iterrows():
        for col in expert_cols:
            expert_id = int(row[col])
            expert_base = BASE_ADDRESS + expert_id * EXPERT_ADDRESS_STRIDE
            # Issue COMMANDS_PER_ACCESS sequential burst-sized commands,
            # representing one real "fetch this expert's weights" event
            # as the several small memory commands it would actually take.
            for cmd_idx in range(COMMANDS_PER_ACCESS):
                addr = expert_base + cmd_idx * BURST_SIZE_BYTES
                lines.append(f"0x{addr:010x} READ {cycle}")
                cycle += 1  # commands within one access are back-to-back
            cycle += CYCLES_BETWEEN_ACCESSES  # gap before the NEXT access begins

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines), cycle


if __name__ == "__main__":
    trace_df = pd.read_csv("/mnt/user-data/outputs/expert_trace.csv")

    num_lines, final_cycle = generate_burst_trace(trace_df, NUM_TOKENS_TO_SAMPLE, "dramsim3_burst_trace.txt")

    print(f"Sampled first {NUM_TOKENS_TO_SAMPLE} tokens (of {len(trace_df)} total)")
    print(f"Transfer unit: {TRANSFER_CHUNK_BYTES} bytes per expert access "
          f"({COMMANDS_PER_ACCESS} commands of {BURST_SIZE_BYTES} bytes each)")
    print(f"Generated {num_lines} trace lines")
    print(f"Final cycle reached: {final_cycle}")
    print(f"\nRun DRAMSim3 with:")
    print(f"  ./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini -c {final_cycle + 1000} -t dramsim3_burst_trace.txt")
