"""
Convert Real Mixtral Routing Data into Our Simulator's Trace Format
=======================================================================
RUN THIS IN GOOGLE COLAB (not locally) -- it needs internet access to
Hugging Face, which isn't available in the sandboxed environment used
to build the rest of this project.

This downloads allenai/analysis_mixtral -- REAL recorded expert routing
decisions from actual Mixtral 8x7B inference on real text -- and
converts it into the same expert_trace.csv format our tier_simulator.py,
nonstationary_experiment.py, hybrid_strategy.py, and periodic_reprofile.py
scripts already expect. Once converted, every experiment we've already
built can be re-run on REAL data instead of synthetic data, with zero
changes to the analysis code itself.

STEPS TO RUN THIS IN COLAB:
  1. In a Colab cell: !pip install datasets
  2. Paste and run this whole script in a Colab cell (or upload as a .py
     file and run with %run convert_real_data.py)
  3. Download the resulting real_expert_trace.csv from Colab's file
     browser (left sidebar) and upload it back into this project's
     folder to re-run our existing experiments on it.
"""

# !pip install datasets   # <-- uncomment and run this first in Colab

from datasets import load_dataset
import numpy as np
import pandas as pd

# --- Step 1: Download the real Mixtral routing dataset ---
print("Downloading allenai/analysis_mixtral (real Mixtral routing data)...")
ds = load_dataset("allenai/analysis_mixtral")["train"]
print(f"Loaded {len(ds)} real text sequences, each with per-token, per-layer expert choices.")

# --- Step 2: Pick which layer to extract (Mixtral has 32 layers, 0-31) ---
# Mixtral's routing behavior differs somewhat by depth (see the Mixtral
# paper's Section 5 / Table 5 -- locality is stronger at deeper layers).
# We'll extract layer 15 (a middle layer) to match the "mid" locality
# regime we used in our synthetic non-stationary trace. Change this to
# 0 or 31 to compare against the first/last layers instead.
LAYER_TO_EXTRACT = 15
TOP_K = 2  # Mixtral routes to top-2 experts; adjust if the data has a
           # different shape than expected -- INSPECT FIRST (see below)

# --- IMPORTANT: inspect the actual shape before assuming its structure ---
sample = ds[0]
exp_ids_sample = sample["exp_ids"]
print(f"\nexp_ids type: {type(exp_ids_sample)}")
print(f"Outer length (should match num_layers or seq_len -- check!): {len(exp_ids_sample)}")
print(f"First inner element: {exp_ids_sample[0][:10] if hasattr(exp_ids_sample[0], '__len__') else exp_ids_sample[0]}")
print("\n--> STOP AND LOOK at the shape printed above before trusting the")
print("    extraction below -- confirm whether exp_ids is organized as")
print("    [layer][token] or [token][layer] before proceeding, since this")
print("    determines how the code below should index into it.")

# --- Step 3: Extract a real token -> expert(s) sequence ---
# This assumes exp_ids is organized as [token_index][layer_index] -> expert_id
# (a single int per layer per token, i.e. TOP-1 only was recorded).
# ADJUST THIS INDEXING based on what the inspection above actually shows.
all_rows = []
for seq_idx in range(len(ds)):
    exp_ids = ds[seq_idx]["exp_ids"]
    for token_idx in range(len(exp_ids)):
        try:
            expert_at_layer = exp_ids[token_idx][LAYER_TO_EXTRACT]
            all_rows.append({"expert_1": expert_at_layer})
        except (IndexError, TypeError):
            continue  # skip malformed entries rather than crash the whole run

real_trace_df = pd.DataFrame(all_rows)
real_trace_df.insert(0, "token_id", np.arange(len(real_trace_df)))

print(f"\nExtracted {len(real_trace_df)} real (token, expert) observations from layer {LAYER_TO_EXTRACT}.")
print(real_trace_df.head(10))

# --- Step 4: Save in a format ready to bring back into our project ---
real_trace_df.to_csv("real_expert_trace.csv", index=False)
print("\nSaved: real_expert_trace.csv")
print("Download this file from Colab's file browser, then bring it back")
print("to continue the analysis with our existing simulator scripts.")

# --- Step 5 (informational): quick sanity check against our synthetic calibration ---
f_i = real_trace_df["expert_1"].value_counts(normalize=True).sort_index()
print("\nReal per-expert selection frequency (compare against our synthetic f_i):")
print(f_i)

repeat_rate = (real_trace_df["expert_1"].values[1:] == real_trace_df["expert_1"].values[:-1]).mean()
print(f"\nReal consecutive same-expert repeat rate at layer {LAYER_TO_EXTRACT}: {repeat_rate:.3f}")
print("(Compare this to the 0.27 target we used, based on Mixtral Table 5's reported range)")
