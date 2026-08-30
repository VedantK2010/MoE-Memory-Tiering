"""
Robustness Check: Second Dataset
===================================
Every result so far rests on ONE random trace (seed=42/7, skew alpha=6).
This script regenerates a completely independent dataset -- different
random seed AND a different skew strength (more extreme hot/cold split)
-- and re-runs the same four-strategy comparison, to check whether the
ranking of strategies (Periodic Re-profile > Hybrid > LRU ~ Static)
holds up, or was specific to the first dataset's particular randomness.
"""

import numpy as np
import pandas as pd

from nonstationary_experiment import (
    generate_nonstationary_trace, static_placement_from_first_phase,
    simulate_static_by_phase, simulate_lru_by_phase,
)
from hybrid_strategy import simulate_hybrid_by_phase
from periodic_reprofile import simulate_periodic_reprofile

NUM_EXPERTS = 8
TOP_K = 2
CAPACITY_K = 4
TOKENS_PER_PHASE = 20_000
NUM_PHASES = 3

# Deliberately different from the original dataset:
#   - different seed (99 instead of 7)
#   - stronger skew (alpha=3 instead of 6 -> more extreme hot/cold split,
#     since LOWER Dirichlet alpha = MORE skew)
NEW_SEED = 99
NEW_SKEW_ALPHA = 3
NEW_P_REPEAT = 0.20  # also test a different locality strength (Mixtral's
                      # layer-0 number was closer to random, ~14%; we pick
                      # a moderate middle value here, distinct from the
                      # 0.27 used before)


def generate_nonstationary_trace_custom(num_experts, top_k, tokens_per_phase, num_phases,
                                         p_repeat_target, seed, skew_alpha):
    """Same logic as generate_nonstationary_trace, but with a configurable
    skew_alpha so we can test a differently-shaped hot/cold distribution."""
    rng = np.random.default_rng(seed)
    all_rows, phase_labels, phase_base_probs = [], [], []

    base_shape = rng.dirichlet(alpha=[skew_alpha] * num_experts)
    roll_amount = max(1, num_experts // num_phases)

    prev_first = None
    for phase in range(num_phases):
        base_probs = np.roll(base_shape, shift=phase * roll_amount)
        phase_base_probs.append(base_probs)
        collision_prob = float(np.sum(base_probs ** 2))
        p_forced = max(0.0, (p_repeat_target - collision_prob) / (1 - collision_prob))

        if prev_first is None:
            prev_first = int(rng.choice(num_experts, p=base_probs))

        for _ in range(tokens_per_phase):
            if rng.random() < p_forced:
                first = prev_first
            else:
                first = int(rng.choice(num_experts, p=base_probs))
            remaining = [e for e in range(num_experts) if e != first]
            remaining_probs = base_probs[remaining] / base_probs[remaining].sum()
            rest = rng.choice(remaining, size=top_k - 1, replace=False, p=remaining_probs)
            all_rows.append([first] + list(rest))
            phase_labels.append(phase)
            prev_first = first

    df = pd.DataFrame(all_rows, columns=[f"expert_{i+1}" for i in range(top_k)])
    df.insert(0, "token_id", np.arange(len(df)))
    df["phase"] = phase_labels
    return df, phase_base_probs


if __name__ == "__main__":
    trace_df, phase_probs = generate_nonstationary_trace_custom(
        NUM_EXPERTS, TOP_K, TOKENS_PER_PHASE, NUM_PHASES, NEW_P_REPEAT, NEW_SEED, NEW_SKEW_ALPHA
    )
    trace_df.to_csv("dataset2_trace.csv", index=False)

    print("Dataset 2: hottest expert per phase")
    for i, probs in enumerate(phase_probs):
        print(f"  Phase {i}: hottest = expert {int(np.argmax(probs))}, "
              f"max share = {probs.max():.3f} (dataset 1 used alpha=6, this uses alpha={NEW_SKEW_ALPHA})")

    # --- Pure Static ---
    tier_map = static_placement_from_first_phase(trace_df, NUM_EXPERTS, CAPACITY_K)
    static_phase = simulate_static_by_phase(trace_df, tier_map)

    # --- Pure LRU ---
    lru_phase = simulate_lru_by_phase(trace_df, CAPACITY_K)

    # --- Hybrid (half reserved, half LRU) ---
    counts = trace_df[trace_df["phase"] == 0]["expert_1"].value_counts()
    ranked_from_phase0 = counts.index.tolist()
    for e in range(NUM_EXPERTS):
        if e not in ranked_from_phase0:
            ranked_from_phase0.append(e)
    hybrid_phase = simulate_hybrid_by_phase(trace_df, CAPACITY_K, CAPACITY_K // 2, ranked_from_phase0)

    # --- Periodic Re-profile (using the winning interval from dataset 1: 2000) ---
    reprofile_df = simulate_periodic_reprofile(trace_df, CAPACITY_K, NUM_EXPERTS, 2000)
    reprofile_phase = reprofile_df.groupby("phase")["avg_time_ns"].mean().reset_index()

    # --- Combine and compare overall averages ---
    static_overall = static_phase["avg_time_ns"].mean()
    lru_overall = lru_phase["avg_time_ns"].mean()
    hybrid_overall = hybrid_phase["avg_time_ns"].mean()
    reprofile_overall = reprofile_phase["avg_time_ns"].mean()

    summary = pd.DataFrame({
        "strategy": ["Pure Static", "Pure LRU", "Hybrid", "Periodic Re-profile"],
        "dataset1_overall_ns": [129513.9, 129304.2, 126278.5, 123395.1],
        "dataset2_overall_ns": [static_overall, lru_overall, hybrid_overall, reprofile_overall],
    })
    summary["dataset1_rank"] = summary["dataset1_overall_ns"].rank().astype(int)
    summary["dataset2_rank"] = summary["dataset2_overall_ns"].rank().astype(int)
    summary.to_csv("robustness_check_summary.csv", index=False)

    print("\n=== Robustness check: strategy ranking across two independent datasets ===")
    print(summary.to_string(index=False))

    if (summary["dataset1_rank"] == summary["dataset2_rank"]).all():
        print("\n--> Strategy ranking is IDENTICAL across both datasets. Result is robust.")
    else:
        print("\n--> Strategy ranking CHANGED between datasets -- worth investigating which part is sensitive.")
