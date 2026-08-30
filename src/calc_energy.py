import pandas as pd

# Physical Constants Derived from DRAMSim3
# Total Energy: 674,878,320 pJ over 32,000 requests = ~21,089 pJ/request = 21.08 nJ/fetch
HBM_ENERGY_NJ_PER_FETCH = 21.08
# CXL 2.0 / PCIe Gen5 PHY controllers overhead (Standard multiplier ~3x local HBM)
CXL_ENERGY_NJ_PER_FETCH = HBM_ENERGY_NJ_PER_FETCH * 3.0

# Real Mixtral Traces Output
TOTAL_TOKENS = 829441

results_layer15 = {
    "Pure Static": 57.51,
    "Periodic Re-profile": 61.02,
    "Pure LRU": 64.94
}

results_layer31 = {
    "Pure Static": 65.76,
    "Pure LRU": 65.91,
    "Periodic Re-profile": 67.26
}

def calc(hit_rates, layer_name):
    rows = []
    for strategy, hit_rate_pct in hit_rates.items():
        hit_rate = hit_rate_pct / 100.0
        hbm_fetches = TOTAL_TOKENS * hit_rate
        cxl_fetches = TOTAL_TOKENS * (1 - hit_rate)
        
        hbm_energy_mj = (hbm_fetches * HBM_ENERGY_NJ_PER_FETCH) / 1e6
        cxl_energy_mj = (cxl_fetches * CXL_ENERGY_NJ_PER_FETCH) / 1e6
        total_energy_mj = hbm_energy_mj + cxl_energy_mj
        avg_energy_nj = (total_energy_mj * 1e6) / TOTAL_TOKENS
        
        rows.append({
            "Layer": layer_name,
            "Strategy": strategy,
            "HBM Hit Rate (%)": hit_rate_pct,
            "Total Energy (mJ)": total_energy_mj,
            "Avg Energy per Token (nJ)": avg_energy_nj
        })
    return rows

df = pd.DataFrame(calc(results_layer15, "Layer 15") + calc(results_layer31, "Layer 31"))
df.to_csv(r"C:\Users\11ave\OneDrive\Documents\Birla Institute Of Technology And Science\Projects\MoE Memory Tiering\results\energy_metrics.csv", index=False)
print("Energy metrics calculated and saved.")
