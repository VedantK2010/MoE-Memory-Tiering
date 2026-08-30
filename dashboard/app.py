import streamlit as st
import pandas as pd

st.set_page_config(page_title="CXL-Based Memory Tiering", layout="wide")

@st.cache_data
def load_data():
    tiering_sweep = pd.read_csv("../results/tiering_sweep_results.csv")
    reprofile_sweep = pd.read_csv("../results/reprofile_interval_sweep.csv")
    energy_metrics = pd.read_csv("../results/energy_metrics.csv")
    return tiering_sweep, reprofile_sweep, energy_metrics

try:
    tiering_sweep, reprofile_sweep, energy_metrics = load_data()
except Exception:
    st.error("Error loading CSV files. Ensure you are running this from the project directory.")
    st.stop()

st.title("Interactive Demo: CXL-Based Memory Tiering for MoE")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Hardware Validation (DRAMSim3)",
    "2. Capacity vs Latency (Baseline)", 
    "3. Synthetic Workload Shift", 
    "4. Real-World Mixtral Trace",
    "5. Energy & Power Analysis"
])

with tab1:
    st.header("Cycle-Accurate Hardware Validation")
    st.markdown("Before evaluating macroscopic caching strategies, we validated our memory physics using the **DRAMSim3** cycle-accurate simulator.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated Commands", "256,000", "Burst sequence (64 bytes each)")
    col2.metric("Avg Burst Latency (HBM2)", "60.71 ns", "Cycle-accurate measurement")
    col3.metric("Row Buffer Hit Rate", "88.08%", "Validates sequential fetching")
    
    st.info("💡 **Hardware Insight:** The massive 88% row-buffer hit rate proves that fetching multi-kilobyte expert chunks sequentially drastically reduces memory activation overhead, perfectly grounding our Python model's macroscopic latency assumptions.")

with tab2:
    st.header("Baseline Trade-off: Capacity vs Latency")
    st.markdown("Formula driving the simulation:")
    st.latex(r"Access\_Time = Base\_Latency + \left( \frac{Transfer\_Size}{Bandwidth} \right)")
    
    hbm_experts = st.slider("Experts mapped to HBM (Capacity Budget)", min_value=0, max_value=8, value=4)
    row = tiering_sweep[tiering_sweep["num_experts_in_hbm"] == hbm_experts].iloc[0]
    avg_lat = row["avg_time_per_access_ns"]
    
    col1, col2 = st.columns(2)
    col1.metric("Average Access Latency", f"{avg_lat / 1000:.1f} µs")
    col2.metric("HBM Capacity Saved", f"{(8 - hbm_experts) * 16} MiB", f"{(8 - hbm_experts) * 12.5}% of Model Parameters") 
    
    st.markdown("#### Full Parameter Sweep")
    st.line_chart(tiering_sweep.set_index("num_experts_in_hbm")[["avg_time_per_access_ns"]])

with tab3:
    st.header("The Synthetic Hypothesis")
    st.markdown("Hypothesis: **Periodic Re-profiling** outperforms Pure Static and LRU under non-stationary workload shifts.")
    
    st.markdown("#### Re-profiling Interval Sweep")
    st.line_chart(reprofile_sweep.set_index("reprofile_interval")[["overall_avg_time_ns"]])
    
    st.markdown("#### Phase-by-Phase Comparison")
    st.image("../results/four_strategy_comparison.png", caption="Periodic Re-profiling successfully learns the shifting block phases, winning the synthetic benchmark.")

with tab4:
    st.header("The Real-World Plot Twist (Mixtral 8x7B)")
    st.markdown("We extracted **~1.6 Million real routing decisions** (Layer 15 + Layer 31) from the HuggingFace `allenai/analysis_mixtral` dataset to test our tiering policies.")
    
    # INTERACTIVE TOGGLE FOR THE TWO LAYERS
    layer_choice = st.radio("Select Neural Network Depth to Analyze:", ["Layer 15 (Middle Layer)", "Layer 31 (Deep Layer)"], horizontal=True)
    
    if "15" in layer_choice:
        st.markdown("### Middle Layers: Highly Reactive")
        st.markdown("At middle layers, context shifts rapidly and unpredictably. **LRU dominates** by adapting instantly.")
        data = {"Pure Static": 57.51, "Periodic (Best)": 61.02, "Pure LRU": 64.94}
        winner = "Pure LRU"
    else:
        st.markdown("### Deep Layers: Highly Specialized")
        st.markdown("At deep layers, experts become highly specialized for deep semantic logic. **Static and Periodic** leap forward as specific experts heavily dominate the routing.")
        data = {"Pure Static": 65.76, "Pure LRU": 65.91, "Periodic (Best)": 67.26}
        winner = "Periodic Re-profiling"
        
    df = pd.DataFrame(list(data.items()), columns=["Strategy", "HBM Hit Rate (%)"]).set_index("Strategy")
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.dataframe(df.style.format("{:.2f}%"))
    with col2:
        st.bar_chart(df)
        
    st.success(f"🏆 **Winner at {layer_choice.split(' ')[0]} {layer_choice.split(' ')[1]}: {winner}**")

with tab5:
    st.header("Hardware Energy & Power Analysis")
    st.markdown("Fetching multi-megabyte experts from memory is extremely power-intensive. Using the cycle-accurate DRAMSim3 traces, we determined that an HBM2 burst costs **~21.08 nJ/fetch**.")
    st.markdown("Because CXL operates over the PCIe bus (Gen5 PHY), CXL accesses incur roughly a **3x energy penalty** (~63.24 nJ/fetch).")
    
    st.markdown("### Total Energy Consumed (Over 829,441 Tokens)")
    st.dataframe(energy_metrics.style.format({
        'HBM Hit Rate (%)': '{:.2f}%',
        'Total Energy (mJ)': '{:.2f} mJ',
        'Avg Energy per Token (nJ)': '{:.2f} nJ'
    }))
    
    layer_filter = st.radio("Select Layer for Energy Chart:", ["Layer 15", "Layer 31"])
    chart_data = energy_metrics[energy_metrics["Layer"] == layer_filter].set_index("Strategy")[["Total Energy (mJ)"]]
    st.bar_chart(chart_data)
    
    st.success("?? **Conclusion:** Intelligent tiering (like LRU at middle layers) doesn't just save latency�it saves massive amounts of power by minimizing trips across the CXL bus!")
