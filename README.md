# CXL-Based Memory Tiering for MoE Models


## Project Overview
This repository contains the simulation framework, traces, and results for analyzing CXL-based memory expansion in Mixture-of-Experts (MoE) models (like Mixtral 8x7B). By using a decoupled simulation methodology (DRAMSim3 for hardware physics, Python for macroscopic routing), we evaluated memory caching strategies to reduce the latency and energy costs associated with CXL memory.

## Directory Structure & File Descriptions

### /src (Source Code)
*   **	ier_simulator.py**: The baseline analytical engine. Models the access latency trade-offs between HBM and CXL.
*   **periodic_reprofile.py**: Implements our core hypothesis—re-evaluating expert popularity periodically vs. Pure Static and Pure LRU caching.
*   **hybrid_strategy.py** & **	une_hybrid_dataset2.py**: Evaluates splitting the cache budget between static reservations and an LRU pool.
*   **generate_trace.py** & **generate_burst_trace.py**: Synthetic data generators mimicking MoE Dirichlet skews and temporal locality for testing.
*   **
obustness_check.py**: A cross-validation script ensuring the strategies hold up under different random seeds and skews.
*   **calc_energy.py**: Calculates total energy (in millijoules) based on HBM/CXL access counts mapped against physical DRAMSim3 data.

### /data (Datasets & Traces)
*   **
eal_expert_trace.csv**: 829,000+ real-world top-2 routing decisions extracted from HuggingFace (Layer 15).
*   **
eal_expert_trace_layer31.csv**: 829,000+ real routing decisions extracted from the deepest layer (Layer 31), highlighting heavy expert specialization.
*   *(Note: Large .csv and .txt trace files are intentionally excluded via .gitignore to preserve repository hygiene).*

### /results (Metrics & Visualization)
*   **	iering_sweep_results.csv** & **	iering_sweep.png**: Demonstrates the linear latency penalty of offloading experts to CXL.
*   **our_strategy_comparison.png**: Phase-by-phase visualization showing how different caching policies react to workload shifts.
*   **energy_metrics.csv**: Millijoule tracking showing how optimal tiering (like LRU) saves ~8% of total memory power.

### /dashboard (Interactive Demo)
*   **pp.py**: A Streamlit web application. Run streamlit run app.py to view an interactive visualization of the simulation results, capacity sliders, and real-world trace hit rates.
