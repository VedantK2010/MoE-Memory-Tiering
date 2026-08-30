import pandas as pd
import numpy as np

trace_path = r"C:\Users\11ave\Downloads\real_expert_trace_layer31.csv"
print(f"Loading {trace_path}...")
df = pd.read_csv(trace_path)

# Mixtral 8x7B has 8 experts. We'll give HBM enough capacity for 4 experts.
NUM_EXPERTS = 8
CAPACITY_K = 4

print(f"Total tokens: {len(df):,}")
print(f"Experts: {NUM_EXPERTS}, HBM Capacity: {CAPACITY_K}")

def eval_static():
    counts = df["expert_1"].value_counts()
    for e in range(NUM_EXPERTS):
        if e not in counts.index:
            counts[e] = 0
    top_k = set(counts.sort_values(ascending=False).head(CAPACITY_K).index)
    hits = sum(1 for e in df["expert_1"] if e in top_k)
    return hits / len(df)

def eval_lru():
    cache = []
    hits = 0
    for e in df["expert_1"]:
        if e in cache:
            hits += 1
            cache.remove(e)
            cache.append(e)
        else:
            if len(cache) >= CAPACITY_K:
                cache.pop(0) # Evict LRU
            cache.append(e)
    return hits / len(df)

def eval_periodic(interval):
    hits = 0
    current_top_k = set(range(CAPACITY_K))
    n = len(df)
    
    for start in range(0, n, interval):
        end = min(start + interval, n)
        window = df.iloc[start:end]["expert_1"].values
        
        # Apply current ranking
        for e in window:
            if e in current_top_k:
                hits += 1
                
        # Reprofile
        counts = pd.Series(window).value_counts()
        for e in range(NUM_EXPERTS):
            if e not in counts.index:
                counts[e] = 0
        current_top_k = set(counts.sort_values(ascending=False).head(CAPACITY_K).index)
        
    return hits / len(df)

print("-" * 40)
print(f"Pure Static HBM Hit Rate: {eval_static()*100:.2f}%")
print(f"Pure LRU HBM Hit Rate:    {eval_lru()*100:.2f}%")
print("-" * 40)

intervals = [100, 500, 1000, 2000, 5000, 10000, 20000]
for interval in intervals:
    hit_rate = eval_periodic(interval)
    print(f"Periodic ({interval:5d}) Hit Rate: {hit_rate*100:.2f}%")
