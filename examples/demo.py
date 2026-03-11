#!/usr/bin/env python3
"""
Q-CLAW Demo — Solve real optimization problems with quantum computing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import qclaw

print("=" * 60)
print("  Q-CLAW v0.1 — Quantum-Classical Logic & Action Wrapper")
print("=" * 60)

# Initialize agent (falls back to simulator if Wukong offline)
agent = qclaw.Agent(qpu_provider="origin_wukong", fallback="simulator")

# ─── 1. Traveling Salesman (Logistics) ───
print("\n🚚 TSP: Delivery Route Optimization")
print("-" * 40)

cities = qclaw.TSP.from_coordinates({
    "Houston": (29.76, -95.37),
    "Dallas": (32.78, -96.80),
    "Austin": (30.27, -97.74),
    "San Antonio": (29.42, -98.49),
    "El Paso": (31.76, -106.49),
})

result = agent.ping(cities, mode="auto")
print(f"  Route: {' → '.join(result.get('route', ['?']))}")
print(f"  Distance: {result.get('total_distance', '?')} km")
print(f"  Qubits: {result.get('qubits_used', '?')}")
print(f"  Backend: {result.get('backend_used', '?')}")
print(f"  Latency: {result.get('total_latency_ms', '?')}ms")

# ─── 2. Portfolio Optimization (Finance) ───
print("\n💰 Portfolio: Quantum Asset Selection")
print("-" * 40)

assets = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "AMD"]
returns = np.array([0.12, 0.10, 0.11, 0.25, 0.15, 0.08, 0.13, 0.20])
cov = np.random.RandomState(42).uniform(0.01, 0.05, (8, 8))
cov = (cov + cov.T) / 2
np.fill_diagonal(cov, np.random.RandomState(42).uniform(0.02, 0.08, 8))

portfolio = qclaw.PortfolioOptimizer(
    assets=assets, returns=returns, covariance=cov,
    risk_factor=0.3, budget=4
)

result = agent.ping(portfolio)
print(f"  Selected: {result.get('selected_assets', [])}")
print(f"  Expected Return: {result.get('expected_return', '?')}")
print(f"  Risk: {result.get('portfolio_risk', '?')}")
print(f"  Backend: {result.get('backend_used', '?')}")
print(f"  Latency: {result.get('total_latency_ms', '?')}ms")

# ─── 3. Max Cut (Network Design) ───
print("\n🔗 MaxCut: Network Partitioning")
print("-" * 40)

adj = np.array([
    [0,1,1,0,0,1],
    [1,0,1,1,0,0],
    [1,1,0,1,1,0],
    [0,1,1,0,1,1],
    [0,0,1,1,0,1],
    [1,0,0,1,1,0],
])

maxcut = qclaw.MaxCut(adj, node_labels=["A","B","C","D","E","F"])
result = agent.ping(maxcut)
print(f"  Set A: {result.get('set_a', [])}")
print(f"  Set B: {result.get('set_b', [])}")
print(f"  Cut Value: {result.get('cut_value', '?')}")
print(f"  Backend: {result.get('backend_used', '?')}")

# ─── 4. Job Scheduling (GPU Compute) ───
print("\n⚡ JobScheduler: GPU Workload Balancing")
print("-" * 40)

scheduler = qclaw.JobScheduler(
    jobs=["train_llm", "inference_batch", "fine_tune", "embeddings", "eval_suite", "data_prep"],
    durations=[120, 30, 60, 15, 45, 20],
    deadlines=[200, 60, 150, 30, 100, 50],
    n_machines=3
)

result = agent.ping(scheduler)
print(f"  Makespan: {result.get('makespan', '?')} min")
for machine, jobs in result.get('machine_loads', {}).items():
    job_names = [j['job'] for j in jobs]
    print(f"  {machine}: {', '.join(job_names)}")
print(f"  Backend: {result.get('backend_used', '?')}")

# ─── Stats ───
print("\n" + "=" * 60)
stats = agent.stats
print(f"  Total Pings: {stats['total_pings']}")
print(f"  Avg Latency: {stats['avg_latency_ms']}ms")
print(f"  QPU Pings: {stats['qpu_pings']}")
print(f"  Sim Pings: {stats['sim_pings']}")
print("=" * 60)
