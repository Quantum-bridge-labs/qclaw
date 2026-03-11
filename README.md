# Q-CLAW

**Quantum-Classical Logic & Action Wrapper**

Route reasoning to LLMs. Route optimization to quantum hardware.

Q-CLAW is the bridge between agentic AI and quantum processing units. It takes real-world optimization problems — logistics, portfolio selection, scheduling, network design — and executes them on OriginQ Wukong, a 72-qubit superconducting quantum computer.

## Why

Current AI agents are bottlenecked by classical logic loops. LLMs hallucinate solutions to optimization problems because they're statistically incapable of solving NP-hard tasks in real-time. Q-CLAW routes these problems to quantum hardware that can.

## Quick Start

```python
import qclaw

agent = qclaw.Agent(qpu_provider="origin_wukong")

# Optimize delivery routes across 8 cities
tsp = qclaw.TSP.from_coordinates({
    "Houston": (29.76, -95.37),
    "Dallas": (32.78, -96.80),
    "Austin": (30.27, -97.74),
    "San Antonio": (29.42, -98.49),
})

result = agent.ping(tsp)
# → Route: Dallas → Austin → Houston → San Antonio
# → Distance: 1238 km
# → Latency: 1127ms
# → Backend: origin_wukong (or simulator fallback)
```

## Problems Supported

| Problem | Use Case | Qubits | Price |
|---------|----------|--------|-------|
| **TSP** | Delivery routing, logistics, supply chain | n² | $5/ping |
| **MaxCut** | Network design, clustering, VLSI layout | n | $5/ping |
| **Portfolio** | Asset selection, risk optimization | n | $10/ping |
| **JobScheduler** | GPU workload balancing, manufacturing | n×m | $5/ping |

## API

```bash
# Solve TSP via API
curl -X POST https://qclaw.gpupulse.dev/ping/tsp \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": {
      "Houston": [29.76, -95.37],
      "Dallas": [32.78, -96.80]
    }
  }'
```

## Architecture

```
Natural Language / Structured Input
        │
    Q-CLAW Agent
        │
   ┌────┴────┐
   │  QUBO   │  ← Maps problems to quantum-native formulation
   │ Mapper  │
   └────┬────┘
        │
   ┌────┴────┐
   │  QAOA   │  ← Quantum Approximate Optimization Algorithm
   │ Circuit │
   └────┬────┘
        │
   ┌────┴─────────────┐
   │OriginQ Wukong    │  ← 72-qubit superconducting QPU (Hefei, China)
   │(real hardware)    │
   └──────────────────┘
        │
   Interpreted Results
```

## Stack

- **QUBO Mapper** — TSP, MaxCut, Portfolio, Scheduling → quantum-ready formulation
- **QAOA Engine** — Variational quantum circuits, auto-tuned parameters
- **OriginQ Backend** — Direct integration with Wukong 72-qubit chip via PyQPanda
- **Simulator Fallback** — Simulated annealing when hardware is in maintenance
- **API Server** — Pay-per-ping REST API with credit system

## Hardware

Q-CLAW connects to **OriginQ Wukong** (悟空), a 72-qubit superconducting quantum computer operated by Origin Quantum (本源量子) in Hefei, China. This gives Q-CLAW access to real quantum computation — not simulation, not emulation.

The Qiskit ↔ OriginIR transpiler (built in-house) enables bidirectional circuit translation between Western and Chinese quantum ecosystems. Patent pending.

## Install

```bash
pip install qclaw

# With OriginQ hardware support
pip install qclaw[originq]
```

## Benchmark

Run the same problem on quantum hardware and classical simulation:

```python
comparison = agent.benchmark(tsp)
print(f"Classical: {comparison['classical']['total_distance']} km in {comparison['classical']['latency_ms']}ms")
print(f"Quantum:   {comparison['quantum']['total_distance']} km in {comparison['quantum']['latency_ms']}ms")
print(f"Speedup:   {comparison['speedup']}x")
```

## License

MIT

## Links

- [GPUPulse](https://gpupulse.dev)
- [Quantum Bridge Labs](https://github.com/Quantum-bridge-labs)
- [Transpiler](https://transpiler.gpupulse.dev)
- [API Docs](https://qclaw.gpupulse.dev)
