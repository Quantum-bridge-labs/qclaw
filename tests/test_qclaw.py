"""
Q-CLAW Tests — Verify solvers return valid results.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qclaw.qubo import QUBOMatrix
from qclaw.problems import TSP, MaxCut, PortfolioOptimizer, JobScheduler
from qclaw.backend import SimulatorBackend, BackendResult


def test_qubo_matrix():
    """QUBO matrix creation and energy calculation."""
    Q = np.array([[1, -2], [-2, 3]], dtype=float)
    qubo = QUBOMatrix(Q)
    assert qubo.n_vars == 2

    # x = [0, 0] → energy = 0
    assert qubo.energy(np.array([0, 0])) == 0.0
    # x = [1, 0] → energy = Q[0,0] = 1
    assert qubo.energy(np.array([1, 0])) == 1.0
    # x = [0, 1] → energy = Q[1,1] = 3
    assert qubo.energy(np.array([0, 1])) == 3.0
    # x = [1, 1] → energy = 1 + 3 + (-2) + (-2) = 0
    assert qubo.energy(np.array([1, 1])) == 0.0
    print("  ✓ QUBO matrix OK")


def test_tsp_from_coordinates():
    """TSP problem from city coordinates."""
    coords = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (1, 1),
        "D": (0, 1),
    }
    tsp = TSP.from_coordinates(coords)
    assert tsp.n == 4
    assert len(tsp.cities) == 4
    assert tsp.qubo.n_vars == 16  # 4x4
    print("  ✓ TSP coordinates → QUBO OK")


def test_tsp_distance_matrix():
    """TSP with explicit distance matrix."""
    cities = ["Houston", "Dallas", "Austin"]
    distances = [
        [0, 239, 165],
        [239, 0, 195],
        [165, 195, 0],
    ]
    tsp = TSP(cities, np.array(distances, dtype=float))
    assert tsp.n == 3
    assert tsp.qubo.n_vars == 9  # 3x3
    print("  ✓ TSP distance matrix → QUBO OK")


def test_maxcut():
    """MaxCut problem formulation."""
    # Triangle graph
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)
    mc = MaxCut(adj, node_labels=["A", "B", "C"])
    assert mc.qubo.n_vars == 3
    print("  ✓ MaxCut OK")


def test_portfolio():
    """Portfolio optimization problem."""
    assets = ["AAPL", "GOOGL", "TSLA"]
    returns = np.array([0.12, 0.08, 0.15])
    cov = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.03, 0.01],
        [0.02, 0.01, 0.06],
    ])
    port = PortfolioOptimizer(assets, returns, cov, risk_factor=0.5)
    assert port.qubo.n_vars == 3
    print("  ✓ Portfolio OK")


def test_job_scheduler():
    """Job scheduling problem."""
    sched = JobScheduler(
        jobs=["render_A", "render_B", "train_C"],
        durations=[10, 20, 15],
        deadlines=[30, 40, 35],
        n_machines=2,
    )
    assert sched.qubo.n_vars == 6  # 3 jobs × 2 machines
    print("  ✓ Job Scheduler OK")


def test_simulator_backend():
    """Simulated annealing solver produces valid results."""
    # Simple 3-variable QUBO — minimum at [1, 0, 1]
    Q = np.array([
        [-5, 2, 1],
        [ 2,-3, 1],
        [ 1, 1,-4],
    ], dtype=float)
    qubo = QUBOMatrix(Q)
    backend = SimulatorBackend(n_iterations=5000)
    result = backend.solve(qubo)

    assert isinstance(result, BackendResult)
    assert result.backend == "local_simulator"
    assert result.latency_ms > 0
    assert len(result.solution) == 3
    assert all(b in (0, 1) for b in result.solution)
    # Should find a low-energy solution
    assert result.energy <= 0, f"Expected negative energy, got {result.energy}"
    print(f"  ✓ Simulator backend OK (energy={result.energy:.2f}, {result.latency_ms:.0f}ms)")


def test_tsp_end_to_end():
    """Full pipeline: TSP → QUBO → Simulator → Route."""
    coords = {
        "Houston": (29.76, -95.37),
        "Dallas": (32.78, -96.80),
        "Austin": (30.27, -97.74),
    }
    tsp = TSP.from_coordinates(coords)
    backend = SimulatorBackend(n_iterations=10000)
    result = backend.solve(tsp.qubo)

    assert isinstance(result, BackendResult)
    assert len(result.solution) == 9  # 3 cities × 3 positions
    print(f"  ✓ TSP end-to-end OK (energy={result.energy:.2f})")


def test_backend_result_serialization():
    """BackendResult.to_dict() produces valid JSON."""
    result = BackendResult(
        solution=np.array([1, 0, 1]),
        energy=-5.0,
        raw={"test": True},
        backend="test",
        latency_ms=42.0,
        shots=100,
    )
    d = result.to_dict()
    json_str = json.dumps(d)
    parsed = json.loads(json_str)
    assert parsed["energy"] == -5.0
    assert parsed["backend"] == "test"
    assert parsed["solution"] == [1, 0, 1]
    print("  ✓ Result serialization OK")


if __name__ == "__main__":
    print("\nQ-CLAW Test Suite\n" + "=" * 40)
    test_qubo_matrix()
    test_tsp_from_coordinates()
    test_tsp_distance_matrix()
    test_maxcut()
    test_portfolio()
    test_job_scheduler()
    test_simulator_backend()
    test_tsp_end_to_end()
    test_backend_result_serialization()
    print("\n" + "=" * 40)
    print("All tests passed ✓\n")
