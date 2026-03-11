"""
Quantum Backends — Execute QUBO on real QPUs or simulators.
Supports OriginQ Wukong (72-qubit superconducting) and local simulation.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Optional, List
from .qubo import QUBOMatrix


class BackendResult:
    """Result from a quantum backend execution."""
    
    def __init__(self, solution: np.ndarray, energy: float, raw: Dict,
                 backend: str, latency_ms: float, shots: int, metadata: Dict = None):
        self.solution = solution
        self.energy = energy
        self.raw = raw
        self.backend = backend
        self.latency_ms = latency_ms
        self.shots = shots
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            "solution": self.solution.tolist(),
            "energy": self.energy,
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "shots": self.shots,
            "metadata": self.metadata,
        }
    
    def __repr__(self):
        return f"BackendResult(energy={self.energy:.4f}, backend='{self.backend}', latency={self.latency_ms:.0f}ms)"


class OriginQBackend:
    """
    Execute QUBO on OriginQ Wukong (72-qubit superconducting QPU).
    Uses QAOA (Quantum Approximate Optimization Algorithm) to solve QUBO.
    """
    
    def __init__(self, api_key: Optional[str] = None, chip_id: int = 2,
                 shots: int = 1000, p_layers: int = 1):
        self.api_key = api_key or self._load_key()
        self.chip_id = chip_id
        self.shots = shots
        self.p_layers = p_layers
        self._qm = None
    
    def _load_key(self) -> str:
        env_path = os.path.expanduser("~/.openclaw/.originq_env")
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                "OriginQ API key not found. Save to ~/.openclaw/.originq_env as ORIGINQ_API_KEY=<key>"
            )
        with open(env_path) as f:
            for line in f:
                if line.startswith("ORIGINQ_API_KEY="):
                    return line.strip().split("=", 1)[1]
        raise ValueError("ORIGINQ_API_KEY not found in env file")
    
    def _connect(self):
        if self._qm is not None:
            return
        from pyqpanda import QCloud
        self._qm = QCloud()
        self._qm.set_configure(72, 72)
        self._qm.init_qvm(self.api_key, enable_pqc_encryption=False, request_time_out=600)
    
    def _build_qaoa_circuit(self, qubo: QUBOMatrix, gamma: float, beta: float):
        """Build a QAOA circuit for the given QUBO problem."""
        from pyqpanda import QProg, H, CNOT, RZ, RX
        
        self._connect()
        n = qubo.n_vars
        
        if n > 72:
            raise ValueError(f"Problem size {n} exceeds Wukong's 72 qubits")
        
        qubits = self._qm.qAlloc_many(n)
        cbits = self._qm.cAlloc_many(n)
        prog = QProg()
        
        # Initial superposition
        for i in range(n):
            prog << H(qubits[i])
        
        # QAOA layers
        for _ in range(self.p_layers):
            # Cost unitary (problem Hamiltonian)
            for i in range(n):
                if abs(qubo.Q[i, i]) > 1e-10:
                    prog << RZ(qubits[i], 2 * gamma * qubo.Q[i, i])
                for j in range(i + 1, n):
                    if abs(qubo.Q[i, j]) > 1e-10:
                        prog << CNOT(qubits[i], qubits[j])
                        prog << RZ(qubits[j], 2 * gamma * qubo.Q[i, j])
                        prog << CNOT(qubits[i], qubits[j])
            
            # Mixer unitary
            for i in range(n):
                prog << RX(qubits[i], 2 * beta)
        
        # Measure all qubits
        from pyqpanda import Measure
        for i in range(n):
            prog << Measure(qubits[i], cbits[i])
        
        return prog, qubits, cbits
    
    def solve(self, qubo: QUBOMatrix, gamma: float = 0.5, beta: float = 0.5) -> BackendResult:
        """
        Solve QUBO on Wukong real quantum hardware.
        
        Args:
            qubo: QUBO problem matrix
            gamma: QAOA cost parameter
            beta: QAOA mixer parameter
        
        Returns:
            BackendResult with best solution found
        """
        self._connect()
        prog, qubits, cbits = self._build_qaoa_circuit(qubo, gamma, beta)
        
        t0 = time.time()
        result = self._qm.real_chip_measure(
            prog,
            shot=self.shots,
            chip_id=self.chip_id,
            is_amend=True,
            is_mapping=True,
            is_optimization=True,
            task_name=f"QCLAW_QAOA_{qubo.n_vars}q"
        )
        latency = (time.time() - t0) * 1000
        
        # Find best solution from measurement results
        best_solution = None
        best_energy = float('inf')
        
        if isinstance(result, dict):
            for bitstring, count in result.items():
                sol = np.array([int(b) for b in bitstring], dtype=int)
                if len(sol) == qubo.n_vars:
                    e = qubo.energy(sol)
                    if e < best_energy:
                        best_energy = e
                        best_solution = sol
        
        if best_solution is None:
            best_solution = np.zeros(qubo.n_vars, dtype=int)
            best_energy = qubo.energy(best_solution)
        
        return BackendResult(
            solution=best_solution,
            energy=best_energy,
            raw=result if isinstance(result, dict) else {"raw": str(result)},
            backend=f"originq_wukong_chip{self.chip_id}",
            latency_ms=latency,
            shots=self.shots,
            metadata={"gamma": gamma, "beta": beta, "p_layers": self.p_layers}
        )
    
    def solve_on_simulator(self, qubo: QUBOMatrix, gamma: float = 0.5, beta: float = 0.5) -> BackendResult:
        """Run on OriginQ cloud simulator instead of real hardware."""
        self._connect()
        prog, qubits, cbits = self._build_qaoa_circuit(qubo, gamma, beta)
        
        t0 = time.time()
        result = self._qm.full_amplitude_measure(
            prog, shot=self.shots, task_name=f"QCLAW_SIM_{qubo.n_vars}q"
        )
        latency = (time.time() - t0) * 1000
        
        best_solution = None
        best_energy = float('inf')
        
        if isinstance(result, dict):
            for bitstring, count in result.items():
                sol = np.array([int(b) for b in bitstring], dtype=int)
                if len(sol) == qubo.n_vars:
                    e = qubo.energy(sol)
                    if e < best_energy:
                        best_energy = e
                        best_solution = sol
        
        if best_solution is None:
            best_solution = np.zeros(qubo.n_vars, dtype=int)
            best_energy = qubo.energy(best_solution)
        
        return BackendResult(
            solution=best_solution,
            energy=best_energy,
            raw=result if isinstance(result, dict) else {"raw": str(result)},
            backend="originq_cloud_simulator",
            latency_ms=latency,
            shots=self.shots,
            metadata={"gamma": gamma, "beta": beta, "p_layers": self.p_layers}
        )


class SimulatorBackend:
    """
    Local classical QUBO solver. No quantum hardware needed.
    Uses simulated annealing for fast approximate solutions.
    """
    
    def __init__(self, n_iterations: int = 10000, temp_start: float = 10.0,
                 temp_end: float = 0.01):
        self.n_iterations = n_iterations
        self.temp_start = temp_start
        self.temp_end = temp_end
    
    def solve(self, qubo: QUBOMatrix) -> BackendResult:
        """Solve QUBO via simulated annealing (classical)."""
        n = qubo.n_vars
        t0 = time.time()
        
        # Random initial solution
        current = np.random.randint(0, 2, n)
        current_energy = qubo.energy(current)
        best = current.copy()
        best_energy = current_energy
        
        for i in range(self.n_iterations):
            temp = self.temp_start * (self.temp_end / self.temp_start) ** (i / self.n_iterations)
            
            # Flip a random bit
            flip = np.random.randint(n)
            candidate = current.copy()
            candidate[flip] = 1 - candidate[flip]
            candidate_energy = qubo.energy(candidate)
            
            delta = candidate_energy - current_energy
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current = candidate
                current_energy = candidate_energy
                
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
        
        latency = (time.time() - t0) * 1000
        
        return BackendResult(
            solution=best,
            energy=best_energy,
            raw={"method": "simulated_annealing", "iterations": self.n_iterations},
            backend="local_simulator",
            latency_ms=latency,
            shots=self.n_iterations,
        )
