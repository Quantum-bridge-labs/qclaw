"""
Q-CLAW Agent — The main interface.
Takes natural language or structured problems, routes to quantum or classical.
"""

import time
import json
import numpy as np
from typing import Dict, Optional, Union
from .qubo import QUBOMapper, QUBOMatrix
from .backend import OriginQBackend, SimulatorBackend, BackendResult
from .problems import TSP, MaxCut, PortfolioOptimizer, JobScheduler


class Agent:
    """
    Q-CLAW Agent — Route optimization problems to quantum hardware.
    
    Usage:
        agent = qclaw.Agent(qpu_provider="origin_wukong")
        result = agent.solve(problem)
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 qpu_provider: str = "origin_wukong",
                 fallback: str = "simulator",
                 chip_id: int = 2,
                 shots: int = 1000,
                 p_layers: int = 1):
        """
        Initialize Q-CLAW Agent.
        
        Args:
            api_key: OriginQ API key (auto-loaded from env if None)
            qpu_provider: "origin_wukong" or "simulator"
            fallback: Fall back to "simulator" if QPU unavailable
            chip_id: OriginQ chip (2=Wukong 72-qubit)
            shots: Measurement shots per execution
            p_layers: QAOA circuit depth
        """
        self.qpu_provider = qpu_provider
        self.fallback = fallback
        self.shots = shots
        
        if qpu_provider == "origin_wukong":
            self.backend = OriginQBackend(
                api_key=api_key, chip_id=chip_id, 
                shots=shots, p_layers=p_layers
            )
        else:
            self.backend = SimulatorBackend()
        
        self.sim_backend = SimulatorBackend()
        self._history = []
    
    def solve(self, problem: Union[TSP, MaxCut, PortfolioOptimizer, JobScheduler],
              use_hardware: bool = True) -> Dict:
        """
        Solve an optimization problem.
        
        Args:
            problem: A Q-CLAW problem instance
            use_hardware: If True, attempt real QPU first
        
        Returns:
            Interpreted results dict with solution + metadata
        """
        t0 = time.time()
        qubo = problem.qubo
        
        # Check qubit budget
        if qubo.n_vars > 72 and use_hardware:
            print(f"  Problem needs {qubo.n_vars} qubits (Wukong max: 72). Using hybrid approach.")
            use_hardware = False
        
        result = None
        backend_used = None
        
        if use_hardware and isinstance(self.backend, OriginQBackend):
            try:
                result = self.backend.solve(qubo)
                backend_used = "wukong"
            except Exception as e:
                err = str(e).lower()
                if "maintenance" in err:
                    print(f"  Wukong in maintenance. Falling back to {self.fallback}.")
                else:
                    print(f"  QPU error: {e}. Falling back.")
                
                if self.fallback == "simulator":
                    result = self.sim_backend.solve(qubo)
                    backend_used = "simulator_fallback"
        
        if result is None:
            result = self.sim_backend.solve(qubo)
            backend_used = "simulator"
        
        total_ms = (time.time() - t0) * 1000
        
        # Interpret through the problem's decoder
        interpreted = problem.interpret(result)
        interpreted["total_latency_ms"] = round(total_ms, 2)
        interpreted["backend_used"] = backend_used
        
        # Log
        self._history.append({
            "problem_type": type(problem).__name__,
            "n_vars": qubo.n_vars,
            "backend": backend_used,
            "energy": result.energy,
            "latency_ms": total_ms,
            "timestamp": time.time(),
        })
        
        return interpreted
    
    def ping(self, problem: Union[TSP, MaxCut, PortfolioOptimizer, JobScheduler],
             mode: str = "auto") -> Dict:
        """
        Quick solve — the API-friendly interface.
        
        Args:
            problem: Optimization problem
            mode: "quantum" (force QPU), "classical" (force sim), "auto" (best effort)
        """
        use_hw = mode != "classical"
        return self.solve(problem, use_hardware=use_hw)
    
    def benchmark(self, problem: Union[TSP, MaxCut, PortfolioOptimizer, JobScheduler]) -> Dict:
        """
        Run on both quantum and classical, compare results.
        Useful for demos and proving quantum advantage.
        """
        # Classical
        classical = self.sim_backend.solve(problem.qubo)
        classical_interpreted = problem.interpret(classical)
        
        # Quantum (if available)
        quantum = None
        quantum_interpreted = None
        if isinstance(self.backend, OriginQBackend):
            try:
                quantum = self.backend.solve(problem.qubo)
                quantum_interpreted = problem.interpret(quantum)
            except Exception as e:
                quantum_interpreted = {"error": str(e)}
        
        return {
            "classical": {
                **classical_interpreted,
                "energy": classical.energy,
            },
            "quantum": quantum_interpreted,
            "speedup": (
                round(classical.latency_ms / quantum.latency_ms, 2)
                if quantum and quantum.latency_ms > 0 else None
            ),
        }
    
    @property 
    def history(self) -> list:
        """Get execution history."""
        return self._history
    
    @property
    def stats(self) -> Dict:
        """Get aggregate stats."""
        if not self._history:
            return {"total_pings": 0}
        return {
            "total_pings": len(self._history),
            "avg_latency_ms": round(np.mean([h["latency_ms"] for h in self._history]), 2),
            "qpu_pings": sum(1 for h in self._history if "wukong" in h["backend"]),
            "sim_pings": sum(1 for h in self._history if "simulator" in h["backend"]),
            "problems_solved": {
                t: sum(1 for h in self._history if h["problem_type"] == t)
                for t in set(h["problem_type"] for h in self._history)
            }
        }
