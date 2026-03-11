"""
Q-CLAW — Quantum-Classical Logic & Action Wrapper
==================================================
Route reasoning to LLMs. Route optimization to QPUs.
The quantum layer for agentic infrastructure.

(c) 2026 GPUPulse / Q-CLAW
"""

__version__ = "0.1.0"

from .agent import Agent
from .qubo import QUBOMapper
from .backend import OriginQBackend, QiskitAerBackend, SimulatorBackend
from .problems import TSP, MaxCut, PortfolioOptimizer, JobScheduler

__all__ = [
    "Agent",
    "QUBOMapper", 
    "OriginQBackend",
    "QiskitAerBackend",
    "SimulatorBackend",
    "TSP",
    "MaxCut",
    "PortfolioOptimizer",
    "JobScheduler",
]
