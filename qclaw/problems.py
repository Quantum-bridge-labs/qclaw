"""
Pre-built optimization problems — the use cases that sell Q-CLAW.
Each one maps a real-world problem to QUBO and interprets results.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .qubo import QUBOMapper, QUBOMatrix
from .backend import BackendResult


class TSP:
    """
    Traveling Salesman Problem solver.
    Find the shortest route visiting all cities exactly once.
    
    Use cases: logistics, delivery routing, supply chain, field service
    """
    
    def __init__(self, cities: List[str], distances: np.ndarray):
        if len(cities) != distances.shape[0]:
            raise ValueError("City count must match distance matrix dimensions")
        self.cities = cities
        self.distances = distances
        self.n = len(cities)
        self.qubo = QUBOMapper.from_tsp(distances)
    
    @classmethod
    def from_coordinates(cls, city_coords: Dict[str, Tuple[float, float]]) -> "TSP":
        """Create TSP from city name → (lat, lon) mapping."""
        cities = list(city_coords.keys())
        n = len(cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = city_coords[cities[i]]
                    lat2, lon2 = city_coords[cities[j]]
                    # Haversine
                    R = 6371
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    distances[i, j] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return cls(cities, distances)
    
    def interpret(self, result: BackendResult) -> Dict:
        """Convert raw quantum result to human-readable route."""
        sol = result.solution
        route = []
        
        for t in range(self.n):
            for i in range(self.n):
                if sol[i * self.n + t] == 1:
                    route.append(self.cities[i])
                    break
        
        # Calculate total distance
        total_dist = 0
        for t in range(len(route)):
            i = self.cities.index(route[t])
            j = self.cities.index(route[(t + 1) % len(route)])
            total_dist += self.distances[i, j]
        
        return {
            "route": route,
            "total_distance": round(total_dist, 2),
            "n_cities": self.n,
            "qubits_used": self.n * self.n,
            "backend": result.backend,
            "latency_ms": result.latency_ms,
        }


class MaxCut:
    """
    Maximum Cut solver.
    Partition graph nodes into two sets maximizing edges between them.
    
    Use cases: network design, VLSI layout, social network analysis, clustering
    """
    
    def __init__(self, adjacency: np.ndarray, node_labels: Optional[List[str]] = None):
        self.adjacency = adjacency
        self.n = adjacency.shape[0]
        self.labels = node_labels or [f"node_{i}" for i in range(self.n)]
        self.qubo = QUBOMapper.from_maxcut(adjacency)
    
    def interpret(self, result: BackendResult) -> Dict:
        sol = result.solution
        set_a = [self.labels[i] for i in range(self.n) if sol[i] == 0]
        set_b = [self.labels[i] for i in range(self.n) if sol[i] == 1]
        
        cut_value = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adjacency[i, j] and sol[i] != sol[j]:
                    cut_value += self.adjacency[i, j]
        
        return {
            "set_a": set_a,
            "set_b": set_b,
            "cut_value": float(cut_value),
            "n_nodes": self.n,
            "qubits_used": self.n,
            "backend": result.backend,
            "latency_ms": result.latency_ms,
        }


class PortfolioOptimizer:
    """
    Quantum Portfolio Optimization.
    Select optimal asset allocation balancing return vs risk.
    
    Use cases: hedge funds, wealth management, ETF construction
    """
    
    def __init__(self, assets: List[str], returns: np.ndarray,
                 covariance: np.ndarray, risk_factor: float = 0.5,
                 budget: int = None):
        self.assets = assets
        self.returns = returns
        self.covariance = covariance
        self.risk_factor = risk_factor
        self.budget = budget or len(assets) // 2
        self.qubo = QUBOMapper.from_portfolio(returns, covariance, risk_factor, self.budget)
    
    def interpret(self, result: BackendResult) -> Dict:
        sol = result.solution
        selected = [self.assets[i] for i in range(len(self.assets)) if sol[i] == 1]
        
        expected_return = sum(self.returns[i] for i in range(len(self.assets)) if sol[i] == 1)
        
        sel_idx = [i for i in range(len(self.assets)) if sol[i] == 1]
        risk = 0
        for i in sel_idx:
            for j in sel_idx:
                risk += self.covariance[i, j]
        
        return {
            "selected_assets": selected,
            "expected_return": round(float(expected_return), 4),
            "portfolio_risk": round(float(risk), 4),
            "n_selected": len(selected),
            "budget": self.budget,
            "qubits_used": len(self.assets),
            "backend": result.backend,
            "latency_ms": result.latency_ms,
        }


class JobScheduler:
    """
    Quantum Job Scheduling.
    Optimally assign jobs to machines minimizing total completion time.
    
    Use cases: cloud compute scheduling, manufacturing, GPU workload balancing
    """
    
    def __init__(self, jobs: List[str], durations: List[int],
                 deadlines: List[int], n_machines: int):
        self.jobs = jobs
        self.durations = durations
        self.deadlines = deadlines
        self.n_machines = n_machines
        self.qubo = QUBOMapper.from_job_schedule(durations, deadlines, n_machines)
    
    def interpret(self, result: BackendResult) -> Dict:
        sol = result.solution
        n_jobs = len(self.jobs)
        assignment = {}
        machine_loads = {m: [] for m in range(self.n_machines)}
        
        for j in range(n_jobs):
            for m in range(self.n_machines):
                if sol[j * self.n_machines + m] == 1:
                    assignment[self.jobs[j]] = f"machine_{m}"
                    machine_loads[m].append({
                        "job": self.jobs[j],
                        "duration": self.durations[j],
                        "deadline": self.deadlines[j]
                    })
                    break
        
        makespan = max(
            sum(j["duration"] for j in jobs) 
            for jobs in machine_loads.values()
        ) if any(machine_loads.values()) else 0
        
        return {
            "assignment": assignment,
            "machine_loads": {f"machine_{m}": jobs for m, jobs in machine_loads.items()},
            "makespan": makespan,
            "n_jobs": n_jobs,
            "n_machines": self.n_machines,
            "qubits_used": n_jobs * self.n_machines,
            "backend": result.backend,
            "latency_ms": result.latency_ms,
        }
