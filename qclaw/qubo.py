"""
QUBO Mapper — Convert optimization problems to Quadratic Unconstrained Binary Optimization.
This is the core translation layer: real-world problem → quantum-ready formulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class QUBOMatrix:
    """Represents a QUBO problem as an upper-triangular matrix."""
    
    def __init__(self, Q: np.ndarray, offset: float = 0.0, labels: Optional[List[str]] = None):
        self.Q = Q
        self.offset = offset
        self.n_vars = Q.shape[0]
        self.labels = labels or [f"x{i}" for i in range(self.n_vars)]
    
    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising model (h, J, offset)."""
        n = self.n_vars
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = self.offset
        
        for i in range(n):
            h[i] = self.Q[i, i] / 2
            offset += self.Q[i, i] / 4
            for j in range(i + 1, n):
                J[i, j] = self.Q[i, j] / 4
                h[i] += self.Q[i, j] / 4
                h[j] += self.Q[i, j] / 4
                offset += self.Q[i, j] / 4
        
        return h, J, offset
    
    def energy(self, solution: np.ndarray) -> float:
        """Compute energy of a binary solution vector."""
        return float(solution @ self.Q @ solution + self.offset)
    
    def to_dict(self) -> Dict:
        """Serialize for API transport."""
        return {
            "n_vars": self.n_vars,
            "Q": self.Q.tolist(),
            "offset": self.offset,
            "labels": self.labels,
        }


class QUBOMapper:
    """Maps real-world optimization problems to QUBO formulations."""
    
    @staticmethod
    def from_tsp(distances: np.ndarray, penalty: float = None) -> QUBOMatrix:
        """
        Traveling Salesman Problem → QUBO.
        
        Args:
            distances: NxN distance matrix between cities
            penalty: Constraint penalty (auto-calculated if None)
        
        Returns:
            QUBOMatrix ready for quantum execution
        """
        n = distances.shape[0]
        N = n * n  # Binary variables: x[i,t] = city i at time t
        
        if penalty is None:
            penalty = float(np.max(distances)) * n * 2
        
        Q = np.zeros((N, N))
        
        def idx(city, time):
            return city * n + time
        
        # Objective: minimize total distance
        for t in range(n):
            t_next = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        qi = idx(i, t)
                        qj = idx(j, t_next)
                        Q[qi, qj] += distances[i, j]
        
        # Constraint 1: each city visited exactly once
        for i in range(n):
            for t1 in range(n):
                Q[idx(i, t1), idx(i, t1)] -= penalty
                for t2 in range(t1 + 1, n):
                    Q[idx(i, t1), idx(i, t2)] += 2 * penalty
        
        # Constraint 2: each time slot has exactly one city
        for t in range(n):
            for i1 in range(n):
                Q[idx(i1, t), idx(i1, t)] -= penalty
                for i2 in range(i1 + 1, n):
                    Q[idx(i1, t), idx(i2, t)] += 2 * penalty
        
        # Make upper triangular
        Q = np.triu(Q + Q.T - np.diag(np.diag(Q)))
        
        labels = [f"city{i}_t{t}" for i in range(n) for t in range(n)]
        return QUBOMatrix(Q, offset=2 * n * penalty, labels=labels)
    
    @staticmethod
    def from_maxcut(adjacency: np.ndarray, weights: Optional[np.ndarray] = None) -> QUBOMatrix:
        """
        Maximum Cut Problem → QUBO.
        
        Args:
            adjacency: NxN adjacency matrix (0/1 or weighted)
            weights: Optional edge weights (defaults to adjacency values)
        
        Returns:
            QUBOMatrix
        """
        n = adjacency.shape[0]
        W = weights if weights is not None else adjacency
        Q = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    w = W[i, j]
                    Q[i, i] -= w
                    Q[j, j] -= w
                    Q[i, j] += 2 * w
        
        labels = [f"node{i}" for i in range(n)]
        return QUBOMatrix(Q, labels=labels)
    
    @staticmethod
    def from_portfolio(returns: np.ndarray, covariance: np.ndarray, 
                       risk_factor: float = 0.5, budget: int = None) -> QUBOMatrix:
        """
        Portfolio Optimization → QUBO.
        Select k assets from n to maximize return while minimizing risk.
        
        Args:
            returns: Expected returns vector (n,)
            covariance: Covariance matrix (n, n)
            risk_factor: Trade-off between return and risk [0, 1]
            budget: Number of assets to select (default: n//2)
        """
        n = len(returns)
        budget = budget or n // 2
        penalty = float(np.max(np.abs(returns))) * n
        
        Q = np.zeros((n, n))
        
        # Maximize returns (minimize negative returns)
        for i in range(n):
            Q[i, i] -= (1 - risk_factor) * returns[i]
        
        # Minimize risk
        Q += risk_factor * covariance
        
        # Budget constraint: select exactly k assets
        for i in range(n):
            Q[i, i] += penalty * (1 - 2 * budget)
            for j in range(i + 1, n):
                Q[i, j] += 2 * penalty
        
        Q = np.triu(Q + Q.T - np.diag(np.diag(Q)))
        
        labels = [f"asset{i}" for i in range(n)]
        return QUBOMatrix(Q, offset=penalty * budget * budget, labels=labels)
    
    @staticmethod
    def from_job_schedule(durations: List[int], deadlines: List[int],
                          n_machines: int, penalty: float = None) -> QUBOMatrix:
        """
        Job Shop Scheduling → QUBO.
        Assign n jobs to m machines minimizing makespan.
        
        Args:
            durations: Job duration list
            deadlines: Job deadline list
            n_machines: Number of machines
        """
        n_jobs = len(durations)
        max_time = max(deadlines) + 1
        N = n_jobs * n_machines  # x[j,m] = job j assigned to machine m
        
        if penalty is None:
            penalty = float(max(durations)) * n_jobs
        
        Q = np.zeros((N, N))
        
        def idx(job, machine):
            return job * n_machines + machine
        
        # Constraint: each job assigned to exactly one machine
        for j in range(n_jobs):
            for m1 in range(n_machines):
                Q[idx(j, m1), idx(j, m1)] -= penalty
                for m2 in range(m1 + 1, n_machines):
                    Q[idx(j, m1), idx(j, m2)] += 2 * penalty
        
        # Objective: balance load across machines (minimize max load)
        for m in range(n_machines):
            for j1 in range(n_jobs):
                for j2 in range(j1 + 1, n_jobs):
                    Q[idx(j1, m), idx(j2, m)] += durations[j1] * durations[j2]
        
        Q = np.triu(Q + Q.T - np.diag(np.diag(Q)))
        
        labels = [f"job{j}_m{m}" for j in range(n_jobs) for m in range(n_machines)]
        return QUBOMatrix(Q, offset=n_jobs * penalty, labels=labels)
