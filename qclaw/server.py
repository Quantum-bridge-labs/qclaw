"""
Q-CLAW API Server — Pay-per-ping quantum optimization.
"""

import time
import json
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict

# Lazy import to avoid loading pyqpanda on import
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from .agent import Agent
        _agent = Agent(qpu_provider="origin_wukong", fallback="simulator")
    return _agent


# Simple API key auth (production would use proper auth)
API_KEYS_FILE = os.path.expanduser("~/.openclaw/.qclaw_api_keys.json")

def load_api_keys() -> Dict:
    try:
        with open(API_KEYS_FILE) as f:
            return json.load(f)
    except:
        return {"demo": {"credits": 100, "name": "Demo User"}}

def save_api_keys(keys: Dict):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


# Pricing
PRICING = {
    "tsp": 5.00,
    "maxcut": 5.00,
    "portfolio": 10.00,
    "job_scheduler": 5.00,
}


class QClawHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        pass  # Quiet
    
    def _send(self, code: int, data: Dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Powered-By", "Q-CLAW/0.1")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self.end_headers()
        self.wfile.write(body)
    
    def _auth(self) -> tuple:
        key = self.headers.get("X-API-Key", "")
        keys = load_api_keys()
        if key not in keys:
            return None, None
        return key, keys[key]
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/":
            self._send(200, {
                "service": "Q-CLAW",
                "version": "0.1.0",
                "description": "Quantum-Classical Logic & Action Wrapper",
                "endpoints": {
                    "POST /ping/tsp": "Traveling Salesman ($5/ping)",
                    "POST /ping/maxcut": "Max Cut ($5/ping)",
                    "POST /ping/portfolio": "Portfolio Optimization ($10/ping)",
                    "POST /ping/job_scheduler": "Job Scheduling ($5/ping)",
                    "GET /status": "Backend status",
                    "GET /pricing": "Current pricing",
                },
                "docs": "https://qclaw.gpupulse.dev/docs"
            })
        
        elif self.path == "/status":
            agent = get_agent()
            self._send(200, {
                "backend": agent.qpu_provider,
                "fallback": agent.fallback,
                "stats": agent.stats,
                "wukong": "checking...",
            })
        
        elif self.path == "/pricing":
            self._send(200, {
                "pricing": PRICING,
                "currency": "USD",
                "payment_methods": ["USDC (Solana)", "GPUPULSE token"],
                "note": "Credits deducted per ping. Quantum hardware pings may cost more."
            })
        
        else:
            self._send(404, {"error": "Not found"})
    
    def do_POST(self):
        if not self.path.startswith("/ping/"):
            self._send(404, {"error": "Not found"})
            return
        
        # Auth
        api_key, user = self._auth()
        if not user:
            self._send(401, {"error": "Invalid API key. Header: X-API-Key"})
            return
        
        # Check credits
        problem_type = self.path.replace("/ping/", "")
        cost = PRICING.get(problem_type)
        if cost is None:
            self._send(400, {"error": f"Unknown problem type: {problem_type}", "available": list(PRICING.keys())})
            return
        
        if user.get("credits", 0) < cost:
            self._send(402, {"error": "Insufficient credits", "balance": user["credits"], "cost": cost})
            return
        
        # Parse body
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except:
            self._send(400, {"error": "Invalid JSON body"})
            return
        
        # Build problem
        try:
            agent = get_agent()
            mode = body.get("mode", "auto")
            
            if problem_type == "tsp":
                if "coordinates" in body:
                    problem = _build_tsp_coords(body["coordinates"])
                elif "distances" in body:
                    cities = body.get("cities", [f"city_{i}" for i in range(len(body["distances"]))])
                    problem = _build_tsp_matrix(cities, body["distances"])
                else:
                    self._send(400, {"error": "TSP needs 'coordinates' or 'distances'"})
                    return
            
            elif problem_type == "maxcut":
                problem = _build_maxcut(body)
            
            elif problem_type == "portfolio":
                problem = _build_portfolio(body)
            
            elif problem_type == "job_scheduler":
                problem = _build_scheduler(body)
            
            else:
                self._send(400, {"error": f"Unknown: {problem_type}"})
                return
            
            # Solve
            result = agent.ping(problem, mode=mode)
            
            # Deduct credits
            keys = load_api_keys()
            keys[api_key]["credits"] -= cost
            save_api_keys(keys)
            
            result["cost_usd"] = cost
            result["credits_remaining"] = keys[api_key]["credits"]
            
            self._send(200, result)
        
        except Exception as e:
            self._send(500, {"error": str(e)})


def _build_tsp_coords(coords: Dict) -> "TSP":
    from .problems import TSP
    return TSP.from_coordinates(coords)

def _build_tsp_matrix(cities, distances) -> "TSP":
    from .problems import TSP
    return TSP(cities, np.array(distances))

def _build_maxcut(body: Dict) -> "MaxCut":
    from .problems import MaxCut
    adj = np.array(body["adjacency"])
    labels = body.get("labels")
    return MaxCut(adj, node_labels=labels)

def _build_portfolio(body: Dict) -> "PortfolioOptimizer":
    from .problems import PortfolioOptimizer
    return PortfolioOptimizer(
        assets=body["assets"],
        returns=np.array(body["returns"]),
        covariance=np.array(body["covariance"]),
        risk_factor=body.get("risk_factor", 0.5),
        budget=body.get("budget"),
    )

def _build_scheduler(body: Dict) -> "JobScheduler":
    from .problems import JobScheduler
    return JobScheduler(
        jobs=body["jobs"],
        durations=body["durations"],
        deadlines=body["deadlines"],
        n_machines=body.get("n_machines", 2),
    )


def serve(port: int = 3850, host: str = "127.0.0.1"):
    """Start Q-CLAW API server."""
    server = HTTPServer((host, port), QClawHandler)
    print(f"[Q-CLAW] API server live on {host}:{port}")
    print(f"[Q-CLAW] Pricing: {json.dumps(PRICING)}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
