"""
Tests for QuantumRouter — AI traffic load balancing.
Simulates AI provider slowdowns and verifies routing behavior.
"""

import time
import random
import pytest
from qclaw.router import QuantumRouter, ProviderStats


# --- Mock providers ---

def make_fast_provider(name: str, latency: float = 0.1):
    """Returns a mock provider that responds quickly."""
    def call_fn(prompt: str) -> str:
        time.sleep(latency)
        return f"[{name}] Response to: {prompt[:30]}..."
    return call_fn


def make_slow_provider(name: str, latency: float = 2.0):
    """Returns a mock provider with high latency."""
    def call_fn(prompt: str) -> str:
        time.sleep(latency)
        return f"[{name}] Slow response"
    return call_fn


def make_failing_provider(name: str, fail_rate: float = 0.8):
    """Returns a mock provider that fails frequently."""
    call_count = {"n": 0}
    def call_fn(prompt: str) -> str:
        call_count["n"] += 1
        if random.random() < fail_rate:
            raise TimeoutError(f"{name} timed out")
        return f"[{name}] Lucky response"
    return call_fn


# --- Tests ---

class TestProviderStats:
    def test_health_score_perfect(self):
        s = ProviderStats(name="test", endpoint="")
        s.record_success(100)
        s.record_success(150)
        assert s.health_score > 0.8

    def test_health_score_zero_when_closed(self):
        s = ProviderStats(name="test", endpoint="")
        s.is_open = False
        assert s.health_score == 0.0

    def test_circuit_breaker_opens(self):
        s = ProviderStats(name="test", endpoint="")
        # Trigger enough failures to open circuit breaker
        for _ in range(10):
            s.record_failure()
        assert not s.is_open
        assert s.error_rate > 0.5

    def test_latency_tracking(self):
        s = ProviderStats(name="test", endpoint="")
        s.record_success(100)
        s.record_success(200)
        s.record_success(300)
        assert 150 < s.avg_latency_ms < 250


class TestQuantumRouter:
    def setup_method(self):
        self.router = QuantumRouter(quantum_noise=0.1)
        self.router.add_provider("fast1", make_fast_provider("fast1", 0.05))
        self.router.add_provider("fast2", make_fast_provider("fast2", 0.05))
        self.router.add_provider("fast3", make_fast_provider("fast3", 0.05))

    def test_basic_routing(self):
        result = self.router.route("Hello, who are you?")
        assert result["success"] is True
        assert result["response"] is not None
        assert result["provider"] in ["fast1", "fast2", "fast3"]
        print(f"\n✅ Routed to: {result['provider']} in {result['latency_ms']}ms")

    def test_routes_multiple_requests(self):
        results = [self.router.route(f"Request {i}") for i in range(10)]
        successes = sum(1 for r in results if r["success"])
        providers_used = set(r["provider"] for r in results if r["success"])
        assert successes == 10
        # Quantum noise should spread across providers
        print(f"\n✅ 10 requests routed to: {providers_used}")

    def test_circuit_breaker_routes_around_failure(self):
        """When a provider fails repeatedly, router should route around it."""
        # Add a failing provider
        self.router.add_provider("flaky", make_failing_provider("flaky", fail_rate=0.9))

        # Force flaky into bad state
        for _ in range(15):
            self.router.providers["flaky"].record_failure()

        # Now route — should avoid flaky
        results = [self.router.route(f"Request {i}", retries=1) for i in range(20)]
        successes = sum(1 for r in results if r["success"])
        flaky_count = sum(1 for r in results if r.get("provider") == "flaky")

        print(f"\n✅ Circuit breaker: {successes}/20 succeeded, flaky used {flaky_count} times")
        assert successes >= 15  # Most should succeed despite flaky provider
        assert not self.router.providers["flaky"].is_open  # Circuit should be open

    def test_weights_avoid_slow_provider(self):
        """Slow provider should get lower routing weight."""
        self.router.add_provider("slow", make_slow_provider("slow", 0.001))
        # Manually mark slow as slow
        for _ in range(10):
            self.router.providers["slow"].record_success(8000)  # 8 second avg

        weights = self.router._quantum_weights()
        print(f"\n✅ Weights after slow provider: {weights}")
        # Slow provider should have lowest weight
        assert weights.get("slow", 1.0) < weights["fast1"]

    def test_all_providers_down_returns_error(self):
        """If all providers fail, route() should return error dict."""
        router = QuantumRouter()
        router.add_provider("bad", make_failing_provider("bad", fail_rate=1.0))

        result = router.route("test", retries=2)
        print(f"\n✅ All-down result: {result}")
        assert result["success"] is False
        assert result["response"] is None

    def test_status_report(self):
        """Status should report all provider health."""
        self.router.route("warmup")
        status = self.router.status()
        print(f"\n✅ Status: {status}")
        assert "providers" in status
        assert "weights" in status
        assert status["total_requests"] >= 1

    def test_simulate_load(self):
        """Simulate load with a provider slowing down mid-test."""
        result = self.router.simulate_load(
            n_requests=50,
            slow_provider="fast1",
            slow_after=20
        )
        print(f"\n✅ Load simulation:")
        print(f"   Route distribution: {result['route_distribution']}")
        print(f"   Avg latency: {result['avg_latency_ms']}ms")
        print(f"   Provider health: {result['final_status']['providers']}")

        # After fast1 slows down, traffic should shift to fast2 and fast3
        total_after_slow = result['route_distribution'].get('fast2', 0) + \
                           result['route_distribution'].get('fast3', 0)
        assert total_after_slow > 0


class TestQuantumNoise:
    def test_high_noise_distributes_evenly(self):
        """High quantum noise should spread traffic evenly regardless of health."""
        router = QuantumRouter(quantum_noise=0.95)
        router.add_provider("a", make_fast_provider("a"))
        router.add_provider("b", make_fast_provider("b"))
        router.add_provider("c", make_fast_provider("c"))

        # Make 'a' look much healthier
        for _ in range(20):
            router.providers["a"].record_success(50)
            router.providers["b"].record_success(2000)
            router.providers["c"].record_success(2000)

        weights = router._quantum_weights()
        print(f"\n✅ High noise weights: {weights}")
        # With 95% noise, weights should be fairly even
        assert max(weights.values()) - min(weights.values()) < 0.5

    def test_low_noise_favors_healthy(self):
        """Low quantum noise should strongly favor healthy providers."""
        router = QuantumRouter(quantum_noise=0.05)
        router.add_provider("healthy", make_fast_provider("healthy"))
        router.add_provider("sick", make_slow_provider("sick"))

        for _ in range(20):
            router.providers["healthy"].record_success(100)
            router.providers["sick"].record_success(9000)

        weights = router._quantum_weights()
        print(f"\n✅ Low noise weights: {weights}")
        assert weights["healthy"] > weights["sick"]


if __name__ == "__main__":
    print("Running QuantumRouter tests...\n")
    pytest.main([__file__, "-v", "-s"])
