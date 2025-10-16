"""
Performance metrics tracking for observability.
Tracks latency, token usage, and other performance indicators.
"""

import time
from typing import Any
from contextvars import ContextVar

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Context variable for metrics storage (thread-safe)
metrics_ctx: ContextVar[dict[str, Any]] = ContextVar("metrics", default=None)


class MetricsTracker:
    """
    Tracks performance metrics for a conversation or request.
    Provides easy timing and metric collection.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            "node_timings": {},
            "tokens": {"input": 0, "output": 0, "total": 0},
            "retrieval_stats": {"queries": 0, "documents_retrieved": 0},
            "total_time": 0.0,
            "start_time": time.time(),
        }
        metrics_ctx.set(self.metrics)

    def start_node(self, node_name: str) -> None:
        """
        Start timing a graph node.

        Args:
            node_name: Name of the node being timed
        """
        if node_name not in self.metrics["node_timings"]:
            self.metrics["node_timings"][node_name] = []

        self.metrics["node_timings"][node_name].append(
            {"start": time.time(), "end": None}
        )

    def end_node(self, node_name: str) -> float:
        """
        End timing a graph node and return elapsed time.

        Args:
            node_name: Name of the node being timed

        Returns:
            Elapsed time in seconds

        Raises:
            ValueError: If node was not started
        """
        if node_name not in self.metrics["node_timings"]:
            raise ValueError(f"Node '{node_name}' was not started")

        timings = self.metrics["node_timings"][node_name]
        if not timings or timings[-1]["end"] is not None:
            raise ValueError(f"No active timing for node '{node_name}'")

        timings[-1]["end"] = time.time()
        elapsed = timings[-1]["end"] - timings[-1]["start"]

        logger.info(
            "node_completed",
            node=node_name,
            elapsed=elapsed,
        )

        return elapsed

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """
        Add token usage to metrics.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.metrics["tokens"]["input"] += input_tokens
        self.metrics["tokens"]["output"] += output_tokens
        self.metrics["tokens"]["total"] += input_tokens + output_tokens

    def add_retrieval(self, num_documents: int) -> None:
        """
        Record a retrieval operation.

        Args:
            num_documents: Number of documents retrieved
        """
        self.metrics["retrieval_stats"]["queries"] += 1
        self.metrics["retrieval_stats"]["documents_retrieved"] += num_documents

    def finalize(self) -> dict[str, Any]:
        """
        Finalize metrics and calculate totals.

        Returns:
            Dictionary with all collected metrics
        """
        self.metrics["total_time"] = time.time() - self.metrics["start_time"]

        # Calculate total time per node
        node_summary = {}
        for node_name, timings in self.metrics["node_timings"].items():
            total_time = sum(
                t["end"] - t["start"] for t in timings if t["end"] is not None
            )
            node_summary[node_name] = {
                "total_time": total_time,
                "call_count": len(timings),
            }

        self.metrics["node_summary"] = node_summary

        # Log final metrics
        logger.info(
            "conversation_metrics",
            total_time=self.metrics["total_time"],
            total_tokens=self.metrics["tokens"]["total"],
            node_summary=node_summary,
        )

        return self.metrics


def get_metrics() -> dict[str, Any] | None:
    """
    Get current metrics from context.

    Returns:
        Current metrics dictionary or None
    """
    return metrics_ctx.get()


def start_node_timing(node_name: str) -> None:
    """
    Convenience function to start timing a node.

    Args:
        node_name: Name of the node
    """
    metrics = metrics_ctx.get()
    if metrics and "node_timings" in metrics:
        if node_name not in metrics["node_timings"]:
            metrics["node_timings"][node_name] = []
        metrics["node_timings"][node_name].append(
            {"start": time.time(), "end": None}
        )


def end_node_timing(node_name: str) -> float | None:
    """
    Convenience function to end timing a node.

    Args:
        node_name: Name of the node

    Returns:
        Elapsed time or None if metrics not initialized
    """
    metrics = metrics_ctx.get()
    if not metrics or node_name not in metrics.get("node_timings", {}):
        return None

    timings = metrics["node_timings"][node_name]
    if not timings or timings[-1]["end"] is not None:
        return None

    timings[-1]["end"] = time.time()
    elapsed = timings[-1]["end"] - timings[-1]["start"]

    logger.info("node_completed", node=node_name, elapsed=elapsed)

    return elapsed

