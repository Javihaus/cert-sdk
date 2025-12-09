"""
CERT SDK Client - Non-blocking LLM trace collection.

Thread-safe client that batches traces and sends them asynchronously.
"""

import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from cert.types import TraceType, ToolCall

__version__ = "0.2.0"


class CertClient:
    """
    Non-blocking client for sending LLM traces to CERT.

    Traces are queued and sent in batches by a background worker thread.
    This ensures trace() calls return immediately without blocking your application.

    Args:
        api_key: Your CERT API key (required)
        project: Project name for grouping traces (default: "default")
        dashboard_url: Custom CERT dashboard URL (default: https://cert-app.vercel.app)
        batch_size: Number of traces to batch before sending (default: 10)
        flush_interval: Seconds between automatic flushes (default: 5.0)
        max_queue_size: Maximum queue size before dropping traces (default: 10000)

    Example:
        >>> client = CertClient(api_key="cert_xxx", project="my-app")
        >>> client.trace(
        ...     type="rag",
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     input_text="What is CERT?",
        ...     output_text="CERT is...",
        ...     duration_ms=1234,
        ...     context="Retrieved documents...",
        ... )
        >>> client.close()  # Flush and stop worker
    """

    DEFAULT_URL = "https://cert-app.vercel.app"
    VALID_TYPES = ("rag", "generation", "agentic")

    def __init__(
        self,
        api_key: str,
        project: str = "default",
        dashboard_url: str = DEFAULT_URL,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000,
    ):
        if not api_key:
            raise ValueError("api_key is required")

        self.api_key = api_key
        self.project = project
        self.endpoint = dashboard_url.rstrip("/") + "/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Queue for traces
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Stats counters (thread-safe via atomic operations)
        self._sent = 0
        self._failed = 0
        self._dropped = 0
        self._lock = threading.Lock()

        # Worker thread
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def __repr__(self) -> str:
        return f"CertClient(project={self.project!r})"

    def __enter__(self) -> "CertClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def trace(
        self,
        *,
        type: TraceType,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        duration_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        context: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        goal_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an LLM trace (non-blocking).

        Args:
            type: Trace type - must be "rag", "generation", or "agentic"
            provider: LLM provider (e.g., "openai", "anthropic")
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus")
            input_text: User prompt or input
            output_text: Model response
            duration_ms: Request duration in milliseconds
            prompt_tokens: Number of input tokens (optional)
            completion_tokens: Number of output tokens (optional)
            context: Retrieved context for RAG traces (optional)
            output_schema: JSON schema for generation traces (optional)
            tool_calls: List of tool calls for agentic traces (optional)
            goal_description: Goal description for agentic traces (optional)
            metadata: Additional metadata dict (optional)

        Returns:
            str: Unique trace ID (UUID)

        Raises:
            ValueError: If type is invalid or tool_calls validation fails
        """
        # Validate type
        if type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}, got {type!r}")

        # Validate tool_calls
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                if "name" not in tc:
                    raise ValueError(f"tool_calls[{i}] missing required 'name' field")
                if not isinstance(tc.get("name"), str):
                    raise ValueError(f"tool_calls[{i}] name must be a string")

        # Build trace payload
        trace_id = str(uuid.uuid4())
        trace_data: Dict[str, Any] = {
            "id": trace_id,
            "type": type,
            "provider": provider,
            "model": model,
            "input": input_text,
            "output": output_text,
            "durationMs": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evalMode": type,  # evalMode matches type
            "projectName": self.project,
        }

        # Add optional fields
        if prompt_tokens is not None:
            trace_data["promptTokens"] = prompt_tokens
        if completion_tokens is not None:
            trace_data["completionTokens"] = completion_tokens
        if context is not None:
            trace_data["context"] = context
        if output_schema is not None:
            trace_data["outputSchema"] = output_schema
        if tool_calls is not None:
            trace_data["toolCalls"] = tool_calls
        if goal_description is not None:
            trace_data["goalDescription"] = goal_description
        if metadata is not None:
            trace_data["metadata"] = metadata

        # Queue trace (non-blocking)
        try:
            self._queue.put_nowait(trace_data)
        except queue.Full:
            with self._lock:
                self._dropped += 1

        return trace_id

    def stats(self) -> Dict[str, int]:
        """
        Get current statistics.

        Returns:
            Dict with traces_sent, traces_failed, traces_dropped, traces_queued
        """
        with self._lock:
            return {
                "traces_sent": self._sent,
                "traces_failed": self._failed,
                "traces_dropped": self._dropped,
                "traces_queued": self._queue.qsize(),
            }

    def close(self, timeout: float = 5.0) -> None:
        """
        Flush remaining traces and stop the worker.

        Args:
            timeout: Maximum seconds to wait for worker to finish
        """
        # Signal worker to stop and flush
        self._stop_event.set()

        # Wait for worker to finish
        self._worker.join(timeout=timeout)

    def _worker_loop(self) -> None:
        """Background worker that batches and sends traces."""
        batch: List[Dict[str, Any]] = []
        last_flush = datetime.now(timezone.utc)

        while True:
            try:
                # Try to get a trace with timeout
                try:
                    trace = self._queue.get(timeout=0.1)
                    batch.append(trace)
                except queue.Empty:
                    pass

                # Check if we should send
                now = datetime.now(timezone.utc)
                elapsed = (now - last_flush).total_seconds()
                should_flush = (
                    len(batch) >= self.batch_size
                    or (batch and elapsed >= self.flush_interval)
                    or (self._stop_event.is_set() and batch)
                )

                if should_flush:
                    self._send_batch(batch)
                    batch = []
                    last_flush = now

                # Exit if stop requested and queue is empty
                if self._stop_event.is_set() and self._queue.empty() and not batch:
                    break

            except Exception:
                # Don't let worker crash
                pass

    def _send_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Send a batch of traces to the API."""
        if not batch:
            return

        try:
            response = requests.post(
                self.endpoint,
                json={"traces": batch},
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=10,
            )

            if response.status_code == 200:
                with self._lock:
                    self._sent += len(batch)
            else:
                with self._lock:
                    self._failed += len(batch)

        except Exception:
            with self._lock:
                self._failed += len(batch)
