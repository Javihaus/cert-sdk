"""
CERT SDK Client - Non-blocking LLM trace collection.

Thread-safe client that batches traces and sends them asynchronously.
"""

import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from cert.types import ToolCall, TraceType

__version__ = "0.2.0"

logger = logging.getLogger("cert")


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
        max_queue_size: Maximum queue size before dropping traces (default: 1000)
        timeout: HTTP request timeout in seconds (default: 10.0)

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
        max_queue_size: int = 1000,
        timeout: float = 10.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")

        self.api_key = api_key
        self.project = project
        self.endpoint = dashboard_url.rstrip("/") + "/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        # Queue for traces
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)

        # Stats counters (thread-safe via lock)
        self._sent = 0
        self._failed = 0
        self._dropped = 0
        self._lock = threading.Lock()

        # Flush event for synchronous flush
        self._flush_event = threading.Event()
        self._flush_complete = threading.Event()

        # Worker thread
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        logger.debug("CertClient initialized for project %r", project)

    def __repr__(self) -> str:
        return f"CertClient(project={self.project!r})"

    def __enter__(self) -> "CertClient":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
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
            logger.debug("Queued trace %s", trace_id)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            logger.warning("Queue full, dropped trace %s", trace_id)

        return trace_id

    def flush(self, timeout: float = 10.0) -> int:
        """
        Force-send all pending traces.

        Blocks until all queued traces are sent or timeout expires.

        Args:
            timeout: Maximum seconds to wait (default: 10.0)

        Returns:
            int: Number of traces sent during this flush
        """
        if self._stop_event.is_set():
            return 0

        with self._lock:
            sent_before = self._sent

        # Signal worker to flush
        self._flush_complete.clear()
        self._flush_event.set()

        # Wait for flush to complete
        self._flush_complete.wait(timeout=timeout)

        with self._lock:
            sent_after = self._sent
            return sent_after - sent_before

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
        logger.debug("Closing CertClient")
        # Signal worker to stop and flush
        self._stop_event.set()

        # Wait for worker to finish
        self._worker.join(timeout=timeout)
        logger.debug("CertClient closed")

    def _worker_loop(self) -> None:
        """Background worker that batches and sends traces."""
        batch: List[Dict[str, Any]] = []
        last_flush = datetime.now(timezone.utc)

        while True:
            try:
                # Check for flush request
                if self._flush_event.is_set():
                    # Drain queue and send everything
                    while not self._queue.empty():
                        try:
                            trace = self._queue.get_nowait()
                            batch.append(trace)
                        except queue.Empty:
                            break
                    if batch:
                        self._send_batch(batch)
                        batch = []
                        last_flush = datetime.now(timezone.utc)
                    self._flush_event.clear()
                    self._flush_complete.set()

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

            except Exception as e:
                # Don't let worker crash
                logger.exception("Worker error: %s", e)

    def _send_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Send a batch of traces to the API."""
        if not batch:
            return

        logger.debug("Sending batch of %d traces", len(batch))

        try:
            response = requests.post(
                self.endpoint,
                json={"traces": batch},
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                with self._lock:
                    self._sent += len(batch)
                logger.debug("Successfully sent %d traces", len(batch))
            else:
                with self._lock:
                    self._failed += len(batch)
                logger.warning(
                    "Failed to send traces: HTTP %d - %s",
                    response.status_code,
                    response.text[:200],
                )

        except requests.exceptions.Timeout:
            with self._lock:
                self._failed += len(batch)
            logger.warning("Timeout sending %d traces", len(batch))
        except Exception as e:
            with self._lock:
                self._failed += len(batch)
            logger.warning("Error sending traces: %s", e)
