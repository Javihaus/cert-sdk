"""
CERT SDK - Python client for LLM monitoring.

Simple, async, non-blocking tracer for production applications.
"""

import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from cert.types import EvalMode, ToolCall

import requests

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _validate_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """
    Validate tool_calls structure before sending.

    Args:
        tool_calls: List of tool call dictionaries

    Raises:
        ValueError: If tool_calls structure is invalid
    """
    for i, tc in enumerate(tool_calls):
        if "name" not in tc:
            raise ValueError(f"tool_calls[{i}] missing required 'name' field")
        if not isinstance(tc.get("name"), str):
            raise ValueError(f"tool_calls[{i}].name must be a string")


class CertClient:
    """
    Non-blocking client for CERT dashboard.

    Traces are queued and sent in batches via background thread.
    Never blocks your application.

    Example:
        >>> client = CertClient(api_key="cert_xxx")
        >>> client.trace(
        ...     provider="anthropic",
        ...     model="claude-sonnet-4",
        ...     input_text="Hello",
        ...     output_text="Hi there!",
        ...     duration_ms=234,
        ...     prompt_tokens=10,
        ...     completion_tokens=15
        ... )
    """

    def __init__(
        self,
        api_key: str,
        dashboard_url: str = "https://dashboard.cert-framework.com",
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        timeout: float = 5.0,
    ):
        """
        Initialize CERT client.

        Args:
            api_key: Your CERT API key from dashboard
            dashboard_url: Dashboard URL (default: production)
            batch_size: Traces per batch (default: 10)
            flush_interval: Seconds between flushes (default: 5.0)
            max_queue_size: Max traces to queue (default: 1000)
            timeout: HTTP timeout in seconds (default: 5.0)
        """
        self.api_key = api_key
        self.endpoint = f"{dashboard_url.rstrip('/')}/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._traces_sent = 0
        self._traces_failed = 0

        self._start_worker()

    def trace(
        self,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        duration_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        eval_mode: EvalMode = "auto",
        context: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        goal_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an LLM trace. Non-blocking.

        Args:
            provider: LLM provider (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-sonnet-4")
            input_text: Input prompt/messages
            output_text: Model response
            duration_ms: Request duration in milliseconds
            prompt_tokens: Input token count
            completion_tokens: Output token count
            eval_mode: Evaluation mode - "rag", "generation", "agentic", or "auto"
                       (auto-detects based on context and tool_calls)
            context: RAG context/retrieved documents (implies "rag" mode)
            output_schema: Expected output structure for generation validation
            tool_calls: List of tool/function calls (implies "agentic" mode)
            goal_description: Description of the agentic goal
            metadata: Optional additional metadata
        """
        # Validate tool_calls structure if provided
        if tool_calls is not None:
            _validate_tool_calls(tool_calls)

        # Resolve eval_mode if set to "auto"
        resolved_mode: str = eval_mode
        if eval_mode == "auto":
            if tool_calls is not None and len(tool_calls) > 0:
                resolved_mode = "agentic"
            elif context is not None and len(context.strip()) > 0:
                resolved_mode = "rag"
            else:
                resolved_mode = "generation"

        # Build base payload
        trace_data: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "provider": provider,
            "model": model,
            "input": input_text,
            "output": output_text,
            "durationMs": duration_ms,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evalMode": resolved_mode,
            "metadata": metadata or {},
        }

        # Only include mode-specific fields if present
        if context is not None:
            trace_data["context"] = context
        if output_schema is not None:
            trace_data["outputSchema"] = output_schema
        if tool_calls is not None:
            trace_data["toolCalls"] = tool_calls
        if goal_description is not None:
            trace_data["goalDescription"] = goal_description

        try:
            self._queue.put_nowait(trace_data)
        except queue.Full:
            self._traces_failed += 1
            logger.warning("CERT: Trace queue full, dropping trace")

    def flush(self, timeout: float = 10.0) -> None:
        """
        Flush all pending traces. Blocks until complete or timeout.

        Call before application exit to ensure traces are sent.

        Args:
            timeout: Maximum seconds to wait
        """
        deadline = time.time() + timeout

        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.1)

        if not self._queue.empty():
            logger.warning(f"CERT: {self._queue.qsize()} traces remaining after flush timeout")

    def close(self) -> None:
        """
        Stop background worker and flush pending traces.

        Call when shutting down your application.
        """
        self.flush()
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=5.0)

    def get_stats(self) -> Dict[str, int]:
        """Get client statistics."""
        return {
            "traces_sent": self._traces_sent,
            "traces_failed": self._traces_failed,
            "traces_queued": self._queue.qsize(),
        }

    def _start_worker(self) -> None:
        """Start background worker thread."""
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        logger.debug("CERT: Background worker started")

    def _worker_loop(self) -> None:
        """Background worker that sends batches."""
        batch = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Calculate timeout until next flush
                time_until_flush = self.flush_interval - (time.time() - last_flush)
                timeout = max(0.1, min(1.0, time_until_flush))

                # Get trace from queue
                trace = self._queue.get(timeout=timeout)
                batch.append(trace)

                # Send if batch full
                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

            except queue.Empty:
                # Flush if interval elapsed
                if batch and (time.time() - last_flush) >= self.flush_interval:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

        # Send remaining traces on shutdown
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: list) -> None:
        """Send batch of traces to dashboard."""
        try:
            response = requests.post(
                self.endpoint,
                json={"traces": batch},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                self._traces_sent += len(batch)
                logger.debug(f"CERT: Sent {len(batch)} traces")
            else:
                self._traces_failed += len(batch)
                logger.warning(
                    f"CERT: Failed to send {len(batch)} traces: "
                    f"HTTP {response.status_code}"
                )

        except requests.exceptions.Timeout:
            self._traces_failed += len(batch)
            logger.warning(f"CERT: Timeout sending {len(batch)} traces")
        except Exception as e:
            self._traces_failed += len(batch)
            logger.warning(f"CERT: Exception sending {len(batch)} traces: {e}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
        return False
