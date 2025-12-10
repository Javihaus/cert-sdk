"""
CERT SDK - Python client for LLM monitoring.

Simple, async, non-blocking tracer for production applications.
"""

import json
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


def extract_context_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Extract context string from tool call outputs.
    
    This is the c_agent = ⊕_i o_i operation from the CERT framework.
    Tool outputs constitute the implicit context that enables full
    RAG-mode metrics (SGI, grounding, NLI) in agentic evaluation.
    
    Args:
        tool_calls: List of tool call dictionaries with 'name', 'output', 'error' keys
        
    Returns:
        Concatenated context string from all tool outputs
        
    Example:
        >>> tool_calls = [
        ...     {"name": "search", "output": {"results": ["doc1", "doc2"]}},
        ...     {"name": "calculate", "output": 42}
        ... ]
        >>> context = extract_context_from_tool_calls(tool_calls)
        >>> print(context)
        [search]: {"results": ["doc1", "doc2"]}
        
        [calculate]: 42
    """
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "unknown_tool")
        
        if tc.get("error"):
            # Include errors - they're part of what the agent saw
            parts.append(f"[{name}] ERROR: {tc['error']}")
        elif tc.get("output") is not None:
            output = tc["output"]
            # Serialize to string if needed
            if isinstance(output, (dict, list)):
                output_str = json.dumps(output, ensure_ascii=False, default=str)
            else:
                output_str = str(output)
            parts.append(f"[{name}]: {output_str}")
        # Skip tool calls with no output and no error (pending/incomplete)
    
    return "\n\n".join(parts)


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
        
    Agentic Mode with Automatic Context:
        >>> # Tool outputs automatically become context
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     input_text="What's the weather?",
        ...     output_text="It's 72°F and sunny.",
        ...     duration_ms=1500,
        ...     eval_mode="agentic",  # or "auto"
        ...     tool_calls=[
        ...         {"name": "weather_api", "input": {"city": "NYC"}, "output": {"temp": 72, "condition": "sunny"}}
        ...     ]
        ... )
        >>> # Context is automatically: "[weather_api]: {"temp": 72, "condition": "sunny"}"
    """

    def __init__(
        self,
        api_key: str,
        dashboard_url: str = "https://dashboard.cert-framework.com",
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        timeout: float = 5.0,
        auto_extract_context: bool = True,
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
            auto_extract_context: Automatically extract context from tool_calls
                                  in agentic mode (default: True)
        """
        self.api_key = api_key
        self.endpoint = f"{dashboard_url.rstrip('/')}/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout
        self.auto_extract_context = auto_extract_context

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
            context: RAG context/retrieved documents (implies "rag" mode).
                     For agentic mode: if None and tool_calls provided, context
                     is automatically extracted from tool outputs (unless
                     auto_extract_context=False).
            output_schema: Expected output structure for generation validation
            tool_calls: List of tool/function calls (implies "agentic" mode)
            goal_description: Description of the agentic goal
            metadata: Optional additional metadata
            
        Note:
            In agentic mode, tool outputs serve as the implicit context that
            enables full metric computation (SGI, grounding, NLI). If context
            is not explicitly provided and tool_calls are present, the SDK
            automatically constructs context from tool outputs.
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

        # === AUTOMATIC CONTEXT EXTRACTION FOR AGENTIC MODE ===
        # This is the key enhancement: c_agent = ⊕_i o_i
        effective_context = context
        if resolved_mode == "agentic" and self.auto_extract_context:
            if context is None and tool_calls and len(tool_calls) > 0:
                # Automatically extract context from tool outputs
                effective_context = extract_context_from_tool_calls(tool_calls)
                logger.debug(
                    f"CERT: Auto-extracted context from {len(tool_calls)} tool calls "
                    f"({len(effective_context)} chars)"
                )
            elif context is None and (not tool_calls or len(tool_calls) == 0):
                # Agentic mode without tools or context - warn about degraded metrics
                logger.warning(
                    "CERT: Agentic mode without tool_calls or context. "
                    "Evaluation will degrade to Generation Mode metrics "
                    "(no SGI, grounding, or NLI available)."
                )

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
        if effective_context is not None:
            trace_data["context"] = effective_context
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
        """Flush all pending traces. Blocks until complete or timeout."""
        batch = []
        try:
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        
        if batch:
            self._send_batch(batch)

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
