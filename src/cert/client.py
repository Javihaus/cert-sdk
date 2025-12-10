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

from cert.types import EvalMode, SpanKind, ToolCall, TraceStatus

import requests

__version__ = "0.3.0"

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
        project: str = "default",
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
            project: Project name for trace organization (default: "default")
            dashboard_url: Dashboard URL (default: production)
            batch_size: Traces per batch (default: 10)
            flush_interval: Seconds between flushes (default: 5.0)
            max_queue_size: Max traces to queue (default: 1000)
            timeout: HTTP timeout in seconds (default: 5.0)
            auto_extract_context: Automatically extract context from tool_calls
                                  in agentic mode (default: True)
        """
        self.api_key = api_key
        self.project = project
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
        # === REQUIRED ===
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        duration_ms: float,
        # === TOKENS ===
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        # total_tokens computed automatically
        # === TIMING (NEW) ===
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        # === STATUS (NEW) ===
        status: TraceStatus = "success",
        error_message: Optional[str] = None,
        # === TRACING/SPANS (NEW) ===
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        name: Optional[str] = None,
        kind: SpanKind = "CLIENT",
        # === EVALUATION CONFIG ===
        eval_mode: EvalMode = "auto",
        context: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        goal_description: Optional[str] = None,
        # === METADATA ===
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an LLM trace. Non-blocking.

        Args:
            provider: LLM provider (e.g., "openai", "anthropic") - maps to 'vendor' in DB
            model: Model name (e.g., "gpt-4", "claude-sonnet-4")
            input_text: Input prompt/messages
            output_text: Model response
            duration_ms: Request duration in milliseconds
            prompt_tokens: Input token count
            completion_tokens: Output token count
            start_time: When the operation started (auto-generated if not provided)
            end_time: When the operation ended (auto-generated if not provided)
            status: Trace status - "success" or "error"
            error_message: Error message if status is "error"
            trace_id: Unique trace identifier for correlating multi-span traces
            span_id: Unique span identifier (auto-generated if not provided)
            parent_span_id: Parent span ID for nested spans
            name: Operation name (defaults to "{provider}.{model}")
            kind: Span kind - "CLIENT", "SERVER", "INTERNAL", "PRODUCER", "CONSUMER"
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

        Returns:
            str: The trace_id (for correlation with child spans)

        Note:
            In agentic mode, tool outputs serve as the implicit context that
            enables full metric computation (SGI, grounding, NLI). If context
            is not explicitly provided and tool_calls are present, the SDK
            automatically constructs context from tool outputs.
        """
        # Validate tool_calls structure if provided
        if tool_calls is not None:
            _validate_tool_calls(tool_calls)

        # Generate IDs if not provided
        _trace_id = trace_id or str(uuid.uuid4())
        _span_id = span_id or f"span-{uuid.uuid4().hex[:8]}"

        # Compute timestamps
        now = datetime.now(timezone.utc)
        _start_time = start_time or now
        _end_time = end_time or now

        # Auto-generate operation name
        _name = name or f"{provider}.{model}"

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

        # Build payload - field names match DB columns (camelCase for JSON)
        trace_data: Dict[str, Any] = {
            # Identity
            "id": str(uuid.uuid4()),
            "traceId": _trace_id,
            "spanId": _span_id,
            # Operation
            "name": _name,
            "kind": kind,
            "source": "cert-sdk",
            # Project
            "project": self.project,
            # LLM details - NOTE: provider → vendor
            "vendor": provider,
            "model": model,
            "inputText": input_text,
            "outputText": output_text,
            # Tokens
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            # Timing
            "durationMs": duration_ms,
            "startTime": _start_time.isoformat(),
            "endTime": _end_time.isoformat(),
            # Status
            "status": status,
            # Evaluation
            "evalMode": resolved_mode,
            # Metadata
            "metadata": metadata or {},
        }

        # === Conditional fields (only if present) ===
        if parent_span_id is not None:
            trace_data["parentSpanId"] = parent_span_id
        if error_message is not None:
            trace_data["errorMessage"] = error_message
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

        return _trace_id

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


class TraceContext:
    """
    Context manager for automatic timing and error capture.

    Automatically captures start/end times, duration, and error status.
    Use this to wrap LLM calls for easier tracing.

    Example:
        >>> with TraceContext(client, provider="openai", model="gpt-4", input_text="Hello") as ctx:
        ...     response = llm.invoke(prompt)
        ...     ctx.set_output(response.content)
        ...     ctx.set_tokens(response.usage.input, response.usage.output)

    With error capture:
        >>> with TraceContext(client, provider="openai", model="gpt-4", input_text="Hello") as ctx:
        ...     try:
        ...         response = llm.invoke(prompt)
        ...         ctx.set_output(response.content)
        ...     except Exception as e:
        ...         # Error is automatically captured, status set to "error"
        ...         raise
    """

    def __init__(
        self,
        client: "CertClient",
        provider: str,
        model: str,
        input_text: str,
        **kwargs,
    ):
        """
        Initialize TraceContext.

        Args:
            client: CertClient instance to send traces to
            provider: LLM provider (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-sonnet-4")
            input_text: Input prompt/messages
            **kwargs: Additional arguments passed to client.trace()
        """
        self.client = client
        self.provider = provider
        self.model = model
        self.input_text = input_text
        self.kwargs = kwargs

        self.start_time: Optional[datetime] = None
        self.trace_id: str = str(uuid.uuid4())
        self.span_id: str = f"span-{uuid.uuid4().hex[:8]}"

        # To be set during execution
        self.output_text: str = ""
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def __enter__(self) -> "TraceContext":
        """Start timing when entering the context."""
        self.start_time = datetime.now(timezone.utc)
        return self

    def set_output(self, output: str) -> None:
        """Set the output text."""
        self.output_text = output

    def set_tokens(self, prompt: int, completion: int) -> None:
        """Set token counts."""
        self.prompt_tokens = prompt
        self.completion_tokens = completion

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log trace when exiting the context."""
        end_time = datetime.now(timezone.utc)
        # start_time is always set in __enter__, so this is safe
        assert self.start_time is not None
        duration_ms = (end_time - self.start_time).total_seconds() * 1000

        status: TraceStatus = "error" if exc_type else "success"
        error_message = str(exc_val) if exc_val else None

        self.client.trace(
            provider=self.provider,
            model=self.model,
            input_text=self.input_text,
            output_text=self.output_text,
            duration_ms=duration_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            trace_id=self.trace_id,
            span_id=self.span_id,
            start_time=self.start_time,
            end_time=end_time,
            status=status,
            error_message=error_message,
            **self.kwargs,
        )
        # Don't suppress exceptions (returning None is equivalent to False)
