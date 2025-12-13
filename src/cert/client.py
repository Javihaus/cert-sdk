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
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from cert.types import EvalMode, SpanKind, ToolCall, TraceStatus

import requests

__version__ = "0.5.0"

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
    """
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "unknown_tool")
        
        if tc.get("error"):
            parts.append(f"[{name}] ERROR: {tc['error']}")
        elif tc.get("output") is not None:
            output = tc["output"]
            if isinstance(output, (dict, list)):
                output_str = json.dumps(output, ensure_ascii=False, default=str)
            else:
                output_str = str(output)
            parts.append(f"[{name}]: {output_str}")
    
    return "\n\n".join(parts)


# New name (v0.4.0+)
extract_knowledge_from_tool_calls = extract_context_from_tool_calls


class CertClient:
    """
    Non-blocking client for CERT dashboard.

    Traces are queued and sent in batches via background thread.
    Never blocks your application.

    Example (Minimal - just 4 required params):
        >>> client = CertClient(api_key="cert_xxx", project="my-app")
        >>> client.trace(
        ...     provider="anthropic",
        ...     model="claude-sonnet-4",
        ...     input_text="Hello",
        ...     output_text="Hi there!"
        ... )
        
    Example (With timing):
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     input_text="What is 2+2?",
        ...     output_text="4",
        ...     duration_ms=234,
        ...     prompt_tokens=10,
        ...     completion_tokens=5
        ... )
        
    Example (RAG/Grounded evaluation):
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     input_text="What is the capital of France?",
        ...     output_text="Paris is the capital of France.",
        ...     knowledge_base="France is a country in Europe. Paris is the capital.",
        ...     evaluation_mode="grounded"
        ... )
        
    Example (Agentic with tool calls):
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     input_text="What's the weather?",
        ...     output_text="It's 72°F and sunny.",
        ...     evaluation_mode="agentic",
        ...     tool_calls=[
        ...         {"name": "weather_api", "input": {"city": "NYC"}, "output": {"temp": 72}}
        ...     ]
        ... )
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
        auto_extract_knowledge: bool = True,
        auto_extract_context: Optional[bool] = None,  # DEPRECATED
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
            auto_extract_knowledge: Automatically extract knowledge from tool_calls
                                    in grounded mode (default: True)
            auto_extract_context: DEPRECATED, use auto_extract_knowledge
        """
        # Handle deprecated auto_extract_context parameter
        if auto_extract_context is not None:
            warnings.warn(
                "auto_extract_context is deprecated, use auto_extract_knowledge instead",
                DeprecationWarning,
                stacklevel=2,
            )
            auto_extract_knowledge = auto_extract_context

        self.api_key = api_key
        self.project = project
        self.endpoint = f"{dashboard_url.rstrip('/')}/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout
        self.auto_extract_knowledge = auto_extract_knowledge

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._traces_sent = 0
        self._traces_failed = 0

        self._start_worker()

    def trace(
        self,
        # === REQUIRED (only 4!) ===
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        # === TIMING (all optional) ===
        duration_ms: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        # === TOKENS (optional) ===
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        # === STATUS (optional) ===
        status: TraceStatus = "success",
        error_message: Optional[str] = None,
        # === TRACING/SPANS (optional) ===
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        name: Optional[str] = None,
        kind: SpanKind = "CLIENT",
        # === EVALUATION CONFIG (optional) ===
        # Primary names (recommended)
        evaluation_mode: Optional[str] = None,  # "grounded", "ungrounded", "agentic", "auto"
        knowledge_base: Optional[str] = None,   # Context for grounded evaluation
        # Aliases for compatibility
        eval_mode: Optional[str] = None,        # Alias for evaluation_mode
        context: Optional[str] = None,          # Alias for knowledge_base
        # Other eval params
        output_schema: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        goal_description: Optional[str] = None,
        # === TASK METADATA (optional) ===
        task_type: Optional[str] = None,        # e.g., "qa", "summarization", "chat"
        context_source: Optional[str] = None,   # e.g., "retrieval", "conversation"
        # === GENERAL METADATA (optional) ===
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an LLM trace. Non-blocking.

        Only 4 parameters are required: provider, model, input_text, output_text.
        Everything else has sensible defaults.

        Args:
            provider: LLM provider (e.g., "openai", "anthropic", "google")
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514")
            input_text: Input prompt/messages
            output_text: Model response
            
            duration_ms: Request duration in milliseconds (optional, defaults to 0)
            start_time: When the operation started (optional)
            end_time: When the operation ended (optional)
            
            prompt_tokens: Input token count (optional)
            completion_tokens: Output token count (optional)
            
            status: Trace status - "success" or "error" (default: "success")
            error_message: Error message if status is "error"
            
            trace_id: Unique trace identifier (auto-generated if not provided)
            span_id: Unique span identifier (auto-generated if not provided)
            parent_span_id: Parent span ID for nested spans
            name: Operation name (defaults to "{provider}.{model}")
            kind: Span kind (default: "CLIENT")
            
            evaluation_mode: "grounded", "ungrounded", "agentic", or "auto"
                            - grounded: Has knowledge_base to verify against
                            - ungrounded: No external context
                            - agentic: Multi-step with tool calls
                            - auto: Auto-detect based on other params
            knowledge_base: Context/documents for grounded evaluation
            
            eval_mode: Alias for evaluation_mode (for compatibility)
            context: Alias for knowledge_base (for compatibility)
            
            output_schema: Expected output structure for validation
            tool_calls: List of tool/function calls for agentic mode
            goal_description: Description of the agentic goal
            
            task_type: Type of task (e.g., "qa", "chat", "summarization")
            context_source: Source of context (e.g., "retrieval", "conversation")
            
            metadata: Additional custom metadata

        Returns:
            Trace ID (UUID string)
        """
        # === Handle deprecated parameter warnings ===
        if eval_mode is not None:
            warnings.warn(
                "eval_mode is deprecated, use evaluation_mode instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if context is not None:
            warnings.warn(
                "context is deprecated, use knowledge_base instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # === Handle parameter aliases ===
        # evaluation_mode / eval_mode
        effective_eval_mode = evaluation_mode or eval_mode or "auto"
        legacy_eval_mode = eval_mode  # Keep original for backwards compat field

        # knowledge_base / context
        effective_knowledge_base = knowledge_base or context

        # === Handle timing ===
        effective_duration = duration_ms if duration_ms is not None else 0

        # Generate timestamps if not provided
        now = datetime.now(timezone.utc)
        effective_end_time = end_time or now
        effective_start_time = start_time or now

        # === Generate IDs ===
        _trace_id = trace_id or str(uuid.uuid4())
        _span_id = span_id or f"span-{uuid.uuid4().hex[:8]}"
        _name = name or f"{provider}.{model}"

        # === Normalize eval_mode to new API values ===
        mode_mapping = {
            "grounded": "grounded",
            "ungrounded": "ungrounded",
            "agentic": "grounded",  # agentic maps to grounded (tool-based context)
            "auto": "auto",
            "rag": "grounded",       # legacy rag -> grounded
            "generation": "ungrounded",  # legacy generation -> ungrounded
        }
        normalized_mode = mode_mapping.get(effective_eval_mode.lower(), "auto")

        # === Determine effective context_source ===
        effective_context_source = context_source

        # === Auto-detect mode if "auto" ===
        if normalized_mode == "auto":
            if tool_calls and len(tool_calls) > 0:
                normalized_mode = "grounded"
                if effective_context_source is None:
                    effective_context_source = "tools"
            elif effective_knowledge_base:
                normalized_mode = "grounded"
            else:
                normalized_mode = "ungrounded"

        # === Auto-extract knowledge from tool_calls for grounded mode ===
        if (
            normalized_mode == "grounded"
            and tool_calls
            and len(tool_calls) > 0
            and effective_knowledge_base is None
            and self.auto_extract_knowledge
        ):
            effective_knowledge_base = extract_knowledge_from_tool_calls(tool_calls)
            if effective_context_source is None:
                effective_context_source = "tools"

        # === Set context_source to "tools" if tool_calls provided ===
        if tool_calls and len(tool_calls) > 0 and effective_context_source is None:
            effective_context_source = "tools"

        # === Validate tool_calls ===
        if tool_calls:
            _validate_tool_calls(tool_calls)

        # === Build metadata ===
        effective_metadata = metadata.copy() if metadata else {}
        if task_type:
            effective_metadata["task_type"] = task_type

        # === Build trace payload ===
        trace_data = {
            "trace_id": _trace_id,
            "span_id": _span_id,
            "name": _name,
            "kind": kind,
            "project_name": self.project,
            "llm_vendor": provider,
            "model": model,
            "input_text": input_text,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": effective_duration,
            "evaluation_mode": normalized_mode,
            "status": status,
            "timestamp": effective_end_time.isoformat(),
            "start_time": effective_start_time.isoformat(),
            "end_time": effective_end_time.isoformat(),
            "source": "cert-sdk",
        }

        # === Add optional fields ===
        if parent_span_id:
            trace_data["parent_span_id"] = parent_span_id
        if error_message:
            trace_data["error_message"] = error_message
        if effective_knowledge_base is not None:
            trace_data["knowledge_base"] = effective_knowledge_base
            trace_data["context"] = effective_knowledge_base  # backwards compat
        if output_schema is not None:
            trace_data["output_schema"] = output_schema
        if tool_calls is not None:
            trace_data["tool_calls"] = tool_calls
        if goal_description is not None:
            trace_data["goal_description"] = goal_description
        if effective_context_source is not None:
            trace_data["context_source"] = effective_context_source
        if effective_metadata:
            trace_data["metadata"] = effective_metadata
        # Include legacy eval_mode for backwards compatibility
        if legacy_eval_mode is not None:
            trace_data["eval_mode"] = legacy_eval_mode

        # === Queue the trace ===
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
                time_until_flush = self.flush_interval - (time.time() - last_flush)
                timeout = max(0.1, min(1.0, time_until_flush))

                trace = self._queue.get(timeout=timeout)
                batch.append(trace)

                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

            except queue.Empty:
                if batch and (time.time() - last_flush) >= self.flush_interval:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

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

    Example:
        >>> with TraceContext(client, provider="openai", model="gpt-4", input_text="Hello") as ctx:
        ...     response = llm.invoke(prompt)
        ...     ctx.set_output(response.content)
        ...     ctx.set_tokens(response.usage.input, response.usage.output)
    """

    def __init__(
        self,
        client: "CertClient",
        provider: str,
        model: str,
        input_text: str,
        **kwargs,
    ):
        self.client = client
        self.provider = provider
        self.model = model
        self.input_text = input_text
        self.kwargs = kwargs

        self.start_time: Optional[datetime] = None
        self.trace_id: str = str(uuid.uuid4())
        self.span_id: str = f"span-{uuid.uuid4().hex[:8]}"

        self.output_text: str = ""
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.knowledge_base: Optional[str] = None
        self.context_source: Optional[str] = None

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

    def set_knowledge_base(self, knowledge_base: str, source: Optional[str] = None) -> None:
        """
        Set the knowledge base for grounded evaluation.

        Args:
            knowledge_base: The context/documents for grounded evaluation
            source: Optional source of the context (e.g., "retrieval", "tools")
        """
        self.knowledge_base = knowledge_base
        if source is not None:
            self.context_source = source
        # Also update kwargs to ensure grounded evaluation mode
        if "evaluation_mode" not in self.kwargs:
            self.kwargs["evaluation_mode"] = "grounded"

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log trace when exiting the context."""
        end_time = datetime.now(timezone.utc)
        assert self.start_time is not None
        duration_ms = (end_time - self.start_time).total_seconds() * 1000

        status: TraceStatus = "error" if exc_type else "success"
        error_message = str(exc_val) if exc_val else None

        # Build trace kwargs
        trace_kwargs = dict(self.kwargs)
        if self.knowledge_base is not None:
            trace_kwargs["knowledge_base"] = self.knowledge_base
        if self.context_source is not None:
            trace_kwargs["context_source"] = self.context_source

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
            **trace_kwargs,
        )
