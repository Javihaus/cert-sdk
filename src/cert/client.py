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

__version__ = "1.0.0"

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
    Extract knowledge/context string from tool call outputs.

    Concatenates all tool outputs into a formatted string that can be used
    as knowledge_base for grounded evaluation. This enables faithfulness
    and grounding metrics for agentic LLM applications.

    Examples:
        >>> tool_calls = [
        ...     {"name": "search", "output": {"results": ["doc1", "doc2"]}},
        ...     {"name": "calculator", "output": 42}
        ... ]
        >>> knowledge = extract_context_from_tool_calls(tool_calls)
        >>> print(knowledge)
        [search]: {"results": ["doc1", "doc2"]}

        [calculator]: 42

        >>> # Errors are also captured
        >>> tool_calls = [{"name": "api", "error": "Connection timeout"}]
        >>> extract_context_from_tool_calls(tool_calls)
        '[api] ERROR: Connection timeout'

    Args:
        tool_calls: List of tool call dictionaries. Each dict can have:
            - "name" (str): Tool name (used as label in output)
            - "output" (any): Tool response (JSON-serialized if dict/list)
            - "error" (str): Error message if the tool failed

    Returns:
        str: Formatted string with all tool outputs, separated by double newlines.
            Empty string if no tools have outputs or errors.

    Note:
        This function is called automatically when auto_extract_knowledge=True
        (the default) and tool_calls are provided without explicit knowledge_base.
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
    Non-blocking client for CERT dashboard LLM monitoring.

    Traces are queued and sent in batches via a background thread,
    ensuring your application is never blocked by monitoring overhead.

    Features:
        - Non-blocking: trace() returns immediately, data sent asynchronously
        - Batching: Traces are batched for efficient network usage
        - Auto-retry: Failed batches are logged but don't crash your app
        - Context manager: Use with `with` statement for automatic cleanup

    Examples:
        Basic usage:
            >>> client = CertClient(api_key="cert_xxx", project="my-app")
            >>> client.trace(
            ...     provider="anthropic",
            ...     model="claude-sonnet-4-20250514",
            ...     input_text="Hello",
            ...     output_text="Hi there!"
            ... )
            >>> client.close()  # Send remaining traces on shutdown

        As context manager (recommended):
            >>> with CertClient(api_key="cert_xxx", project="my-app") as client:
            ...     client.trace(
            ...         provider="openai",
            ...         model="gpt-4o",
            ...         input_text="What is 2+2?",
            ...         output_text="4",
            ...         duration_ms=234,
            ...         prompt_tokens=10,
            ...         completion_tokens=5
            ...     )
            ... # Automatically flushes and closes

        Grounded evaluation (RAG):
            >>> client.trace(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     input_text="What is the capital of France?",
            ...     output_text="Paris is the capital of France.",
            ...     knowledge_base="France is a country in Europe. Paris is the capital.",
            ...     evaluation_mode="grounded"
            ... )

        With tool calls (knowledge auto-extracted):
            >>> client.trace(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     input_text="What's the weather?",
            ...     output_text="It's 72°F and sunny.",
            ...     tool_calls=[
            ...         {"name": "weather_api", "input": {"city": "NYC"}, "output": {"temp": 72}}
            ...     ]
            ... )

        Check statistics:
            >>> stats = client.get_stats()
            >>> print(f"Sent: {stats['traces_sent']}, Failed: {stats['traces_failed']}")

    Attributes:
        api_key: Your CERT API key.
        project: Project name for organizing traces.
        endpoint: The API endpoint URL.
        auto_extract_knowledge: Whether to auto-extract knowledge from tool outputs.

    See Also:
        TraceContext: Context manager for automatic timing and error capture.
        extract_knowledge_from_tool_calls: Utility to extract context from tool outputs.
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
        Log an LLM trace to the CERT dashboard. Non-blocking.

        Only 4 parameters are required: provider, model, input_text, output_text.
        Everything else has sensible defaults.

        Examples:
            Minimal trace (just the essentials):
                >>> client.trace(
                ...     provider="openai",
                ...     model="gpt-4o",
                ...     input_text="What is 2+2?",
                ...     output_text="4"
                ... )

            With timing and token counts:
                >>> client.trace(
                ...     provider="anthropic",
                ...     model="claude-sonnet-4-20250514",
                ...     input_text="Explain quantum computing",
                ...     output_text="Quantum computing uses...",
                ...     duration_ms=1234.5,
                ...     prompt_tokens=15,
                ...     completion_tokens=150
                ... )

            Grounded evaluation (RAG/retrieval):
                >>> client.trace(
                ...     provider="openai",
                ...     model="gpt-4o",
                ...     input_text="What is the capital of France?",
                ...     output_text="Paris is the capital of France.",
                ...     evaluation_mode="grounded",
                ...     knowledge_base="France is a country in Europe. Paris is the capital.",
                ...     context_source="retrieval"
                ... )

            With tool calls (auto-extracts knowledge):
                >>> client.trace(
                ...     provider="openai",
                ...     model="gpt-4o",
                ...     input_text="What's the weather in NYC?",
                ...     output_text="It's 72°F and sunny in New York.",
                ...     tool_calls=[
                ...         {"name": "get_weather", "input": {"city": "NYC"}, "output": {"temp": 72, "condition": "sunny"}}
                ...     ]
                ... )

            Error handling:
                >>> client.trace(
                ...     provider="openai",
                ...     model="gpt-4o",
                ...     input_text="Generate something",
                ...     output_text="",
                ...     status="error",
                ...     error_message="Rate limit exceeded"
                ... )

        Args:
            provider: LLM provider name (e.g., "openai", "anthropic", "google", "cohere").
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514", "gemini-pro").
            input_text: The input prompt or messages sent to the LLM.
            output_text: The response text from the LLM.

            duration_ms: Request duration in milliseconds. Defaults to 0 if not provided.
            start_time: When the LLM call started (datetime with timezone).
            end_time: When the LLM call completed (datetime with timezone).

            prompt_tokens: Number of input tokens. Defaults to 0.
            completion_tokens: Number of output tokens. Defaults to 0.

            status: Trace status - "success" or "error". Defaults to "success".
            error_message: Error details when status is "error".

            trace_id: Unique trace identifier. Auto-generated UUID if not provided.
            span_id: Unique span identifier. Auto-generated if not provided.
            parent_span_id: Parent span ID for creating nested/hierarchical traces.
            name: Operation name. Defaults to "{provider}.{model}".
            kind: Span kind - "CLIENT", "SERVER", "PRODUCER", "CONSUMER", or "INTERNAL".

            evaluation_mode: Controls which metrics are computed:
                - "grounded": Full metrics including faithfulness, NLI (requires knowledge_base)
                - "ungrounded": Basic metrics only (coherence, toxicity)
                - "auto": Auto-detect based on knowledge_base/tool_calls presence (default)
            knowledge_base: Reference documents/context for grounded evaluation.
                When provided, enables faithfulness and grounding metrics.
            context_source: Source of the knowledge_base - "retrieval", "tools", "conversation".

            eval_mode: DEPRECATED. Use evaluation_mode instead.
            context: DEPRECATED. Use knowledge_base instead.

            output_schema: JSON Schema for structured output validation.
            tool_calls: List of tool/function calls. Each dict should have:
                - "name" (required): Tool name
                - "input" (optional): Tool input parameters
                - "output" (optional): Tool response (auto-extracted as knowledge_base)
                - "error" (optional): Error message if tool failed
            goal_description: High-level description of what the agent is trying to achieve.

            task_type: Type of task - "qa", "summarization", "chat", "code_generation", etc.
            metadata: Additional custom key-value pairs to attach to the trace.

        Returns:
            str: The trace_id (UUID string) that can be used for correlation.

        Note:
            This method is non-blocking. Traces are queued and sent in batches
            by a background thread. Use client.flush() to force immediate send,
            or client.close() on shutdown to ensure all traces are sent.
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
    Use this when you want precise timing without manual bookkeeping.

    Features:
        - Automatic timing: start_time, end_time, duration_ms captured automatically
        - Error capture: Exceptions are caught and logged with status="error"
        - Flexible: Set output, tokens, and knowledge_base during execution

    Examples:
        Basic usage with OpenAI:
            >>> with TraceContext(client, provider="openai", model="gpt-4o", input_text="Hello") as ctx:
            ...     response = openai.chat.completions.create(
            ...         model="gpt-4o",
            ...         messages=[{"role": "user", "content": "Hello"}]
            ...     )
            ...     ctx.set_output(response.choices[0].message.content)
            ...     ctx.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

        With RAG/grounded evaluation:
            >>> with TraceContext(client, provider="openai", model="gpt-4o", input_text=question) as ctx:
            ...     # Retrieve documents
            ...     docs = retriever.get_relevant_docs(question)
            ...     ctx.set_knowledge_base("\\n".join(docs), source="retrieval")
            ...     # Generate response
            ...     response = llm.generate(question, context=docs)
            ...     ctx.set_output(response.text)

        With tool calls:
            >>> with TraceContext(
            ...     client,
            ...     provider="anthropic",
            ...     model="claude-sonnet-4-20250514",
            ...     input_text="What's the weather?",
            ...     tool_calls=[{"name": "get_weather", "output": {"temp": 72}}]
            ... ) as ctx:
            ...     ctx.set_output("It's 72°F")

        Error handling (automatic):
            >>> with TraceContext(client, provider="openai", model="gpt-4o", input_text="Hi") as ctx:
            ...     raise ValueError("API Error")  # Automatically logged with status="error"

    Attributes:
        trace_id (str): Unique trace identifier (auto-generated UUID).
        span_id (str): Unique span identifier (auto-generated).
        start_time (datetime): When the context was entered.
        output_text (str): The LLM response (set via set_output).
        prompt_tokens (int): Input token count (set via set_tokens).
        completion_tokens (int): Output token count (set via set_tokens).
        knowledge_base (str): Context for grounded evaluation (set via set_knowledge_base).

    Args:
        client: The CertClient instance to use for logging.
        provider: LLM provider name (e.g., "openai", "anthropic").
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        input_text: The input prompt or messages.
        **kwargs: Additional arguments passed to client.trace() (e.g., tool_calls, metadata).
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
