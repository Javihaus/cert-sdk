"""
CERT SDK - Python client for LLM monitoring v0.5.0

Two-mode evaluation architecture: Grounded vs Ungrounded
Bias detection: Demographic bias and custom policies
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

from cert.types import (
    EvaluationMode,
    ContextSource,
    SpanKind,
    ToolCall,
    TraceStatus,
    _map_legacy_mode,
    _infer_context_source,
    # Deprecated
    EvalMode,
)

from cert.bias.models import (
    DemographicBiasConfig,
    CustomPolicy,
    BiasSeverity,
)
from cert.bias.templates import get_template, list_templates

import requests

__version__ = "0.5.0"

logger = logging.getLogger(__name__)


def _validate_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """Validate tool_calls structure before sending."""
    for i, tc in enumerate(tool_calls):
        if "name" not in tc:
            raise ValueError(f"tool_calls[{i}] missing required 'name' field")
        if not isinstance(tc.get("name"), str):
            raise ValueError(f"tool_calls[{i}].name must be a string")


def extract_knowledge_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Extract knowledge base string from tool call outputs.

    This is the c_agent = ⊕_i o_i operation from the CERT framework.
    Tool outputs constitute the implicit knowledge base that enables
    grounded evaluation with full metrics (SGI, source_accuracy, faithfulness).

    Args:
        tool_calls: List of tool call dictionaries with 'name', 'output', 'error' keys

    Returns:
        Concatenated knowledge string from all tool outputs
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


# Backwards compatibility alias
extract_context_from_tool_calls = extract_knowledge_from_tool_calls


class CertClient:
    """
    Non-blocking client for CERT dashboard.

    Traces are queued and sent in batches via background thread.
    Never blocks your application.

    Example (New API):
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

    Grounded Mode with Knowledge Base:
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     input_text="What's the weather?",
        ...     output_text="It's 72°F and sunny.",
        ...     duration_ms=1500,
        ...     evaluation_mode="grounded",
        ...     knowledge_base="Weather data: NYC temp 72°F, sunny skies",
        ...     context_source="retrieval"
        ... )

    Grounded Mode with Tool Outputs (Auto-detected):
        >>> client.trace(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     input_text="What's the weather?",
        ...     output_text="It's 72°F and sunny.",
        ...     duration_ms=1500,
        ...     tool_calls=[
        ...         {"name": "weather_api", "input": {"city": "NYC"},
        ...          "output": {"temp": 72, "condition": "sunny"}}
        ...     ]
        ... )
        >>> # Automatically: evaluation_mode="grounded", context_source="tools"
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
        # Bias detection (v0.5.0)
        demographic_bias: Optional[Union[bool, DemographicBiasConfig]] = True,
        custom_policy: Optional[Union[str, CustomPolicy]] = None,
        # Deprecated parameter name (backwards compatibility)
        auto_extract_context: Optional[bool] = None,
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
            auto_extract_knowledge: Automatically extract knowledge_base from
                                    tool_calls outputs (default: True)
            demographic_bias: Enable demographic bias detection (v0.5.0)
                - True: Enable with default config (default)
                - False: Disable
                - DemographicBiasConfig: Custom configuration
            custom_policy: Custom policy for domain-specific bias detection (v0.5.0)
                - str: Template name (e.g., "financial_services")
                - CustomPolicy: Custom policy instance
            auto_extract_context: DEPRECATED - use auto_extract_knowledge
        """
        self.api_key = api_key
        self.project = project
        self.endpoint = f"{dashboard_url.rstrip('/')}/api/v1/traces"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        # Handle deprecated parameter
        if auto_extract_context is not None:
            warnings.warn(
                "auto_extract_context is deprecated, use auto_extract_knowledge instead",
                DeprecationWarning,
                stacklevel=2
            )
            self.auto_extract_knowledge = auto_extract_context
        else:
            self.auto_extract_knowledge = auto_extract_knowledge

        # Initialize demographic bias config
        self._demographic_bias_config: Optional[DemographicBiasConfig] = None
        if demographic_bias is True:
            self._demographic_bias_config = DemographicBiasConfig()
        elif isinstance(demographic_bias, DemographicBiasConfig):
            self._demographic_bias_config = demographic_bias
        # If False or None, demographic bias is disabled

        # Initialize custom policy
        self._custom_policy: Optional[CustomPolicy] = None
        if custom_policy is not None:
            if isinstance(custom_policy, str):
                self._custom_policy = get_template(custom_policy)
            elif isinstance(custom_policy, CustomPolicy):
                self._custom_policy = custom_policy
            else:
                raise TypeError(
                    f"custom_policy must be str or CustomPolicy, got {type(custom_policy)}"
                )

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
        # === TIMING ===
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        # === STATUS ===
        status: TraceStatus = "success",
        error_message: Optional[str] = None,
        # === TRACING/SPANS ===
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        name: Optional[str] = None,
        kind: SpanKind = "CLIENT",
        # === EVALUATION CONFIG (v0.4.0) ===
        evaluation_mode: EvaluationMode = "auto",
        knowledge_base: Optional[str] = None,
        context_source: Optional[ContextSource] = None,
        # === TOOL CALLS ===
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        goal_description: Optional[str] = None,
        # === GENERATION MODE ===
        output_schema: Optional[Dict[str, Any]] = None,
        # === BIAS DETECTION (v0.5.0) ===
        task_type: Optional[str] = None,
        skip_bias_detection: bool = False,
        demographic_bias_override: Optional[Union[bool, DemographicBiasConfig]] = None,
        custom_policy_override: Optional[Union[str, CustomPolicy]] = None,
        # === METADATA ===
        metadata: Optional[Dict[str, Any]] = None,
        # === DEPRECATED (v0.3.x compatibility) ===
        eval_mode: Optional[str] = None,  # DEPRECATED: Use evaluation_mode
        context: Optional[str] = None,     # DEPRECATED: Use knowledge_base
    ) -> str:
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
            start_time: When the operation started
            end_time: When the operation ended
            status: Trace status - "success" or "error"
            error_message: Error message if status is "error"
            trace_id: Unique trace identifier for correlation
            span_id: Unique span identifier
            parent_span_id: Parent span ID for nested spans
            name: Operation name (defaults to "{provider}.{model}")
            kind: Span kind

            evaluation_mode: "grounded", "ungrounded", or "auto" (default)
                - grounded: Has knowledge_base, enables full metric suite
                - ungrounded: No knowledge_base, basic metrics only
                - auto: Detect based on knowledge_base/tool_calls presence

            knowledge_base: Source knowledge for grounded evaluation.
                If not provided but tool_calls have outputs, knowledge
                is automatically extracted (unless auto_extract_knowledge=False).

            context_source: How knowledge was obtained:
                - "retrieval": From RAG/vector search
                - "tools": From tool/function call outputs
                - "conversation": From conversation history
                - "user_provided": Explicitly provided by user

            tool_calls: List of tool/function calls [{name, input, output, error}]
            goal_description: Task goal for completion tracking
            output_schema: Expected output structure for validation

            task_type: Task categorization for custom policy filtering (v0.5.0)
            skip_bias_detection: Skip bias evaluation for this trace (v0.5.0)
            demographic_bias_override: Override client-level demographic config (v0.5.0)
            custom_policy_override: Override client-level custom policy (v0.5.0)

            metadata: Additional custom fields

            eval_mode: DEPRECATED - Use evaluation_mode instead
            context: DEPRECATED - Use knowledge_base instead

        Returns:
            trace_id: Unique identifier for this trace
        """
        # Handle deprecated parameters with warnings
        if eval_mode is not None:
            warnings.warn(
                "eval_mode is deprecated, use evaluation_mode instead",
                DeprecationWarning,
                stacklevel=2
            )
            if evaluation_mode == "auto":
                evaluation_mode = _map_legacy_mode(eval_mode)

        if context is not None:
            warnings.warn(
                "context is deprecated, use knowledge_base instead",
                DeprecationWarning,
                stacklevel=2
            )
            if knowledge_base is None:
                knowledge_base = context

        # Validate tool_calls if provided
        if tool_calls:
            _validate_tool_calls(tool_calls)

        # Auto-extract knowledge from tool outputs
        effective_knowledge = knowledge_base
        if (
            self.auto_extract_knowledge
            and effective_knowledge is None
            and tool_calls
        ):
            extracted = extract_knowledge_from_tool_calls(tool_calls)
            if extracted:
                effective_knowledge = extracted
                if context_source is None:
                    context_source = "tools"

        # Resolve evaluation mode
        resolved_mode = evaluation_mode
        if evaluation_mode == "auto":
            if effective_knowledge:
                resolved_mode = "grounded"
            else:
                resolved_mode = "ungrounded"

        # Infer context source if not provided
        if context_source is None and effective_knowledge:
            context_source = _infer_context_source(
                effective_knowledge, tool_calls, None
            )

        # Generate IDs
        _trace_id = trace_id or str(uuid.uuid4())
        _span_id = span_id or f"span-{uuid.uuid4().hex[:8]}"

        # Build trace data with NEW field names
        now = datetime.now(timezone.utc)
        trace_data: Dict[str, Any] = {
            # Identity
            "id": str(uuid.uuid4()),
            "trace_id": _trace_id,
            "span_id": _span_id,
            "parent_span_id": parent_span_id,
            "name": name or f"{provider}.{model}",
            "kind": kind,
            "source": "cert-sdk",
            # LLM details (use new field names for server)
            "llm_vendor": provider,
            "llm_model": model,
            "input_text": input_text,
            "output_text": output_text,
            # Tokens
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Timing
            "duration_ms": duration_ms,
            "start_time": (start_time or now).isoformat(),
            "end_time": (end_time or now).isoformat(),
            # Status
            "status": status,
            # Project
            "project_name": self.project,
            # === NEW v0.4.0 FIELDS ===
            "evaluation_mode": resolved_mode,
            "context_source": context_source,
            # Metadata
            "metadata": metadata or {},
        }

        # Remove None values for optional fields
        if parent_span_id is None:
            del trace_data["parent_span_id"]
        if context_source is None:
            del trace_data["context_source"]

        # Add optional fields
        if error_message:
            trace_data["error_message"] = error_message
        if effective_knowledge is not None:
            trace_data["knowledge_base"] = effective_knowledge
            # Also send as 'context' for backwards compatibility with older servers
            trace_data["context"] = effective_knowledge
        if output_schema is not None:
            trace_data["output_schema"] = output_schema
        if tool_calls is not None:
            trace_data["tool_calls"] = tool_calls
        if goal_description is not None:
            trace_data["goal_description"] = goal_description

        # Legacy field for backward compatibility with older servers
        if eval_mode is not None:
            trace_data["eval_mode"] = eval_mode

        # Add task_type if provided (for custom policy filtering)
        if task_type is not None:
            trace_data["task_type"] = task_type

        # Add bias configuration unless skipped
        if not skip_bias_detection:
            bias_config = self._build_bias_config(
                demographic_bias_override,
                custom_policy_override,
            )
            if bias_config:
                trace_data["bias_config"] = bias_config

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

    # =========================================================================
    # Bias Configuration Methods
    # =========================================================================

    def _build_bias_config(
        self,
        demographic_override: Optional[Union[bool, DemographicBiasConfig]],
        policy_override: Optional[Union[str, CustomPolicy]],
    ) -> Optional[Dict[str, Any]]:
        """Build bias configuration for trace payload."""
        config: Dict[str, Any] = {}

        # Determine demographic config to use
        demographic_config = None
        if demographic_override is not None:
            if demographic_override is True:
                demographic_config = DemographicBiasConfig()
            elif isinstance(demographic_override, DemographicBiasConfig):
                demographic_config = demographic_override
            # If False, leave as None (disabled)
        else:
            demographic_config = self._demographic_bias_config

        if demographic_config:
            config["demographic"] = demographic_config.to_dict()

        # Determine custom policy to use
        custom_policy = None
        if policy_override is not None:
            if isinstance(policy_override, str):
                custom_policy = get_template(policy_override)
            elif isinstance(policy_override, CustomPolicy):
                custom_policy = policy_override
        else:
            custom_policy = self._custom_policy

        if custom_policy:
            config["custom_policy"] = custom_policy.to_dict()

        return config if config else None

    def configure_demographic_bias(
        self,
        config: DemographicBiasConfig,
    ) -> "CertClient":
        """
        Set demographic bias configuration.

        Example:
            config = DemographicBiasConfig()
            config.enable("political").set_threshold("gender", 0.7)
            client.configure_demographic_bias(config)
        """
        self._demographic_bias_config = config
        return self

    def enable_demographic_category(self, category_id: str) -> "CertClient":
        """Enable a specific demographic bias category."""
        if self._demographic_bias_config is None:
            self._demographic_bias_config = DemographicBiasConfig()
        self._demographic_bias_config.enable(category_id)
        return self

    def disable_demographic_category(self, category_id: str) -> "CertClient":
        """Disable a specific demographic bias category."""
        if self._demographic_bias_config:
            self._demographic_bias_config.disable(category_id)
        return self

    def set_bias_threshold(
        self,
        category_id: str,
        threshold: float,
    ) -> "CertClient":
        """Set detection threshold for a demographic bias category."""
        if self._demographic_bias_config is None:
            self._demographic_bias_config = DemographicBiasConfig()
        self._demographic_bias_config.set_threshold(category_id, threshold)
        return self

    def set_custom_policy(
        self,
        policy: Union[str, CustomPolicy],
    ) -> "CertClient":
        """
        Set custom policy for domain-specific bias detection.

        Args:
            policy: Template name (str) or CustomPolicy instance

        Example:
            client.set_custom_policy("financial_services")
        """
        if isinstance(policy, str):
            self._custom_policy = get_template(policy)
        else:
            self._custom_policy = policy
        return self

    def clear_custom_policy(self) -> "CertClient":
        """Remove custom policy configuration."""
        self._custom_policy = None
        return self

    def disable_bias_detection(self) -> "CertClient":
        """Disable all bias detection."""
        self._demographic_bias_config = None
        self._custom_policy = None
        return self

    # =========================================================================
    # Static Discovery Methods
    # =========================================================================

    @staticmethod
    def list_demographic_categories() -> List[Dict[str, Any]]:
        """List all available demographic bias categories."""
        config = DemographicBiasConfig()
        return [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "consensus": cat.consensus.value,
                "default_enabled": cat.enabled,
                "default_threshold": cat.threshold,
                "default_severity": cat.severity.value,
            }
            for cat in config.categories
        ]

    @staticmethod
    def list_policy_templates() -> List[Dict[str, Any]]:
        """List available custom policy templates."""
        return list_templates()

    @staticmethod
    def get_policy_template(name: str) -> CustomPolicy:
        """Get a custom policy template by name."""
        return get_template(name)


class TraceContext:
    """
    Context manager for automatic timing and error capture.

    Automatically captures start/end times, duration, and error status.

    Example:
        >>> with TraceContext(client, provider="openai", model="gpt-4",
        ...                   input_text="Hello") as ctx:
        ...     response = llm.invoke(prompt)
        ...     ctx.set_output(response.content)
        ...     ctx.set_tokens(response.usage.input, response.usage.output)

    With knowledge base:
        >>> with TraceContext(client, provider="openai", model="gpt-4",
        ...                   input_text="What is X?",
        ...                   knowledge_base="X is defined as...",
        ...                   context_source="retrieval") as ctx:
        ...     response = rag_chain.invoke(query)
        ...     ctx.set_output(response)
    """

    def __init__(
        self,
        client: "CertClient",
        provider: str,
        model: str,
        input_text: str,
        # NEW v0.4.0 parameters
        evaluation_mode: EvaluationMode = "auto",
        knowledge_base: Optional[str] = None,
        context_source: Optional[ContextSource] = None,
        # DEPRECATED
        eval_mode: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize TraceContext.

        Args:
            client: CertClient instance to send traces to
            provider: LLM provider
            model: Model name
            input_text: Input prompt/messages
            evaluation_mode: Evaluation mode (grounded/ungrounded/auto)
            knowledge_base: Source knowledge for grounded evaluation
            context_source: How knowledge was obtained
            eval_mode: DEPRECATED - use evaluation_mode
            context: DEPRECATED - use knowledge_base
            **kwargs: Additional arguments passed to client.trace()
        """
        # Handle deprecated parameters
        if eval_mode is not None:
            warnings.warn(
                "eval_mode is deprecated, use evaluation_mode instead",
                DeprecationWarning,
                stacklevel=2
            )
            evaluation_mode = _map_legacy_mode(eval_mode)

        if context is not None:
            warnings.warn(
                "context is deprecated, use knowledge_base instead",
                DeprecationWarning,
                stacklevel=2
            )
            knowledge_base = context

        self.client = client
        self.provider = provider
        self.model = model
        self.input_text = input_text
        self.evaluation_mode = evaluation_mode
        self.knowledge_base = knowledge_base
        self.context_source = context_source

        # Extract tool_calls from kwargs if present (so we can handle it separately)
        self.tool_calls: Optional[List[Dict[str, Any]]] = kwargs.pop("tool_calls", None)

        # Remaining kwargs
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

    def set_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Set tool calls (enables grounded evaluation with tools)."""
        self.tool_calls = tool_calls

    def set_knowledge_base(self, knowledge: str, source: ContextSource = "retrieval") -> None:
        """Set knowledge base for grounded evaluation."""
        self.knowledge_base = knowledge
        self.context_source = source

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log trace when exiting the context."""
        end_time = datetime.now(timezone.utc)
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
            evaluation_mode=self.evaluation_mode,
            knowledge_base=self.knowledge_base,
            context_source=self.context_source,
            tool_calls=self.tool_calls,
            **self.kwargs,
        )
