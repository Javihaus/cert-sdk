"""
CERT SDK - LLM Monitoring for Production Applications

Two-mode evaluation architecture (v0.4.0+):
- Grounded: Has knowledge_base -> full metric suite
- Ungrounded: No knowledge_base -> basic metrics
"""

from cert.client import (
    CertClient,
    TraceContext,
    extract_knowledge_from_tool_calls,
    # Backwards compatibility
    extract_context_from_tool_calls,
    __version__,
)

from cert.types import (
    # New types (v0.4.0+)
    EvaluationMode,
    ContextSource,
    SpanKind,
    ToolCall,
    TraceStatus,
    # Deprecated types
    EvalMode,
)

__all__ = [
    # Client
    "CertClient",
    "TraceContext",
    # Types (new)
    "EvaluationMode",
    "ContextSource",
    "SpanKind",
    "ToolCall",
    "TraceStatus",
    # Utilities
    "extract_knowledge_from_tool_calls",
    # Backwards compatibility (deprecated)
    "EvalMode",
    "extract_context_from_tool_calls",
    # Version
    "__version__",
]
