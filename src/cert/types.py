"""
CERT SDK Type Definitions v2.0

Type hints for LLM tracing with two-mode evaluation architecture.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
import warnings


# =============================================================================
# NEW TYPES (v0.4.0+)
# =============================================================================

# Two-mode evaluation: grounded (has knowledge) vs ungrounded (no knowledge)
EvaluationMode = Literal["grounded", "ungrounded", "auto"]

# How the knowledge base was obtained
ContextSource = Literal["retrieval", "tools", "conversation", "user_provided"]

# Span kind for distributed tracing (unchanged)
SpanKind = Literal["CLIENT", "SERVER", "INTERNAL", "PRODUCER", "CONSUMER"]

# Trace status (unchanged)
TraceStatus = Literal["success", "error"]


class ToolCall(TypedDict, total=False):
    """
    Structure for tool/function calls in traces.

    Attributes:
        name: Tool or function name (required)
        input: Input arguments passed to the tool
        output: Result returned by the tool
        error: Error message if the tool call failed
    """
    name: str
    input: Dict[str, Any]
    output: Any
    error: Optional[str]


# =============================================================================
# DEPRECATED TYPES (backwards compatibility - remove in v1.0.0)
# =============================================================================

# Legacy three-mode evaluation
EvalMode = Literal["rag", "generation", "agentic", "auto"]


def _map_legacy_mode(mode: str) -> EvaluationMode:
    """
    Map legacy eval_mode values to new evaluation_mode values.

    Args:
        mode: Legacy mode string

    Returns:
        New evaluation mode
    """
    mapping: Dict[str, EvaluationMode] = {
        "rag": "grounded",
        "generation": "ungrounded",
        "agentic": "grounded",
        "grounded": "grounded",
        "ungrounded": "ungrounded",
        "auto": "auto",
    }
    return mapping.get(mode, "auto")


def _infer_context_source(
    knowledge_base: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]],
    context_source: Optional[ContextSource] = None,
) -> Optional[ContextSource]:
    """
    Infer context source from trace data.

    Args:
        knowledge_base: Explicit knowledge base
        tool_calls: Tool call list
        context_source: Explicit context source

    Returns:
        Inferred or explicit context source
    """
    if context_source:
        return context_source

    # Check tool calls first (more specific)
    if tool_calls:
        has_outputs = any(tc.get("output") is not None for tc in tool_calls)
        if has_outputs:
            return "tools"

    # Has knowledge base but not from tools
    if knowledge_base:
        return "retrieval"

    return None
