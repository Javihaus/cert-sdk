"""
CERT SDK Type Definitions.

Type hints for LLM tracing with evaluation mode support.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


# User-declared trace type (required)
TraceType = Literal["rag", "generation", "agentic"]

# Auto-detected evaluation mode (backwards compatibility)
EvalMode = Literal["rag", "generation", "agentic", "auto"]


class ToolCall(TypedDict, total=False):
    """
    Structure for tool/function calls in agentic traces.

    Attributes:
        name: Tool or function name (required)
        input: Input arguments passed to the tool
        output: Result returned by the tool
        error: Error message if the tool call failed
        duration_ms: Tool execution time in milliseconds
    """

    name: str  # Required
    input: Dict[str, Any]
    output: Any
    error: Optional[str]
    duration_ms: Optional[float]


class TraceData(TypedDict, total=False):
    """
    Complete trace structure sent to CERT API.

    This matches the expected API payload format.
    """

    # Required fields
    id: str
    type: TraceType
    provider: str
    model: str
    input: str
    output: str
    durationMs: float
    timestamp: str

    # Token counts
    promptTokens: int
    completionTokens: int

    # Classification
    evalMode: str
    projectName: str

    # Mode-specific fields
    context: str  # RAG
    outputSchema: Dict[str, Any]  # Generation
    toolCalls: List[ToolCall]  # Agentic
    goalDescription: str  # Agentic

    # Additional data
    metadata: Dict[str, Any]
