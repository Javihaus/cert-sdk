"""
CERT SDK Type Definitions.

Type hints for LLM tracing with evaluation mode support.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


# Evaluation modes for trace classification
EvalMode = Literal["rag", "generation", "agentic", "auto"]


class ToolCall(TypedDict, total=False):
    """
    Structure for tool/function calls in agentic traces.

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
