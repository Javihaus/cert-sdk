"""
CERT SDK - LLM Monitoring for Production Applications

Two-mode evaluation architecture (v1.0.0):
- Grounded: Has knowledge_base -> full metric suite
- Ungrounded: No knowledge_base -> basic metrics

Bias detection (v1.0.0):
- Demographic bias: Standard categories (gender, race, etc.)
- Custom policies: Domain-specific rules
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
    # New types (v1.0.0)
    EvaluationMode,
    ContextSource,
    SpanKind,
    ToolCall,
    TraceStatus,
    # Deprecated types
    EvalMode,
)

from cert.bias import (
    # Enums
    BiasSeverity,
    BiasConsensus,
    # Configuration
    DemographicBiasConfig,
    DemographicCategory,
    CustomPolicy,
    PolicyDimension,
    # Results
    CategoryResult,
    DimensionResult,
    BiasEvaluationResult,
    # Templates
    get_template as get_policy_template,
    list_templates as list_policy_templates,
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
    # Bias detection (v1.0.0)
    "BiasSeverity",
    "BiasConsensus",
    "DemographicBiasConfig",
    "DemographicCategory",
    "CustomPolicy",
    "PolicyDimension",
    "CategoryResult",
    "DimensionResult",
    "BiasEvaluationResult",
    "get_policy_template",
    "list_policy_templates",
    # Version
    "__version__",
]
