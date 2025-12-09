"""
CERT SDK - LLM monitoring and evaluation for production applications.

Simple, non-blocking tracing with quality assessment and EU AI Act compliance.

Example:
    >>> from cert import CertClient
    >>> client = CertClient(api_key="cert_xxx", project="my-app")
    >>> client.trace(
    ...     type="rag",
    ...     provider="openai",
    ...     model="gpt-4o",
    ...     input_text="What is CERT?",
    ...     output_text="CERT is...",
    ...     duration_ms=1234,
    ... )
    >>> client.close()
"""

from cert.client import CertClient, __version__
from cert.types import EvalMode, TraceType, ToolCall, TraceData

__all__ = [
    "CertClient",
    "TraceType",
    "EvalMode",
    "ToolCall",
    "TraceData",
    "__version__",
]
