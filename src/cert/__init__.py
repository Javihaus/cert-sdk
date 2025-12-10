"""
CERT SDK for LLM monitoring and reliability evaluation.

The CERT SDK provides non-blocking tracing for production AI systems
with automatic context extraction for agentic pipelines.

Quick Start:
    >>> from cert import CertClient
    >>> 
    >>> client = CertClient(api_key="cert_xxx")
    >>> 
    >>> # RAG Mode - with explicit context
    >>> client.trace(
    ...     provider="anthropic",
    ...     model="claude-sonnet-4",
    ...     input_text="What is the capital of France?",
    ...     output_text="The capital of France is Paris.",
    ...     duration_ms=234,
    ...     context="France is a country in Europe. Its capital is Paris.",
    ... )
    >>> 
    >>> # Agentic Mode - tool outputs become context automatically
    >>> client.trace(
    ...     provider="openai",
    ...     model="gpt-4",
    ...     input_text="What's the weather in NYC?",
    ...     output_text="It's 72°F and sunny in NYC.",
    ...     duration_ms=1500,
    ...     tool_calls=[
    ...         {"name": "weather_api", "output": {"temp": 72, "condition": "sunny"}}
    ...     ]
    ... )

Framework Integrations:
    The SDK provides optional integrations for popular agent frameworks:
    
    >>> # LangChain
    >>> from cert.integrations.langchain import CERTLangChainHandler
    >>> handler = CERTLangChainHandler(client)
    >>> agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    
    >>> # AutoGen
    >>> from cert.integrations.autogen import CERTAutoGenHandler
    >>> handler = CERTAutoGenHandler(client)
    >>> result = handler.trace_conversation(initiator, recipient, message)
    
    >>> # CrewAI
    >>> from cert.integrations.crewai import CERTCrewAIHandler
    >>> handler = CERTCrewAIHandler(client)
    >>> result = handler.trace_crew(crew, inputs={"topic": "..."})

Evaluation Modes:
    - "rag": Full metrics (semantic, NLI, grounding, SGI) with explicit context
    - "agentic": Tool outputs serve as context, enabling full RAG metrics
    - "generation": Reduced metrics without context (quality signals only)
    - "auto": Automatically detect based on context and tool_calls
"""

from cert.client import CertClient, extract_context_from_tool_calls
from cert.types import EvalMode, ToolCall

__version__ = "0.1.0"

__all__ = [
    # Core client
    "CertClient",
    
    # Utility functions
    "extract_context_from_tool_calls",
    
    # Type definitions
    "EvalMode",
    "ToolCall",
]
