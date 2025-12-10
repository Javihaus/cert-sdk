"""
CERT SDK Integrations - Framework-specific handlers for automatic tracing.

Available integrations:
- LangChain: CERTLangChainHandler
- AutoGen: CERTAutoGenHandler
- CrewAI: CERTCrewAIHandler

Each integration provides automatic capture of:
- LLM calls with provider/model detection
- Tool/function invocations with inputs and outputs
- Token usage and timing metrics
- Chain/agent execution flows

Usage:
    from cert import CertClient
    from cert.integrations import CERTLangChainHandler, CERTAutoGenHandler, CERTCrewAIHandler

    client = CertClient(api_key="...", project="my-project")

    # LangChain
    handler = CERTLangChainHandler(client)
    chain.invoke({"input": "Hello"}, config={"callbacks": [handler]})

    # AutoGen
    handler = CERTAutoGenHandler(client)
    with handler.trace_conversation(assistant, user_proxy):
        user_proxy.initiate_chat(assistant, message="Hello")

    # CrewAI
    handler = CERTCrewAIHandler(client)
    with handler.trace_crew(crew, inputs={"topic": "AI"}):
        result = crew.kickoff(inputs={"topic": "AI"})
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid requiring all frameworks
__all__ = [
    "CERTLangChainHandler",
    "CERTAutoGenHandler",
    "CERTCrewAIHandler",
    "LANGCHAIN_AVAILABLE",
    "AUTOGEN_AVAILABLE",
    "CREWAI_AVAILABLE",
]

# Availability flags
LANGCHAIN_AVAILABLE = False
AUTOGEN_AVAILABLE = False
CREWAI_AVAILABLE = False

try:
    from cert.integrations.langchain import CERTLangChainHandler, LANGCHAIN_AVAILABLE
except ImportError:
    CERTLangChainHandler = None  # type: ignore

try:
    from cert.integrations.autogen import CERTAutoGenHandler, AUTOGEN_AVAILABLE
except ImportError:
    CERTAutoGenHandler = None  # type: ignore

try:
    from cert.integrations.crewai import CERTCrewAIHandler, CREWAI_AVAILABLE
except ImportError:
    CERTCrewAIHandler = None  # type: ignore


def __getattr__(name: str):
    """Lazy import for better error messages."""
    if name == "CERTLangChainHandler":
        from cert.integrations.langchain import CERTLangChainHandler
        return CERTLangChainHandler
    elif name == "CERTAutoGenHandler":
        from cert.integrations.autogen import CERTAutoGenHandler
        return CERTAutoGenHandler
    elif name == "CERTCrewAIHandler":
        from cert.integrations.crewai import CERTCrewAIHandler
        return CERTCrewAIHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
