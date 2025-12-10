"""
CERT SDK Framework Integrations.

Optional integrations for popular agent frameworks:
- LangChain
- AutoGen
- CrewAI

These provide callback handlers and adapters that automatically
capture tool calls and trace them to CERT with proper context extraction.

Installation:
    pip install cert-sdk[langchain]  # For LangChain integration
    pip install cert-sdk[autogen]    # For AutoGen integration
    pip install cert-sdk[crewai]     # For CrewAI integration
    pip install cert-sdk[all]        # All integrations
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid requiring all framework dependencies
if TYPE_CHECKING:
    from cert.integrations.langchain import CERTLangChainHandler
    from cert.integrations.autogen import CERTAutoGenHandler
    from cert.integrations.crewai import CERTCrewAIHandler


def get_langchain_handler():
    """Get LangChain callback handler for CERT tracing."""
    from cert.integrations.langchain import CERTLangChainHandler
    return CERTLangChainHandler


def get_autogen_handler():
    """Get AutoGen handler for CERT tracing."""
    from cert.integrations.autogen import CERTAutoGenHandler
    return CERTAutoGenHandler


def get_crewai_handler():
    """Get CrewAI handler for CERT tracing."""
    from cert.integrations.crewai import CERTCrewAIHandler
    return CERTCrewAIHandler


__all__ = [
    "get_langchain_handler",
    "get_autogen_handler", 
    "get_crewai_handler",
]
