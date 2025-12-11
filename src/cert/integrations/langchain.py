"""
CERT LangChain Integration - Automatic tracing for LangChain chains and agents.

Provides seamless integration with LangChain framework for:
- Automatic capture of chain/agent executions
- Tool call tracking with inputs and outputs
- Model and provider detection from LLM configurations
- Token usage and timing metrics

Usage:
    from cert import CertClient
    from cert.integrations.langchain import CERTLangChainHandler

    client = CertClient(api_key="...", project="my-project")
    handler = CERTLangChainHandler(client)

    # Use as a callback handler
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
    result = chain.invoke({"input": "Hello!"})

    # Or with agents
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
    result = agent_executor.invoke({"input": "What's the weather?"})
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from cert.types import EvaluationMode, ContextSource
from uuid import UUID

logger = logging.getLogger(__name__)

# Check if LangChain is available
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older import path
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        # Type stubs for when LangChain is not installed
        BaseCallbackHandler = object
        LLMResult = Any

if TYPE_CHECKING:
    from cert.client import CertClient


@dataclass
class _AgentRun:
    """Internal tracking for a single agent/chain run."""

    run_id: str
    input_text: str
    start_time: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    current_tool: Optional[Dict[str, Any]] = None
    output_text: Optional[str] = None
    provider: str = "unknown"
    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW v0.4.0 fields
    knowledge_base: Optional[str] = None
    context_source: Optional[ContextSource] = None

    def build_knowledge_base(self) -> Optional[str]:
        """
        Build knowledge base from tool outputs.

        Returns:
            Concatenated knowledge string or None
        """
        parts = []
        for tc in self.tool_calls:
            if tc.get("output") is not None:
                name = tc.get("name", "tool")
                output = tc["output"]
                if isinstance(output, (dict, list)):
                    output_str = json.dumps(output, ensure_ascii=False, default=str)
                else:
                    output_str = str(output)
                parts.append(f"[{name}]: {output_str}")
        return "\n\n".join(parts) if parts else None


class CERTLangChainHandler(BaseCallbackHandler):
    """CERT callback handler for LangChain chains and agents.

    Provides automatic tracing of LangChain executions, including:
    - Chain and agent input/output capture
    - Tool/function calls with inputs and outputs
    - Model information and token usage
    - Timing information

    Automatically extracts knowledge base from tool outputs for
    grounded evaluation.

    Args:
        cert_client: Initialized CertClient instance
        default_provider: Default provider if detection fails (default: "openai")
        default_model: Default model if detection fails (default: "gpt-4")
        auto_flush: Whether to flush after each chain completion (default: True)
        auto_extract_knowledge: Automatically extract knowledge from tool outputs
                                (default: True)

    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.langchain import CERTLangChainHandler
        >>>
        >>> client = CertClient(api_key="...", project="langchain-demo")
        >>> handler = CERTLangChainHandler(client)
        >>>
        >>> # Use with any LangChain chain
        >>> chain = prompt | llm | parser
        >>> chain.invoke({"input": "Hello!"}, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        cert_client: "CertClient",
        default_provider: str = "openai",
        default_model: str = "gpt-4",
        auto_flush: bool = True,
        auto_extract_knowledge: bool = True,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain langchain-core"
            )

        super().__init__()
        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        self.auto_extract_knowledge = auto_extract_knowledge

        # Track active runs
        self._runs: Dict[str, _AgentRun] = {}

    # =========================================================================
    # Model/Provider Detection
    # =========================================================================

    def _extract_model_from_serialized(
        self, serialized: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Extract provider and model from serialized chain/LLM config.

        Handles various LangChain patterns:
        - Direct LLM configs (ChatOpenAI, ChatAnthropic, etc.)
        - Chain configs with nested LLM
        - Agent configs
        """
        provider = self.default_provider
        model = self.default_model

        # Common locations for model info
        kwargs = serialized.get("kwargs", {})

        # Try direct model_name (most common)
        model_name = kwargs.get("model_name") or kwargs.get("model")

        # Try nested LLM config (common in chains)
        if not model_name:
            llm_config = kwargs.get("llm", {})
            if isinstance(llm_config, dict):
                llm_kwargs = llm_config.get("kwargs", {})
                model_name = llm_kwargs.get("model_name") or llm_kwargs.get("model")

        # Try from serialized ID (e.g., ["langchain", "chat_models", "openai", "ChatOpenAI"])
        if not model_name:
            serialized_id = serialized.get("id", [])
            if serialized_id and isinstance(serialized_id, list):
                # Last element is usually class name like "ChatOpenAI"
                class_name = serialized_id[-1] if serialized_id else ""
                # Try to infer from class name
                if class_name:
                    inferred = self._infer_from_class_name(class_name)
                    if inferred[1] != self.default_model:
                        return inferred

        if model_name:
            model = model_name
            provider = self._infer_provider(model_name, serialized)

        return provider, model

    def _infer_provider(self, model_name: str, serialized: Dict[str, Any]) -> str:
        """Infer provider from model name and serialized data."""
        if not model_name:
            return self.default_provider

        model_lower = model_name.lower()
        serialized_str = str(serialized).lower()

        # OpenAI patterns
        if any(p in model_lower for p in ["gpt", "o1-", "o3-", "davinci", "curie", "babbage"]):
            return "openai"
        if "openai" in serialized_str or "chatopenai" in serialized_str:
            return "openai"

        # Anthropic patterns
        if any(p in model_lower for p in ["claude", "anthropic"]):
            return "anthropic"
        if "anthropic" in serialized_str or "chatanthropic" in serialized_str:
            return "anthropic"

        # Google patterns
        if any(p in model_lower for p in ["gemini", "palm", "bard"]):
            return "google"
        if "google" in serialized_str or "chatgoogle" in serialized_str:
            return "google"

        # Azure OpenAI
        if "azure" in model_lower or "azure" in serialized_str:
            return "azure"

        # AWS Bedrock
        if "bedrock" in serialized_str:
            return "bedrock"

        # Mistral
        if any(p in model_lower for p in ["mistral", "mixtral"]):
            return "mistral"

        # Ollama (local)
        if "ollama" in serialized_str:
            return "ollama"

        # Groq
        if "groq" in serialized_str:
            return "groq"

        # Cohere
        if "command" in model_lower or "cohere" in serialized_str:
            return "cohere"

        return self.default_provider

    def _is_class_name(self, name: str) -> bool:
        """Check if a string looks like a class name rather than a model name.

        LangChain serialized IDs contain class names like "ChatOpenAI", "RunnableSequence",
        "StrOutputParser", etc. These should not be used as model names.
        """
        if not name:
            return False

        # Common LangChain class name patterns
        class_patterns = [
            "runnable", "chain", "parser", "output", "sequence",
            "prompt", "template", "memory", "retriever", "tool",
            "agent", "executor", "lambda", "passthrough", "branch",
            "router", "fallback", "retry", "with", "bind", "assign",
        ]

        name_lower = name.lower()

        # Check for common patterns
        if any(pattern in name_lower for pattern in class_patterns):
            return True

        # Check for PascalCase (likely a class name)
        if name[0].isupper() and any(c.isupper() for c in name[1:]):
            # But allow model names like "GPT-4" or "Claude-3"
            if not any(m in name_lower for m in ["gpt", "claude", "gemini", "llama", "mistral"]):
                return True

        return False

    def _infer_from_class_name(self, class_name: str) -> Tuple[str, str]:
        """Infer provider from LangChain class name."""
        class_lower = class_name.lower()

        if "openai" in class_lower:
            return "openai", "gpt-4"
        elif "anthropic" in class_lower:
            return "anthropic", "claude-3-sonnet"
        elif "google" in class_lower or "gemini" in class_lower:
            return "google", "gemini-pro"
        elif "bedrock" in class_lower:
            return "bedrock", "anthropic.claude-v2"
        elif "azure" in class_lower:
            return "azure", "gpt-4"
        elif "ollama" in class_lower:
            return "ollama", "llama2"
        elif "groq" in class_lower:
            return "groq", "mixtral-8x7b"
        elif "mistral" in class_lower:
            return "mistral", "mistral-large"
        elif "cohere" in class_lower:
            return "cohere", "command"

        return self.default_provider, self.default_model

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_parent_run(self, parent_id: Optional[str]) -> Optional[_AgentRun]:
        """Find the parent run for nested events."""
        if parent_id and parent_id in self._runs:
            return self._runs[parent_id]
        # Return most recent run as fallback
        if self._runs:
            return list(self._runs.values())[-1]
        return None

    def _messages_to_string(self, messages: List[Any]) -> str:
        """Convert a list of messages to a string."""
        parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", str(msg))
            else:
                content = str(msg)
            parts.append(str(content))
        return "\n".join(parts)

    def _serialize_output(self, output: Any) -> str:
        """Serialize output to string."""
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            return json.dumps(output, ensure_ascii=False, default=str)
        if hasattr(output, "content"):
            return str(output.content)
        return str(output)

    # =========================================================================
    # Chain Lifecycle Callbacks
    # =========================================================================

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start event."""
        run_id_str = str(run_id)

        # Extract input text
        input_text = inputs.get("input", inputs.get("query", inputs.get("question", "")))
        if not input_text:
            # Try to get any string value
            for v in inputs.values():
                if isinstance(v, str):
                    input_text = v
                    break
        if isinstance(input_text, list):
            input_text = self._messages_to_string(input_text)

        # Extract model info from chain's serialized config
        provider, model = self._extract_model_from_serialized(serialized)

        # Get chain name
        chain_name = serialized.get("name", "")
        if not chain_name:
            serialized_id = serialized.get("id", [])
            if serialized_id:
                chain_name = serialized_id[-1]

        self._runs[run_id_str] = _AgentRun(
            run_id=run_id_str,
            input_text=str(input_text) if input_text else str(inputs),
            start_time=time.time(),
            provider=provider,
            model=model,
            metadata={
                "chain_name": chain_name,
                "tags": tags or [],
                **(metadata or {}),
            },
        )

        logger.debug(f"CERT LangChain: Chain started {run_id_str} ({chain_name})")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end event and send trace."""
        run_id_str = str(run_id)
        run = self._runs.pop(run_id_str, None)

        if run is None:
            logger.debug(f"CERT LangChain: No run found for {run_id_str}")
            return

        # Extract output text
        output_text = outputs.get("output", outputs.get("result", outputs.get("answer", "")))
        if not output_text:
            # Try to get any value
            for v in outputs.values():
                if v:
                    output_text = v
                    break
        output_text = self._serialize_output(output_text)

        # Calculate duration
        duration_ms = (time.time() - run.start_time) * 1000

        # Warn if we couldn't detect model
        if run.provider == "unknown" or run.model == "unknown":
            logger.warning(
                f"CERT LangChain: Could not detect provider/model for chain {run_id_str}. "
                f"Using defaults: provider={run.provider}, model={run.model}. "
                "Consider using trace_from_result() for manual control."
            )

        # Build knowledge base from tool outputs
        knowledge_base: Optional[str] = None
        context_source: Optional[ContextSource] = None

        if self.auto_extract_knowledge and run.tool_calls:
            knowledge_base = run.build_knowledge_base()
            if knowledge_base:
                context_source = "tools"

        # Determine evaluation mode
        evaluation_mode: EvaluationMode = "grounded" if knowledge_base else "ungrounded"

        # Send trace to CERT
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            evaluation_mode=evaluation_mode,
            knowledge_base=knowledge_base,
            context_source=context_source,
            tool_calls=run.tool_calls if run.tool_calls else None,
            goal_description=run.input_text,
            metadata={
                "langchain_run_id": run.run_id,
                "langchain_tool_call_count": len(run.tool_calls),
                "langchain_detected_provider": run.provider,
                "langchain_detected_model": run.model,
                **run.metadata,
            },
        )

        if self.auto_flush:
            self.cert_client.flush()

        logger.debug(
            f"CERT LangChain: Traced chain {run_id_str} "
            f"with {len(run.tool_calls)} tool calls"
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error event."""
        run_id_str = str(run_id)
        run = self._runs.pop(run_id_str, None)

        if run is None:
            return

        # Calculate duration
        duration_ms = (time.time() - run.start_time) * 1000

        # Send error trace
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=f"Error: {str(error)}",
            duration_ms=duration_ms,
            status="error",
            error_message=str(error),
            evaluation_mode="ungrounded",
            metadata={
                "langchain_run_id": run.run_id,
                "langchain_error": True,
                **run.metadata,
            },
        )

        if self.auto_flush:
            self.cert_client.flush()

        logger.warning(f"CERT LangChain: Chain {run_id_str} failed: {error}")

    # =========================================================================
    # LLM Lifecycle Callbacks
    # =========================================================================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM invocation start."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)

        if run is None:
            return

        # Extract model info from serialized
        model_name = serialized.get("kwargs", {}).get("model_name", "")
        if not model_name:
            model_name = serialized.get("kwargs", {}).get("model", "")

        # Only update model if we found a valid model name (not a class name or "unknown")
        # This prevents overwriting the model detected in on_chain_start
        if model_name and not self._is_class_name(model_name):
            run.model = model_name
            # Infer provider from the newly detected model
            run.provider = self._infer_provider(model_name, serialized)
        elif run.model == "unknown" or run.model == self.default_model:
            # Only try class name inference if we don't already have a good model
            serialized_id = serialized.get("id", [])
            if serialized_id:
                class_name = serialized_id[-1]
                inferred_provider, inferred_model = self._infer_from_class_name(class_name)
                # Only update if we got something better than current defaults
                if inferred_model != self.default_model:
                    run.provider = inferred_provider
                    run.model = inferred_model

        logger.debug(f"CERT LangChain: LLM start - {run.provider}/{run.model}")

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM invocation end."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)

        if run is None:
            return

        # Extract token usage
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage", {})

        run.prompt_tokens += token_usage.get("prompt_tokens", 0)
        run.completion_tokens += token_usage.get("completion_tokens", 0)

        logger.debug(
            f"CERT LangChain: LLM end - tokens: {run.prompt_tokens}+{run.completion_tokens}"
        )

    # =========================================================================
    # Tool Lifecycle Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool invocation start."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)

        if run is None:
            return

        tool_name = serialized.get("name", "unknown_tool")

        # Parse input
        try:
            tool_input = json.loads(input_str) if isinstance(input_str, str) else input_str
        except json.JSONDecodeError:
            tool_input = {"input": input_str}

        run.current_tool = {
            "name": tool_name,
            "input": tool_input,
            "start_time": time.time(),
        }

        logger.debug(f"CERT LangChain: Tool start - {tool_name}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool invocation end."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)

        if run is None or run.current_tool is None:
            return

        # Serialize output
        tool_output: Any
        if isinstance(output, (dict, list)):
            tool_output = output
        elif hasattr(output, "dict"):
            tool_output = output.dict()
        else:
            tool_output = str(output)

        run.current_tool["output"] = tool_output
        run.current_tool["duration_ms"] = (
            time.time() - run.current_tool.get("start_time", time.time())
        ) * 1000
        del run.current_tool["start_time"]

        run.tool_calls.append(run.current_tool)
        run.current_tool = None

        logger.debug(f"CERT LangChain: Tool end - {len(run.tool_calls)} total calls")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool invocation error."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)

        if run is None or run.current_tool is None:
            return

        run.current_tool["error"] = str(error)
        run.current_tool["duration_ms"] = (
            time.time() - run.current_tool.get("start_time", time.time())
        ) * 1000
        del run.current_tool["start_time"]

        run.tool_calls.append(run.current_tool)
        run.current_tool = None

        logger.warning(f"CERT LangChain: Tool error - {error}")

    # =========================================================================
    # Agent Callbacks
    # =========================================================================

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action (tool selection)."""
        # Actions are typically handled via on_tool_start
        pass

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish."""
        # Final output is typically handled via on_chain_end
        pass

    # =========================================================================
    # Manual Tracing
    # =========================================================================

    def trace_from_result(
        self,
        input_text: str,
        result: Dict[str, Any],
        provider: str = "openai",
        model: str = "gpt-4",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually trace from a LangChain result with intermediate steps.

        Use this when you need more control over tracing, or when using
        agents that return intermediate steps.

        Args:
            input_text: The input query
            result: Result dict from agent.invoke(return_intermediate_steps=True)
            provider: LLM provider name
            model: Model name
            duration_ms: Request duration (optional)
            metadata: Additional metadata

        Example:
            >>> result = agent.invoke(
            ...     {"input": "Calculate 2+2"},
            ...     return_intermediate_steps=True
            ... )
            >>> handler.trace_from_result(
            ...     input_text="Calculate 2+2",
            ...     result=result,
            ...     provider="openai",
            ...     model="gpt-4"
            ... )
        """
        # Extract output
        output_text = result.get("output", str(result))

        # Extract tool calls from intermediate steps
        tool_calls = []
        intermediate_steps = result.get("intermediate_steps", [])

        for step in intermediate_steps:
            # Handle (AgentAction, observation) tuples
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                action, observation = step[0], step[1]
                if hasattr(action, "tool"):
                    tool_call = {
                        "name": action.tool,
                        "input": (
                            action.tool_input
                            if isinstance(action.tool_input, dict)
                            else {"input": str(action.tool_input)}
                        ),
                        "output": (
                            observation
                            if isinstance(observation, (dict, list, str, int, float))
                            else str(observation)
                        ),
                    }
                    tool_calls.append(tool_call)

        # Build knowledge base from tool outputs
        knowledge_base: Optional[str] = None
        context_source: Optional[ContextSource] = None

        if self.auto_extract_knowledge and tool_calls:
            parts = []
            for tc in tool_calls:
                if tc.get("output"):
                    output_str = (
                        json.dumps(tc["output"], default=str)
                        if isinstance(tc["output"], (dict, list))
                        else str(tc["output"])
                    )
                    parts.append(f"[{tc['name']}]: {output_str}")
            if parts:
                knowledge_base = "\n\n".join(parts)
                context_source = "tools"

        # Determine evaluation mode
        evaluation_mode: EvaluationMode = "grounded" if knowledge_base else "ungrounded"

        self.cert_client.trace(
            provider=provider,
            model=model,
            input_text=input_text,
            output_text=str(output_text),
            duration_ms=duration_ms or 0,
            evaluation_mode=evaluation_mode,
            knowledge_base=knowledge_base,
            context_source=context_source,
            tool_calls=tool_calls if tool_calls else None,
            goal_description=input_text,
            metadata=metadata or {},
        )

        if self.auto_flush:
            self.cert_client.flush()

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "CERTLangChainHandler":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - flush pending traces."""
        self.cert_client.flush()
