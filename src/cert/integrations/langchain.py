"""
CERT SDK LangChain Integration.

Provides callback handler for automatic tracing of LangChain agents and chains.
Captures tool calls, model responses, and timing information.

Usage:
    from cert import CertClient
    from cert.integrations.langchain import CERTLangChainHandler
    
    client = CertClient(api_key="cert_xxx")
    handler = CERTLangChainHandler(client)
    
    # Option 1: Use with agent invoke
    agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    
    # Option 2: Use as context manager for automatic flush
    with handler:
        agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    
    # Option 3: Manual tracing with intermediate steps
    result = agent.invoke({"input": query}, return_intermediate_steps=True)
    handler.trace_from_result(
        input_text=query,
        result=result,
        provider="openai",
        model="gpt-4"
    )

Requirements:
    pip install langchain langchain-core
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence, TYPE_CHECKING
from uuid import UUID

logger = logging.getLogger(__name__)


# Check if langchain is available
LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
    _LangChainBase = BaseCallbackHandler
except ImportError:
    _LangChainBase = object  # type: ignore[misc,assignment]
    if TYPE_CHECKING:
        # Stubs for type checking when langchain is not installed
        AgentAction = Any
        AgentFinish = Any
        LLMResult = Any
        BaseMessage = Any


@dataclass
class _AgentRun:
    """Internal tracking for a single agent run."""
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


class CERTLangChainHandler(_LangChainBase):  # type: ignore[misc,valid-type]
    """
    LangChain callback handler for CERT tracing.
    
    Automatically captures:
    - Agent/chain input and output
    - Tool invocations with their outputs
    - Model information and token usage
    - Timing information
    
    Tool outputs are automatically extracted as context for
    agentic evaluation, enabling full CERT metrics (SGI, grounding, NLI).
    
    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.langchain import CERTLangChainHandler
        >>> 
        >>> client = CertClient(api_key="cert_xxx")
        >>> handler = CERTLangChainHandler(client)
        >>> 
        >>> # Run agent with handler
        >>> result = agent.invoke(
        ...     {"input": "What's the weather in NYC?"},
        ...     config={"callbacks": [handler]}
        ... )
        >>> 
        >>> # Handler automatically traces with tool outputs as context
    """
    
    name: str = "CERTLangChainHandler"
    
    def __init__(
        self,
        cert_client: "CertClient",
        default_provider: str = "openai",  # More sensible default
        default_model: str = "gpt-4",      # Better than "unknown"
        auto_flush: bool = True,
    ) -> None:
        """
        Initialize the LangChain handler.
        
        Args:
            cert_client: CertClient instance for sending traces
            default_provider: Default provider name if not detected
            default_model: Default model name if not detected
            auto_flush: Flush traces automatically after each run
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install langchain langchain-core"
            )
        
        super().__init__()
        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        
        # Track active runs
        self._runs: Dict[str, _AgentRun] = {}

    # === Helper methods ===

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
            if serialized_id:
                model_name = serialized_id[-1]  # Last element is usually class name
        
        if model_name:
            model = model_name
            provider = self._infer_provider(model_name, serialized)
        
        return provider, model
    
    
    def _infer_provider(self, model_name: str, serialized: Dict[str, Any]) -> str:
        """Infer provider from model name and serialized data."""
        model_lower = model_name.lower()
        serialized_str = str(serialized).lower()
        
        # OpenAI patterns
        if any(p in model_lower for p in ["gpt", "o1", "davinci", "curie", "babbage"]):
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
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        
        # Ollama (local)
        if "ollama" in serialized_str:
            return "ollama"
        
        return self.default_provider
    
    # === Chain/Agent Lifecycle ===
    
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
        input_text = inputs.get("input", inputs.get("query", str(inputs)))
        if isinstance(input_text, list):
            input_text = self._messages_to_string(input_text)
        
        # NEW: Try to extract model info from chain's serialized config
        provider, model = self._extract_model_from_serialized(serialized)
        
        self._runs[run_id_str] = _AgentRun(
            run_id=run_id_str,
            input_text=str(input_text),
            start_time=time.time(),
            provider=provider,  # NEW: Set initial provider
            model=model,        # NEW: Set initial model
            metadata=metadata or {},
        )
        
        logger.debug(f"CERT LangChain: Chain started {run_id_str}")

    
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
        output_text = outputs.get("output", outputs.get("result", str(outputs)))
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False)
        
        # Calculate duration
        duration_ms = (time.time() - run.start_time) * 1000
        
        # NEW: Log warning if we still have unknown provider/model
        if run.provider == "unknown" or run.model == "unknown":
            logger.warning(
                f"CERT LangChain: Could not detect provider/model for chain {run_id_str}. "
                f"Using defaults: provider={run.provider}, model={run.model}. "
                "Consider using trace_from_result() for manual control."
            )
        
        # Send trace to CERT
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=str(output_text),
            duration_ms=duration_ms,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            eval_mode="agentic" if run.tool_calls else "generation",
            tool_calls=run.tool_calls if run.tool_calls else None,
            goal_description=run.input_text,
            metadata={
                "langchain_run_id": run.run_id,
                "langchain_detected_provider": run.provider,
                "langchain_detected_model": run.model,
                **run.metadata,
            },
        )
        
        if self.auto_flush:
            self.cert_client.flush()

    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error."""
        run_id_str = str(run_id)
        run = self._runs.pop(run_id_str, None)
        
        if run is None:
            return
        
        duration_ms = (time.time() - run.start_time) * 1000
        
        # Trace the error
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=f"ERROR: {str(error)}",
            duration_ms=duration_ms,
            eval_mode="agentic" if run.tool_calls else "generation",
            tool_calls=run.tool_calls if run.tool_calls else None,
            metadata={
                "langchain_run_id": run.run_id,
                "error": str(error),
                **run.metadata,
            },
        )
        
        if self.auto_flush:
            self.cert_client.flush()
        
        logger.warning(f"CERT LangChain: Chain {run_id_str} errored: {error}")
    
    # === Tool Lifecycle ===
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool invocation start."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)
        
        if run is None:
            return
        
        # Parse input
        try:
            input_parsed = json.loads(input_str) if input_str.startswith("{") else input_str
        except json.JSONDecodeError:
            input_parsed = input_str
        
        tool_call = {
            "name": serialized.get("name", "unknown_tool"),
            "input": input_parsed if isinstance(input_parsed, dict) else {"query": input_parsed},
            "output": None,
            "error": None,
        }
        
        run.current_tool = tool_call
        logger.debug(f"CERT LangChain: Tool {tool_call['name']} started")
    
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
        if isinstance(output, (dict, list)):
            run.current_tool["output"] = output
        else:
            run.current_tool["output"] = str(output)
        
        run.tool_calls.append(run.current_tool)
        logger.debug(f"CERT LangChain: Tool {run.current_tool['name']} completed")
        run.current_tool = None
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool error."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)
        
        if run is None or run.current_tool is None:
            return
        
        run.current_tool["error"] = str(error)
        run.tool_calls.append(run.current_tool)
        logger.debug(f"CERT LangChain: Tool {run.current_tool['name']} errored")
        run.current_tool = None
    
    # === LLM Lifecycle (for token tracking) ===
    
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
            model_name = serialized.get("id", ["unknown"])[-1]
        
        run.model = model_name
        
        # Infer provider from model name
        if "gpt" in model_name.lower() or "openai" in str(serialized).lower():
            run.provider = "openai"
        elif "claude" in model_name.lower() or "anthropic" in str(serialized).lower():
            run.provider = "anthropic"
        elif "gemini" in model_name.lower() or "google" in str(serialized).lower():
            run.provider = "google"
        else:
            run.provider = self.default_provider
    
    def on_llm_end(
        self,
        response: LLMResult,
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
        
        # Extract token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            run.prompt_tokens += usage.get("prompt_tokens", 0)
            run.completion_tokens += usage.get("completion_tokens", 0)
    
    # === Agent Events ===
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action (before tool execution)."""
        # Tool start will handle this
        pass
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish."""
        parent_id = str(parent_run_id) if parent_run_id else None
        run = self._find_parent_run(parent_id)
        
        if run is None:
            return
        
        run.output_text = finish.return_values.get("output", str(finish.return_values))
    
    # === Helper Methods ===
    
    def _find_parent_run(self, parent_id: Optional[str]) -> Optional[_AgentRun]:
        """Find the parent run for nested events."""
        if parent_id and parent_id in self._runs:
            return self._runs[parent_id]
        # If no parent, return the most recent run
        if self._runs:
            return list(self._runs.values())[-1]
        return None
    
    def _messages_to_string(self, messages: List[Any]) -> str:
        """Convert message list to string."""
        parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                parts.append(str(msg.content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    
    # === Static Helper for Manual Tracing ===
    
    def trace_from_result(
        self,
        input_text: str,
        result: Dict[str, Any],
        provider: str = "openai",
        model: str = "unknown",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Manually trace from a LangChain result with intermediate steps.
        
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
        
        for action, observation in intermediate_steps:
            if hasattr(action, "tool"):
                tool_call = {
                    "name": action.tool,
                    "input": action.tool_input if isinstance(action.tool_input, dict) 
                             else {"input": str(action.tool_input)},
                    "output": observation if isinstance(observation, (dict, list, str, int, float))
                             else str(observation),
                }
                tool_calls.append(tool_call)
        
        self.cert_client.trace(
            provider=provider,
            model=model,
            input_text=input_text,
            output_text=str(output_text),
            duration_ms=duration_ms or 0,
            eval_mode="agentic" if tool_calls else "generation",
            tool_calls=tool_calls if tool_calls else None,
            goal_description=input_text,
            metadata=metadata or {},
        )
        
        if self.auto_flush:
            self.cert_client.flush()
    
    # === Context Manager ===
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush any pending traces."""
        self.cert_client.flush()
        return False
