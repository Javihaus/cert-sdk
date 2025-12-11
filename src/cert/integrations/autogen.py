"""
CERT AutoGen Integration - Automatic tracing for AutoGen agents and conversations.

Provides seamless integration with Microsoft AutoGen framework for:
- Automatic capture of agent conversations
- Tool/function call tracking with inputs and outputs
- Model and provider detection from agent configurations
- Token usage and timing metrics

Usage:
    from cert import CertClient
    from cert.integrations.autogen import CERTAutoGenHandler

    client = CertClient(api_key="...", project="my-project")
    handler = CERTAutoGenHandler(client)

    # Option 1: Wrap individual conversations
    with handler.trace_conversation(assistant, user_proxy, metadata={"task": "coding"}):
        user_proxy.initiate_chat(assistant, message="Write a hello world program")

    # Option 2: Wrap agents for automatic tracing
    handler.wrap_agent(assistant)
    handler.wrap_agent(user_proxy)
    user_proxy.initiate_chat(assistant, message="Write a hello world program")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from cert.types import EvaluationMode, ContextSource

logger = logging.getLogger(__name__)

# Check if AutoGen is available
try:
    from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Type stubs for when AutoGen is not installed
    Agent = Any
    AssistantAgent = Any
    ConversableAgent = Any
    UserProxyAgent = Any

if TYPE_CHECKING:
    from cert.client import CertClient


@dataclass
class _ConversationRun:
    """Internal tracking for a single conversation run."""

    run_id: str
    input_text: str
    start_time: float
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    current_tool: Optional[Dict[str, Any]] = None
    output_text: Optional[str] = None
    provider: str = "unknown"
    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW v0.4.0 fields
    knowledge_base: Optional[str] = None
    context_source: Optional[ContextSource] = None

    def build_knowledge_base(self) -> Optional[str]:
        """
        Build unified knowledge base from conversation history and tool outputs.

        Returns:
            Concatenated knowledge string or None
        """
        parts = []

        # Add tool outputs
        for tc in self.tool_calls:
            if tc.get("output") is not None:
                name = tc.get("name", "tool")
                output = tc["output"]
                if isinstance(output, (dict, list)):
                    output_str = json.dumps(output, ensure_ascii=False, default=str)
                else:
                    output_str = str(output)
                parts.append(f"[{name}]: {output_str}")

        # Add prior assistant messages as conversation context
        for msg in self.messages[:-1]:  # Exclude current message
            if msg.get("role") == "assistant" and msg.get("content"):
                parts.append(f"[prior_response]: {msg['content']}")

        return "\n\n".join(parts) if parts else None


class CERTAutoGenHandler:
    """CERT callback handler for AutoGen agents.

    Provides automatic tracing of AutoGen agent conversations, including:
    - Message exchanges between agents
    - Function/tool calls with inputs and outputs
    - Model information and token usage
    - Timing information

    Automatically extracts knowledge base from tool outputs and
    conversation history for grounded evaluation.

    Args:
        cert_client: Initialized CertClient instance
        default_provider: Default provider if detection fails (default: "openai")
        default_model: Default model if detection fails (default: "gpt-4")
        auto_flush: Whether to flush after each conversation (default: True)
        auto_extract_knowledge: Automatically extract knowledge from tool outputs
                                and conversation history (default: True)

    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.autogen import CERTAutoGenHandler
        >>>
        >>> client = CertClient(api_key="...", project="autogen-demo")
        >>> handler = CERTAutoGenHandler(client)
        >>>
        >>> with handler.trace_conversation(assistant, user_proxy):
        ...     user_proxy.initiate_chat(assistant, message="Hello!")
    """

    def __init__(
        self,
        cert_client: "CertClient",
        default_provider: str = "openai",
        default_model: str = "gpt-4",
        auto_flush: bool = True,
        auto_extract_knowledge: bool = True,
    ) -> None:
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. Install it with: pip install pyautogen"
            )

        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        self.auto_extract_knowledge = auto_extract_knowledge

        # Track active conversation runs
        self._runs: Dict[str, _ConversationRun] = {}

        # Track wrapped agents for cleanup
        self._wrapped_agents: Dict[int, Dict[str, Any]] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    @contextmanager
    def trace_conversation(
        self,
        *agents: Agent,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager to trace a conversation between agents.

        Wraps the agents, traces the conversation, and restores original
        behavior on exit.

        Args:
            *agents: AutoGen agents participating in the conversation
            metadata: Additional metadata to include in the trace

        Yields:
            The conversation run ID

        Example:
            >>> with handler.trace_conversation(assistant, user_proxy) as run_id:
            ...     result = user_proxy.initiate_chat(assistant, message="Hello")
            ...     print(f"Traced as: {run_id}")
        """
        run_id = str(uuid.uuid4())

        # Detect model from first agent with LLM config
        provider, model = self._extract_model_from_agents(agents)

        run = _ConversationRun(
            run_id=run_id,
            input_text="",  # Will be set from first message
            start_time=time.time(),
            provider=provider,
            model=model,
            metadata=metadata or {},
        )
        self._runs[run_id] = run

        # Wrap all agents
        original_hooks = {}
        for agent in agents:
            original_hooks[id(agent)] = self._wrap_agent_hooks(agent, run_id)

        try:
            yield run_id
        finally:
            # Calculate duration
            run.duration_ms = (time.time() - run.start_time) * 1000

            # Extract final output
            if not run.output_text:
                run.output_text = self._extract_final_output(run)

            # Send trace
            self._send_trace(run)

            # Restore original hooks
            for agent in agents:
                self._restore_agent_hooks(agent, original_hooks.get(id(agent), {}))

            # Cleanup
            del self._runs[run_id]

    def wrap_agent(self, agent: Agent) -> None:
        """Wrap an agent for automatic tracing of all its conversations.

        This modifies the agent in-place to trace all future conversations.
        Use unwrap_agent() to restore original behavior.

        Args:
            agent: AutoGen agent to wrap

        Example:
            >>> handler.wrap_agent(assistant)
            >>> handler.wrap_agent(user_proxy)
            >>> # All future chats will be traced
            >>> user_proxy.initiate_chat(assistant, message="Hello")
        """
        agent_id = id(agent)
        if agent_id in self._wrapped_agents:
            logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} is already wrapped")
            return

        # Create a persistent run for this agent
        run_id = str(uuid.uuid4())
        provider, model = self._extract_model_info(agent)

        run = _ConversationRun(
            run_id=run_id,
            input_text="",
            start_time=time.time(),
            provider=provider,
            model=model,
            metadata={"agent_name": getattr(agent, "name", "unknown")},
        )
        self._runs[run_id] = run

        # Store original hooks and wrap
        original_hooks = self._wrap_agent_hooks(agent, run_id)
        self._wrapped_agents[agent_id] = {
            "original_hooks": original_hooks,
            "run_id": run_id,
            "agent": agent,
        }

    def unwrap_agent(self, agent: Agent) -> None:
        """Restore an agent to its original unwrapped state.

        Args:
            agent: Previously wrapped AutoGen agent
        """
        agent_id = id(agent)
        if agent_id not in self._wrapped_agents:
            logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} is not wrapped")
            return

        wrapped_info = self._wrapped_agents.pop(agent_id)

        # Send final trace
        run_id = wrapped_info["run_id"]
        if run_id in self._runs:
            run = self._runs.pop(run_id)
            run.duration_ms = (time.time() - run.start_time) * 1000
            if not run.output_text:
                run.output_text = self._extract_final_output(run)
            self._send_trace(run)

        # Restore hooks
        self._restore_agent_hooks(agent, wrapped_info["original_hooks"])

    # =========================================================================
    # Model/Provider Detection
    # =========================================================================

    def _extract_model_from_agents(self, agents: tuple) -> Tuple[str, str]:
        """Extract model info from a list of agents."""
        for agent in agents:
            provider, model = self._extract_model_info(agent)
            if model != self.default_model:
                return provider, model
        return self.default_provider, self.default_model

    def _extract_model_info(self, agent: Any) -> Tuple[str, str]:
        """Extract provider and model from AutoGen agent configuration.

        Handles various AutoGen patterns:
        - AssistantAgent with llm_config
        - ConversableAgent with config_list
        - UserProxyAgent (usually no LLM)
        """
        provider = self.default_provider
        model = self.default_model

        # Try llm_config (most common)
        llm_config = getattr(agent, "llm_config", None)
        if llm_config and isinstance(llm_config, dict):
            # Check if LLM is disabled
            if llm_config.get("config_list") is False:
                return "none", "no-llm"

            # Direct model specification
            model_name = llm_config.get("model")

            # Check config_list (AutoGen pattern for multiple models)
            if not model_name:
                config_list = llm_config.get("config_list", [])
                if config_list and isinstance(config_list, list) and len(config_list) > 0:
                    first_config = config_list[0]
                    if isinstance(first_config, dict):
                        model_name = first_config.get("model")

                        # Check for api_type (Azure, etc.)
                        api_type = first_config.get("api_type", "")
                        if api_type == "azure":
                            provider = "azure"
                        elif "base_url" in first_config:
                            # Custom endpoint - try to infer
                            base_url = first_config.get("base_url", "").lower()
                            if "anthropic" in base_url:
                                provider = "anthropic"
                            elif "ollama" in base_url:
                                provider = "ollama"
                            elif "groq" in base_url:
                                provider = "groq"

            if model_name:
                model = model_name
                if provider == self.default_provider:
                    provider = self._infer_provider(model_name)

        # Fallback: check agent name for hints
        if model == self.default_model:
            agent_name = getattr(agent, "name", "")
            if agent_name:
                inferred_provider, inferred_model = self._infer_from_name(agent_name)
                if inferred_model != self.default_model:
                    provider, model = inferred_provider, inferred_model

        return provider, model

    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        if not model_name:
            return self.default_provider

        model_lower = model_name.lower()

        # OpenAI patterns
        if any(p in model_lower for p in ["gpt", "o1-", "o3-", "davinci", "curie", "text-embedding"]):
            return "openai"

        # Anthropic patterns
        if any(p in model_lower for p in ["claude", "anthropic"]):
            return "anthropic"

        # Google patterns
        if any(p in model_lower for p in ["gemini", "palm", "bard"]):
            return "google"

        # Azure (usually has deployment name)
        if "azure" in model_lower:
            return "azure"

        # Mistral
        if any(p in model_lower for p in ["mistral", "mixtral"]):
            return "mistral"

        # Meta/Llama
        if any(p in model_lower for p in ["llama", "meta"]):
            return "meta"

        # Cohere
        if "command" in model_lower or "cohere" in model_lower:
            return "cohere"

        # Groq
        if "groq" in model_lower:
            return "groq"

        return self.default_provider

    def _infer_from_name(self, agent_name: str) -> Tuple[str, str]:
        """Try to infer model from agent name (best effort)."""
        name_lower = agent_name.lower()

        if "gpt4" in name_lower or "gpt-4" in name_lower:
            return "openai", "gpt-4"
        elif "gpt3" in name_lower or "gpt-3" in name_lower:
            return "openai", "gpt-3.5-turbo"
        elif "claude" in name_lower:
            return "anthropic", "claude-3-sonnet"
        elif "gemini" in name_lower:
            return "google", "gemini-pro"
        elif "llama" in name_lower:
            return "meta", "llama-3"
        elif "mistral" in name_lower:
            return "mistral", "mistral-large"

        return self.default_provider, self.default_model

    # =========================================================================
    # Agent Hook Management
    # =========================================================================

    def _wrap_agent_hooks(self, agent: Agent, run_id: str) -> Dict[str, Any]:
        """Wrap agent's hooks to capture messages and function calls.

        Returns the original hooks for later restoration.
        """
        original_hooks = {}

        # Store and wrap receive hook
        if hasattr(agent, "_process_received_message"):
            original_hooks["_process_received_message"] = agent._process_received_message
            agent._process_received_message = self._create_message_hook(
                agent, run_id, original_hooks["_process_received_message"]
            )

        # Store and wrap function execution hooks (for function calling)
        if hasattr(agent, "_execute_function"):
            original_hooks["_execute_function"] = agent._execute_function
            agent._execute_function = self._create_function_hook(
                agent, run_id, original_hooks["_execute_function"]
            )

        # For ConversableAgent, also wrap generate_reply if available
        if hasattr(agent, "generate_reply"):
            original_hooks["generate_reply"] = agent.generate_reply
            agent.generate_reply = self._create_reply_hook(
                agent, run_id, original_hooks["generate_reply"]
            )

        return original_hooks

    def _restore_agent_hooks(self, agent: Agent, original_hooks: Dict[str, Any]) -> None:
        """Restore agent's original hooks."""
        for hook_name, original_func in original_hooks.items():
            if hasattr(agent, hook_name):
                setattr(agent, hook_name, original_func)

    def _create_message_hook(
        self,
        agent: Agent,
        run_id: str,
        original_func: Callable,
    ) -> Callable:
        """Create a wrapped message processing hook."""
        def wrapped_process_message(message: Any, sender: Any, *args, **kwargs):
            # Record the message
            self._on_message(run_id, agent, sender, message)
            # Call original
            return original_func(message, sender, *args, **kwargs)

        return wrapped_process_message

    def _create_function_hook(
        self,
        agent: Agent,
        run_id: str,
        original_func: Callable,
    ) -> Callable:
        """Create a wrapped function execution hook."""
        def wrapped_execute_function(func_call: Dict[str, Any], *args, **kwargs):
            # Record function start
            self._on_function_start(run_id, func_call)

            try:
                result = original_func(func_call, *args, **kwargs)
                self._on_function_end(run_id, func_call, result)
                return result
            except Exception as e:
                self._on_function_error(run_id, func_call, e)
                raise

        return wrapped_execute_function

    def _create_reply_hook(
        self,
        agent: Agent,
        run_id: str,
        original_func: Callable,
    ) -> Callable:
        """Create a wrapped reply generation hook."""
        def wrapped_generate_reply(messages: Any = None, sender: Any = None, *args, **kwargs):
            result = original_func(messages, sender, *args, **kwargs)

            # Update run with reply
            if run_id in self._runs:
                run = self._runs[run_id]
                if result:
                    run.output_text = str(result) if not isinstance(result, str) else result

            return result

        return wrapped_generate_reply

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_message(
        self,
        run_id: str,
        agent: Agent,
        sender: Any,
        message: Any,
    ) -> None:
        """Handle incoming message event."""
        if run_id not in self._runs:
            return

        run = self._runs[run_id]

        # Extract message content
        if isinstance(message, dict):
            content = message.get("content", str(message))
        else:
            content = str(message)

        # Record message
        msg_record = {
            "role": getattr(sender, "name", "unknown") if sender else "user",
            "content": content,
            "timestamp": time.time(),
        }
        run.messages.append(msg_record)

        # Set input text from first user message
        if not run.input_text and msg_record["role"] != getattr(agent, "name", ""):
            run.input_text = content

        logger.debug(f"CERT AutoGen: Message from {msg_record['role']}: {content[:50]}...")

    def _on_function_start(self, run_id: str, func_call: Dict[str, Any]) -> None:
        """Handle function call start event."""
        if run_id not in self._runs:
            return

        run = self._runs[run_id]

        func_name = func_call.get("name", "unknown")
        func_args = self._parse_arguments(func_call.get("arguments", {}))

        run.current_tool = {
            "name": func_name,
            "input": func_args,
            "start_time": time.time(),
        }

        logger.debug(f"CERT AutoGen: Function call started: {func_name}")

    def _on_function_end(
        self,
        run_id: str,
        func_call: Dict[str, Any],
        result: Any,
    ) -> None:
        """Handle function call end event."""
        if run_id not in self._runs:
            return

        run = self._runs[run_id]

        if run.current_tool:
            run.current_tool["output"] = self._serialize_result(result)
            run.current_tool["duration_ms"] = (
                time.time() - run.current_tool.get("start_time", time.time())
            ) * 1000
            del run.current_tool["start_time"]

            run.tool_calls.append(run.current_tool)
            run.current_tool = None

            logger.debug(f"CERT AutoGen: Function call completed: {func_call.get('name')}")

    def _on_function_error(
        self,
        run_id: str,
        func_call: Dict[str, Any],
        error: Exception,
    ) -> None:
        """Handle function call error event."""
        if run_id not in self._runs:
            return

        run = self._runs[run_id]

        if run.current_tool:
            run.current_tool["error"] = str(error)
            run.current_tool["duration_ms"] = (
                time.time() - run.current_tool.get("start_time", time.time())
            ) * 1000
            del run.current_tool["start_time"]

            run.tool_calls.append(run.current_tool)
            run.current_tool = None

            logger.warning(f"CERT AutoGen: Function call failed: {func_call.get('name')}: {error}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_arguments(self, arguments: Any) -> Dict[str, Any]:
        """Parse function arguments to dictionary format."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
        return {"raw": str(arguments)}

    def _serialize_result(self, result: Any) -> Any:
        """Serialize function result for storage."""
        if result is None:
            return None
        if isinstance(result, (str, int, float, bool)):
            return result
        if isinstance(result, (list, dict)):
            try:
                json.dumps(result)
                return result
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    def _extract_final_output(self, run: _ConversationRun) -> str:
        """Extract final output from conversation messages."""
        if not run.messages:
            return ""

        # Get last assistant/agent message
        for msg in reversed(run.messages):
            role = msg.get("role", "")
            if role not in ["user", "human"]:
                return msg.get("content", "")

        # Fallback to last message
        return run.messages[-1].get("content", "")

    def _send_trace(self, run: _ConversationRun) -> None:
        """Send trace to CERT."""
        # Warn if we couldn't detect model
        if run.provider == "unknown" or run.model == "unknown":
            logger.warning(
                f"CERT AutoGen: Could not detect provider/model for conversation {run.run_id}. "
                f"Using defaults: provider={run.provider}, model={run.model}. "
                "Consider setting llm_config on your agents."
            )

        # Build knowledge base from tool outputs + conversation history
        knowledge_base = None
        context_source: Optional[ContextSource] = None

        if self.auto_extract_knowledge:
            knowledge_base = run.build_knowledge_base()
            if knowledge_base:
                # Determine source
                has_tool_outputs = any(tc.get("output") for tc in run.tool_calls)
                context_source = "tools" if has_tool_outputs else "conversation"

        # Determine evaluation mode
        evaluation_mode: EvaluationMode = "grounded" if knowledge_base else "ungrounded"

        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text or "",
            output_text=run.output_text or "",
            duration_ms=run.duration_ms,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            evaluation_mode=evaluation_mode,
            knowledge_base=knowledge_base,
            context_source=context_source,
            tool_calls=run.tool_calls if run.tool_calls else None,
            goal_description=run.input_text,
            metadata={
                "autogen_run_id": run.run_id,
                "autogen_message_count": len(run.messages),
                "autogen_tool_call_count": len(run.tool_calls),
                "autogen_detected_provider": run.provider,
                "autogen_detected_model": run.model,
                **run.metadata,
            },
        )

        if self.auto_flush:
            self.cert_client.flush()

        logger.debug(
            f"CERT AutoGen: Traced conversation {run.run_id} "
            f"with {len(run.messages)} messages and {len(run.tool_calls)} tool calls"
        )

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "CERTAutoGenHandler":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all wrapped agents."""
        # Unwrap all agents
        for agent_id in list(self._wrapped_agents.keys()):
            wrapped_info = self._wrapped_agents[agent_id]
            self.unwrap_agent(wrapped_info["agent"])

        # Flush any remaining traces
        self.cert_client.flush()
