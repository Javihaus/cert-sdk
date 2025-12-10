"""
CERT SDK AutoGen Integration.

Provides handler for automatic tracing of AutoGen agents and conversations.
Captures tool calls, agent messages, and timing information.

Usage:
    from cert import CertClient
    from cert.integrations.autogen import CERTAutoGenHandler
    
    client = CertClient(api_key="cert_xxx")
    handler = CERTAutoGenHandler(client)
    
    # Wrap your agents
    assistant = handler.wrap_agent(AssistantAgent("assistant", llm_config=...))
    user_proxy = handler.wrap_agent(UserProxyAgent("user", ...))
    
    # Or trace manually
    handler.trace_conversation(
        initiator=user_proxy,
        recipient=assistant,
        message="What's the weather?"
    )

Requirements:
    pip install pyautogen
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union

logger = logging.getLogger(__name__)


# Check if autogen is available
try:
    from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    ConversableAgent = Any
    AssistantAgent = Any
    UserProxyAgent = Any


@dataclass
class _ConversationRun:
    """Internal tracking for an AutoGen conversation."""
    run_id: str
    input_text: str
    start_time: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    output_text: Optional[str] = None
    provider: str = "openai"
    model: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CERTAutoGenHandler:
    """
    AutoGen handler for CERT tracing.
    
    Wraps AutoGen agents to capture:
    - Conversation messages
    - Function/tool calls with outputs
    - Model information
    - Timing information
    
    Tool outputs are automatically extracted as context for
    agentic evaluation, enabling full CERT metrics (SGI, grounding, NLI).
    
    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.autogen import CERTAutoGenHandler
        >>> from autogen import AssistantAgent, UserProxyAgent
        >>> 
        >>> client = CertClient(api_key="cert_xxx")
        >>> handler = CERTAutoGenHandler(client)
        >>> 
        >>> # Create agents
        >>> assistant = AssistantAgent("assistant", llm_config=llm_config)
        >>> user = UserProxyAgent("user", code_execution_config=False)
        >>> 
        >>> # Trace a conversation
        >>> result = handler.trace_conversation(
        ...     initiator=user,
        ...     recipient=assistant,
        ...     message="What's 2 + 2?"
        ... )
    """
    
    def __init__(
        self,
        cert_client: Any,
        default_provider: str = "openai",
        default_model: str = "gpt-4",
        auto_flush: bool = True,
    ):
        """
        Initialize the AutoGen handler.
        
        Args:
            cert_client: CertClient instance for sending traces
            default_provider: Default provider name
            default_model: Default model name
            auto_flush: Flush traces automatically after each conversation
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is required for this integration. "
                "Install with: pip install pyautogen"
            )
        
        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        
        # Track conversations
        self._runs: Dict[str, _ConversationRun] = {}
        self._current_run_id: Optional[str] = None
    
    def trace_conversation(
        self,
        initiator: ConversableAgent,
        recipient: ConversableAgent,
        message: str,
        **kwargs: Any,
    ) -> Any:
        """
        Trace an AutoGen conversation.
        
        Wraps the conversation to capture all messages, tool calls,
        and the final output for CERT evaluation.
        
        Args:
            initiator: The agent initiating the conversation
            recipient: The agent receiving the message
            message: Initial message to send
            **kwargs: Additional arguments passed to initiate_chat
            
        Returns:
            Result from the conversation
        """
        import uuid
        
        run_id = str(uuid.uuid4())
        run = _ConversationRun(
            run_id=run_id,
            input_text=message,
            start_time=time.time(),
            provider=self.default_provider,
            model=self._extract_model(recipient),
        )
        
        self._runs[run_id] = run
        self._current_run_id = run_id
        
        # Store original hooks
        original_receive = recipient.receive
        original_execute = getattr(recipient, "_execute_function", None)
        
        # Wrap receive to capture messages
        def traced_receive(message, sender, *args, **kwargs):
            self._on_message(run_id, message, sender)
            return original_receive(message, sender, *args, **kwargs)
        
        # Wrap function execution to capture tool calls
        if original_execute:
            def traced_execute(func_call, *args, **kwargs):
                self._on_function_start(run_id, func_call)
                try:
                    result = original_execute(func_call, *args, **kwargs)
                    self._on_function_end(run_id, func_call, result)
                    return result
                except Exception as e:
                    self._on_function_error(run_id, func_call, e)
                    raise
            recipient._execute_function = traced_execute
        
        recipient.receive = traced_receive
        
        try:
            # Run the conversation
            result = initiator.initiate_chat(recipient, message=message, **kwargs)
            
            # Capture final output
            run.output_text = self._extract_final_output(recipient)
            
            # Send trace
            self._send_trace(run)
            
            return result
            
        finally:
            # Restore original hooks
            recipient.receive = original_receive
            if original_execute:
                recipient._execute_function = original_execute
            
            self._runs.pop(run_id, None)
            self._current_run_id = None
    
    def wrap_agent(
        self,
        agent: ConversableAgent,
    ) -> ConversableAgent:
        """
        Wrap an AutoGen agent for automatic tracing.
        
        All messages and function calls through this agent will be traced.
        
        Args:
            agent: The AutoGen agent to wrap
            
        Returns:
            The wrapped agent (same instance, modified in place)
        """
        handler = self
        original_receive = agent.receive
        original_execute = getattr(agent, "_execute_function", None)
        
        def traced_receive(message, sender, *args, **kwargs):
            if handler._current_run_id:
                handler._on_message(handler._current_run_id, message, sender)
            return original_receive(message, sender, *args, **kwargs)
        
        if original_execute:
            def traced_execute(func_call, *args, **kwargs):
                if handler._current_run_id:
                    handler._on_function_start(handler._current_run_id, func_call)
                try:
                    result = original_execute(func_call, *args, **kwargs)
                    if handler._current_run_id:
                        handler._on_function_end(handler._current_run_id, func_call, result)
                    return result
                except Exception as e:
                    if handler._current_run_id:
                        handler._on_function_error(handler._current_run_id, func_call, e)
                    raise
            agent._execute_function = traced_execute
        
        agent.receive = traced_receive
        return agent
    
    # === Internal Hooks ===
    
    def _on_message(self, run_id: str, message: Any, sender: Any) -> None:
        """Handle message event."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        msg_content = message if isinstance(message, str) else message.get("content", str(message))
        sender_name = getattr(sender, "name", "unknown")
        
        run.messages.append({
            "role": sender_name,
            "content": msg_content,
        })
    
    def _on_function_start(self, run_id: str, func_call: Dict[str, Any]) -> None:
        """Handle function call start."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        # func_call is typically {"name": ..., "arguments": ...}
        logger.debug(f"CERT AutoGen: Function {func_call.get('name')} started")
    
    def _on_function_end(
        self,
        run_id: str,
        func_call: Dict[str, Any],
        result: Any,
    ) -> None:
        """Handle function call end."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        tool_call = {
            "name": func_call.get("name", "unknown"),
            "input": self._parse_arguments(func_call.get("arguments", {})),
            "output": result if isinstance(result, (dict, list, str, int, float)) else str(result),
        }
        run.tool_calls.append(tool_call)
        logger.debug(f"CERT AutoGen: Function {tool_call['name']} completed")
    
    def _on_function_error(
        self,
        run_id: str,
        func_call: Dict[str, Any],
        error: Exception,
    ) -> None:
        """Handle function call error."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        tool_call = {
            "name": func_call.get("name", "unknown"),
            "input": self._parse_arguments(func_call.get("arguments", {})),
            "error": str(error),
        }
        run.tool_calls.append(tool_call)
        logger.debug(f"CERT AutoGen: Function {tool_call['name']} errored")
    
    # === Helper Methods ===
    
    def _extract_model(self, agent: ConversableAgent) -> str:
        """Extract model name from agent config."""
        try:
            llm_config = getattr(agent, "llm_config", {}) or {}
            config_list = llm_config.get("config_list", [])
            if config_list:
                return config_list[0].get("model", self.default_model)
            return llm_config.get("model", self.default_model)
        except Exception:
            return self.default_model
    
    def _extract_final_output(self, agent: ConversableAgent) -> str:
        """Extract final output from agent chat history."""
        try:
            chat_messages = getattr(agent, "chat_messages", {})
            if chat_messages:
                # Get the last message from the last conversation
                for messages in chat_messages.values():
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            return last_msg.get("content", str(last_msg))
                        return str(last_msg)
            return ""
        except Exception:
            return ""
    
    def _parse_arguments(self, args: Any) -> Dict[str, Any]:
        """Parse function arguments to dict."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {"input": args}
        return {"input": str(args)}
    
    def _send_trace(self, run: _ConversationRun) -> None:
        """Send trace to CERT."""
        duration_ms = (time.time() - run.start_time) * 1000
        
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=run.output_text or "",
            duration_ms=duration_ms,
            eval_mode="agentic" if run.tool_calls else "generation",
            tool_calls=run.tool_calls if run.tool_calls else None,
            goal_description=run.input_text,
            metadata={
                "autogen_run_id": run.run_id,
                "message_count": len(run.messages),
                **run.metadata,
            },
        )
        
        if self.auto_flush:
            self.cert_client.flush()
        
        logger.debug(
            f"CERT AutoGen: Traced conversation {run.run_id} "
            f"with {len(run.tool_calls)} tool calls"
        )
    
    # === Context Manager ===
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cert_client.flush()
        return False
