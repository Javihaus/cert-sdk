"""
CERT CrewAI Integration - Automatic tracing for CrewAI crews and tasks.

Provides seamless integration with CrewAI framework for:
- Automatic capture of crew executions
- Task-level tracing with inputs and outputs
- Tool call tracking with inputs and outputs
- Model and provider detection from agent/crew configurations
- Support for both sequential and hierarchical processes

Usage:
    from cert import CertClient
    from cert.integrations.crewai import CERTCrewAIHandler

    client = CertClient(api_key="...", project="my-project")
    handler = CERTCrewAIHandler(client)

    # Option 1: Trace entire crew execution
    with handler.trace_crew(crew, inputs={"topic": "AI"}) as run_id:
        result = crew.kickoff(inputs={"topic": "AI"})

    # Option 2: Trace individual tasks
    with handler.trace_task(task, agent, inputs={"query": "..."}) as run_id:
        result = task.execute(agent)

    # Option 3: Wrap tools for automatic tracing
    wrapped_tool = handler.wrap_tool(my_tool)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from cert.types import EvaluationMode, ContextSource

logger = logging.getLogger(__name__)

# Check if CrewAI is available
try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Type stubs for when CrewAI is not installed
    Agent = Any
    Crew = Any
    Process = Any
    Task = Any
    BaseTool = Any

if TYPE_CHECKING:
    from cert.client import CertClient


@dataclass
class _CrewRun:
    """Internal tracking for a single crew/task run."""

    run_id: str
    input_text: str
    start_time: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    task_outputs: List[Dict[str, Any]] = field(default_factory=list)
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
        Build knowledge base from tool outputs and task results.

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

        # Add completed task outputs as prior knowledge
        for task in self.task_outputs:
            if task.get("output"):
                task_desc = task.get("description", "task")[:50]
                parts.append(f"[task:{task_desc}]: {task['output']}")

        return "\n\n".join(parts) if parts else None


class CERTCrewAIHandler:
    """CERT callback handler for CrewAI crews and tasks.

    Provides automatic tracing of CrewAI executions, including:
    - Crew kickoff with all task outputs
    - Individual task execution
    - Tool calls with inputs and outputs
    - Model information and timing metrics

    Automatically extracts knowledge base from tool outputs and
    task results for grounded evaluation.

    Args:
        cert_client: Initialized CertClient instance
        default_provider: Default provider if detection fails (default: "openai")
        default_model: Default model if detection fails (default: "gpt-4")
        auto_flush: Whether to flush after each trace (default: True)
        auto_extract_knowledge: Automatically extract knowledge from tool outputs
                                and task results (default: True)

    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.crewai import CERTCrewAIHandler
        >>>
        >>> client = CertClient(api_key="...", project="crewai-demo")
        >>> handler = CERTCrewAIHandler(client)
        >>>
        >>> with handler.trace_crew(my_crew, inputs={"topic": "AI"}):
        ...     result = my_crew.kickoff(inputs={"topic": "AI"})
    """

    def __init__(
        self,
        cert_client: "CertClient",
        default_provider: str = "openai",
        default_model: str = "gpt-4",
        auto_flush: bool = True,
        auto_extract_knowledge: bool = True,
    ) -> None:
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Install it with: pip install crewai"
            )

        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        self.auto_extract_knowledge = auto_extract_knowledge

        # Track active runs
        self._runs: Dict[str, _CrewRun] = {}

        # Track wrapped tools for cleanup
        self._wrapped_tools: Dict[int, Dict[str, Any]] = {}

        # Track original tool methods on agents
        self._original_tool_methods: Dict[int, Dict[int, bool]] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    @contextmanager
    def trace_crew(
        self,
        crew: "Crew",
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager to trace a crew execution.

        Wraps all agent tools, traces the crew kickoff, and captures
        all task outputs.

        Args:
            crew: CrewAI Crew instance
            inputs: Inputs to pass to crew.kickoff()
            metadata: Additional metadata to include in the trace

        Yields:
            The run ID for this trace

        Example:
            >>> with handler.trace_crew(my_crew, inputs={"topic": "AI"}) as run_id:
            ...     result = my_crew.kickoff(inputs={"topic": "AI"})
            ...     print(f"Traced as: {run_id}")
        """
        run_id = str(uuid.uuid4())

        # Extract model info from crew
        provider, model = self._extract_crew_model(crew)

        # Format input text
        input_text = json.dumps(inputs) if inputs else ""

        run = _CrewRun(
            run_id=run_id,
            input_text=input_text,
            start_time=time.time(),
            provider=provider,
            model=model,
            metadata={
                "crew_name": getattr(crew, "name", None) or "unnamed_crew",
                "process_type": self._get_process_type(crew),
                "agent_count": len(getattr(crew, "agents", [])),
                "task_count": len(getattr(crew, "tasks", [])),
                **(metadata or {}),
            },
        )
        self._runs[run_id] = run

        # Wrap all agent tools
        self._wrap_crew_tools(crew, run_id)

        try:
            yield run_id
        finally:
            # Calculate duration
            run.duration_ms = (time.time() - run.start_time) * 1000

            # Extract task outputs
            run.task_outputs = self._extract_task_outputs(crew)

            # Extract final output if not set
            if not run.output_text and run.task_outputs:
                last_output = run.task_outputs[-1]
                run.output_text = last_output.get("output", "")

            # Send trace
            self._send_trace(run)

            # Restore original tools
            self._restore_crew_tools(crew)

            # Cleanup
            del self._runs[run_id]

    @contextmanager
    def trace_task(
        self,
        task: "Task",
        agent: Optional["Agent"] = None,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager to trace a single task execution.

        Args:
            task: CrewAI Task instance
            agent: Agent executing the task (optional, will use task.agent)
            inputs: Task inputs
            metadata: Additional metadata to include in the trace

        Yields:
            The run ID for this trace

        Example:
            >>> with handler.trace_task(my_task, my_agent, inputs={"query": "..."}) as run_id:
            ...     result = my_task.execute(my_agent)
        """
        run_id = str(uuid.uuid4())

        # Use task's agent if not provided
        task_agent = agent or getattr(task, "agent", None)

        # Extract model info
        provider, model = self._extract_model_info(task_agent, task)

        # Format input text
        input_text = (
            json.dumps(inputs)
            if inputs
            else getattr(task, "description", "") or str(task)
        )

        run = _CrewRun(
            run_id=run_id,
            input_text=input_text,
            start_time=time.time(),
            provider=provider,
            model=model,
            metadata={
                "task_description": getattr(task, "description", "")[:200],
                "agent_name": getattr(task_agent, "role", None) if task_agent else None,
                **(metadata or {}),
            },
        )
        self._runs[run_id] = run

        # Wrap agent tools if available
        if task_agent:
            self._wrap_agent_tools(task_agent, run_id)

        try:
            yield run_id
        finally:
            # Calculate duration
            run.duration_ms = (time.time() - run.start_time) * 1000

            # Extract output from task if available
            if not run.output_text:
                task_output = getattr(task, "output", None)
                if task_output:
                    run.output_text = (
                        str(task_output.raw) if hasattr(task_output, "raw") else str(task_output)
                    )

            # Send trace
            self._send_trace(run)

            # Restore tools
            if task_agent:
                self._restore_agent_tools(task_agent)

            # Cleanup
            del self._runs[run_id]

    def wrap_tool(self, tool: "BaseTool", run_id: Optional[str] = None) -> "BaseTool":
        """Wrap a CrewAI tool for automatic tracing.

        Args:
            tool: CrewAI BaseTool instance
            run_id: Optional run ID to associate with (uses default if not provided)

        Returns:
            The wrapped tool (same instance, modified in place)

        Example:
            >>> wrapped_tool = handler.wrap_tool(my_search_tool)
            >>> # Tool calls will now be traced
        """
        tool_id = id(tool)
        if tool_id in self._wrapped_tools:
            return tool

        # Store original _run method
        original_run = getattr(tool, "_run", None)
        if original_run is None:
            logger.warning(f"Tool {getattr(tool, 'name', 'unknown')} has no _run method")
            return tool

        self._wrapped_tools[tool_id] = {
            "original_run": original_run,
            "tool": tool,
            "run_id": run_id,
        }

        # Create wrapped method
        @wraps(original_run)
        def wrapped_run(*args, **kwargs):
            active_run_id = run_id or self._get_active_run_id()
            return self._execute_tool_with_tracing(
                tool, original_run, active_run_id, *args, **kwargs
            )

        tool._run = wrapped_run
        return tool

    def unwrap_tool(self, tool: "BaseTool") -> "BaseTool":
        """Restore a tool to its original unwrapped state.

        Args:
            tool: Previously wrapped CrewAI BaseTool instance

        Returns:
            The unwrapped tool
        """
        tool_id = id(tool)
        if tool_id not in self._wrapped_tools:
            return tool

        wrapped_info = self._wrapped_tools.pop(tool_id)
        tool._run = wrapped_info["original_run"]
        return tool

    # =========================================================================
    # Model/Provider Detection
    # =========================================================================

    def _extract_crew_model(self, crew: "Crew") -> Tuple[str, str]:
        """Extract default model from crew configuration."""
        # Check crew's manager_llm (used for hierarchical process)
        manager_llm = getattr(crew, "manager_llm", None)
        if manager_llm:
            provider, model = self._parse_llm_config(manager_llm)
            if model != self.default_model:
                return provider, model

        # Check first agent's LLM as fallback
        agents = getattr(crew, "agents", [])
        if agents and len(agents) > 0:
            return self._extract_model_info(agent=agents[0])

        return self.default_provider, self.default_model

    def _extract_model_info(
        self,
        agent: Optional["Agent"] = None,
        task: Optional["Task"] = None,
    ) -> Tuple[str, str]:
        """Extract provider and model from CrewAI agent/task configuration.

        CrewAI hierarchy:
        1. Task can override agent's LLM
        2. Agent has llm attribute (can be string or LLM object)
        3. Crew can have default LLM
        """
        provider = self.default_provider
        model = self.default_model

        # Check task-level LLM override first
        if task:
            task_llm = getattr(task, "llm", None)
            if task_llm:
                p, m = self._parse_llm_config(task_llm)
                if m != self.default_model:
                    return p, m

        # Check agent's LLM
        if agent:
            agent_llm = getattr(agent, "llm", None)
            if agent_llm:
                return self._parse_llm_config(agent_llm)

            # Some CrewAI versions use llm_config
            llm_config = getattr(agent, "llm_config", None)
            if llm_config:
                return self._parse_llm_config(llm_config)

        return provider, model

    def _parse_llm_config(self, llm: Any) -> Tuple[str, str]:
        """Parse LLM configuration from various formats.

        CrewAI supports:
        - String model names: "gpt-4", "claude-3-opus"
        - LangChain LLM objects
        - LiteLLM model strings: "anthropic/claude-3-opus"
        - Custom LLM objects with model_name attribute
        """
        provider = self.default_provider
        model = self.default_model

        if llm is None:
            return provider, model

        if isinstance(llm, str):
            # Direct model string (common in CrewAI)
            model = llm
            provider = self._infer_provider(llm)
        elif hasattr(llm, "model_name"):
            # LangChain-style LLM object
            model = str(llm.model_name)
            provider = self._infer_provider(model)
        elif hasattr(llm, "model"):
            # Some LLM wrappers use .model
            model = str(llm.model)
            provider = self._infer_provider(model)
        elif hasattr(llm, "model_id"):
            # Hugging Face style
            model = str(llm.model_id)
            provider = self._infer_provider(model)
        elif isinstance(llm, dict):
            # Dictionary config
            model = llm.get("model_name") or llm.get("model") or self.default_model
            provider = llm.get("provider") or self._infer_provider(str(model))

        return provider, model

    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name.

        CrewAI often uses LiteLLM format: "provider/model"
        e.g., "anthropic/claude-3-opus", "openai/gpt-4"
        """
        if not model_name:
            return self.default_provider

        model_str = str(model_name)
        model_lower = model_str.lower()

        # LiteLLM format: "provider/model"
        if "/" in model_str:
            provider_part = model_str.split("/")[0].lower()
            known_providers = [
                "openai", "anthropic", "google", "azure", "bedrock",
                "ollama", "mistral", "cohere", "groq", "together",
                "huggingface", "replicate", "anyscale", "deepinfra",
            ]
            if provider_part in known_providers:
                return provider_part

        # OpenAI patterns
        if any(p in model_lower for p in ["gpt", "o1-", "o3-", "davinci", "curie", "text-embedding"]):
            return "openai"

        # Anthropic patterns
        if any(p in model_lower for p in ["claude", "anthropic"]):
            return "anthropic"

        # Google patterns
        if any(p in model_lower for p in ["gemini", "palm", "bard"]):
            return "google"

        # Groq (fast inference, often runs Llama)
        if "groq" in model_lower:
            return "groq"

        # Mistral
        if any(p in model_lower for p in ["mistral", "mixtral"]):
            return "mistral"

        # Meta/Llama
        if any(p in model_lower for p in ["llama", "meta"]):
            return "meta"

        # Cohere
        if "command" in model_lower or "cohere" in model_lower:
            return "cohere"

        # Ollama (local)
        if "ollama" in model_lower:
            return "ollama"

        return self.default_provider

    def _get_process_type(self, crew: "Crew") -> str:
        """Get the process type of a crew."""
        process = getattr(crew, "process", None)
        if process is None:
            return "sequential"

        if hasattr(process, "value"):
            return str(process.value)
        elif hasattr(process, "name"):
            return str(process.name).lower()
        else:
            return str(process).lower()

    # =========================================================================
    # Tool Wrapping
    # =========================================================================

    def _wrap_crew_tools(self, crew: "Crew", run_id: str) -> None:
        """Wrap all tools from all agents in a crew."""
        agents = getattr(crew, "agents", [])
        for agent in agents:
            self._wrap_agent_tools(agent, run_id)

    def _wrap_agent_tools(self, agent: "Agent", run_id: str) -> None:
        """Wrap all tools on an agent."""
        tools = getattr(agent, "tools", [])
        if not tools:
            return

        agent_id = id(agent)
        self._original_tool_methods[agent_id] = {}

        for tool in tools:
            if hasattr(tool, "_run"):
                tool_id = id(tool)
                if tool_id not in self._wrapped_tools:
                    self.wrap_tool(tool, run_id)
                    self._original_tool_methods[agent_id][tool_id] = True

    def _restore_crew_tools(self, crew: "Crew") -> None:
        """Restore all tools from all agents in a crew."""
        agents = getattr(crew, "agents", [])
        for agent in agents:
            self._restore_agent_tools(agent)

    def _restore_agent_tools(self, agent: "Agent") -> None:
        """Restore all tools on an agent."""
        agent_id = id(agent)
        if agent_id not in self._original_tool_methods:
            return

        tools = getattr(agent, "tools", [])
        for tool in tools:
            tool_id = id(tool)
            if tool_id in self._original_tool_methods[agent_id]:
                self.unwrap_tool(tool)

        del self._original_tool_methods[agent_id]

    def _execute_tool_with_tracing(
        self,
        tool: "BaseTool",
        original_run: Callable,
        run_id: Optional[str],
        *args,
        **kwargs,
    ) -> Any:
        """Execute a tool with tracing."""
        tool_name = getattr(tool, "name", "unknown_tool")

        # Record start
        self._on_tool_start(run_id, tool_name, args, kwargs)

        try:
            result = original_run(*args, **kwargs)
            self._on_tool_end(run_id, tool_name, result)
            return result
        except Exception as e:
            self._on_tool_error(run_id, tool_name, e)
            raise

    def _get_active_run_id(self) -> Optional[str]:
        """Get the currently active run ID."""
        if self._runs:
            return list(self._runs.keys())[-1]
        return None

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_tool_start(
        self,
        run_id: Optional[str],
        tool_name: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Handle tool call start event."""
        if not run_id or run_id not in self._runs:
            return

        run = self._runs[run_id]

        # Combine args and kwargs into input
        tool_input = {}
        if args:
            if len(args) == 1:
                tool_input = {"input": args[0]}
            else:
                tool_input = {"args": list(args)}
        tool_input.update(kwargs)

        run.current_tool = {
            "name": tool_name,
            "input": tool_input,
            "start_time": time.time(),
        }

        logger.debug(f"CERT CrewAI: Tool call started: {tool_name}")

    def _on_tool_end(
        self,
        run_id: Optional[str],
        tool_name: str,
        result: Any,
    ) -> None:
        """Handle tool call end event."""
        if not run_id or run_id not in self._runs:
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

            logger.debug(f"CERT CrewAI: Tool call completed: {tool_name}")

    def _on_tool_error(
        self,
        run_id: Optional[str],
        tool_name: str,
        error: Exception,
    ) -> None:
        """Handle tool call error event."""
        if not run_id or run_id not in self._runs:
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

            logger.warning(f"CERT CrewAI: Tool call failed: {tool_name}: {error}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _serialize_result(self, result: Any) -> Any:
        """Serialize tool result for storage."""
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

    def _extract_task_outputs(self, crew: "Crew") -> List[Dict[str, Any]]:
        """Extract outputs from all tasks in a crew."""
        outputs = []
        tasks = getattr(crew, "tasks", [])

        for i, task in enumerate(tasks):
            task_output = getattr(task, "output", None)
            agent = getattr(task, "agent", None)

            output_dict = {
                "task_index": i,
                "description": getattr(task, "description", "")[:200],
                "agent_role": getattr(agent, "role", None) if agent else None,
            }

            if task_output:
                if hasattr(task_output, "raw"):
                    output_dict["output"] = str(task_output.raw)
                elif hasattr(task_output, "result"):
                    output_dict["output"] = str(task_output.result)
                else:
                    output_dict["output"] = str(task_output)

                # Extract model used for this task if available
                if agent:
                    p, m = self._extract_model_info(agent, task)
                    output_dict["provider"] = p
                    output_dict["model"] = m

            outputs.append(output_dict)

        return outputs

    def _send_trace(self, run: _CrewRun) -> None:
        """Send trace to CERT."""
        # Warn if we couldn't detect model
        if run.provider == "unknown" or run.model == "unknown":
            logger.warning(
                f"CERT CrewAI: Could not detect provider/model for run {run.run_id}. "
                f"Using defaults: provider={run.provider}, model={run.model}. "
                "Consider setting the llm attribute on your agents."
            )

        # Build knowledge base from tool outputs and task results
        knowledge_base: Optional[str] = None
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
                "crewai_run_id": run.run_id,
                "crewai_task_count": len(run.task_outputs),
                "crewai_tool_call_count": len(run.tool_calls),
                "crewai_detected_provider": run.provider,
                "crewai_detected_model": run.model,
                "crewai_task_outputs": run.task_outputs,
                **run.metadata,
            },
        )

        if self.auto_flush:
            self.cert_client.flush()

        logger.debug(
            f"CERT CrewAI: Traced run {run.run_id} "
            f"with {len(run.task_outputs)} tasks and {len(run.tool_calls)} tool calls"
        )

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "CERTCrewAIHandler":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all wrapped tools."""
        # Unwrap all tools
        for tool_id in list(self._wrapped_tools.keys()):
            wrapped_info = self._wrapped_tools[tool_id]
            self.unwrap_tool(wrapped_info["tool"])

        # Flush any remaining traces
        self.cert_client.flush()
