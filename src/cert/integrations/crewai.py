"""
CERT SDK CrewAI Integration.

Provides handler for automatic tracing of CrewAI agents and crews.
Captures tool calls, agent outputs, and timing information.

Usage:
    from cert import CertClient
    from cert.integrations.crewai import CERTCrewAIHandler
    
    client = CertClient(api_key="cert_xxx")
    handler = CERTCrewAIHandler(client)
    
    # Trace crew execution
    result = handler.trace_crew(
        crew=my_crew,
        inputs={"topic": "AI safety"}
    )
    
    # Or trace individual tasks
    handler.trace_task(
        agent=researcher,
        task=research_task,
        context="background info..."
    )

Requirements:
    pip install crewai crewai-tools
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import uuid as uuid_module

logger = logging.getLogger(__name__)


# Check if crewai is available
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = Any
    Task = Any
    Crew = Any
    Process = Any
    BaseTool = Any


@dataclass
class _CrewRun:
    """Internal tracking for a CrewAI crew execution."""
    run_id: str
    input_text: str
    start_time: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    task_outputs: List[Dict[str, Any]] = field(default_factory=list)
    output_text: Optional[str] = None
    provider: str = "openai"
    model: str = "gpt-4"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CERTCrewAIHandler:
    """
    CrewAI handler for CERT tracing.
    
    Captures:
    - Crew inputs and final output
    - Individual task outputs
    - Tool invocations with results
    - Timing information
    
    Tool outputs are automatically extracted as context for
    agentic evaluation, enabling full CERT metrics (SGI, grounding, NLI).
    
    Example:
        >>> from cert import CertClient
        >>> from cert.integrations.crewai import CERTCrewAIHandler
        >>> from crewai import Agent, Task, Crew
        >>> 
        >>> client = CertClient(api_key="cert_xxx")
        >>> handler = CERTCrewAIHandler(client)
        >>> 
        >>> # Create crew
        >>> researcher = Agent(role="Researcher", ...)
        >>> task = Task(description="Research AI safety", agent=researcher)
        >>> crew = Crew(agents=[researcher], tasks=[task])
        >>> 
        >>> # Trace crew execution
        >>> result = handler.trace_crew(
        ...     crew=crew,
        ...     inputs={"topic": "AI safety"}
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
        Initialize the CrewAI handler.
        
        Args:
            cert_client: CertClient instance for sending traces
            default_provider: Default provider name
            default_model: Default model name
            auto_flush: Flush traces automatically after each crew run
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is required for this integration. "
                "Install with: pip install crewai crewai-tools"
            )
        
        self.cert_client = cert_client
        self.default_provider = default_provider
        self.default_model = default_model
        self.auto_flush = auto_flush
        
        # Track runs
        self._runs: Dict[str, _CrewRun] = {}
    
    def trace_crew(
        self,
        crew: Crew,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Trace a CrewAI crew execution.
        
        Wraps the crew kickoff to capture all task outputs, tool calls,
        and the final result for CERT evaluation.
        
        Args:
            crew: The CrewAI crew to execute
            inputs: Input dict for the crew
            **kwargs: Additional arguments passed to kickoff
            
        Returns:
            Result from crew.kickoff()
        """
        run_id = str(uuid_module.uuid4())
        
        # Build input text from inputs dict
        input_text = json.dumps(inputs or {}, ensure_ascii=False)
        
        run = _CrewRun(
            run_id=run_id,
            input_text=input_text,
            start_time=time.time(),
            provider=self.default_provider,
            model=self._extract_model(crew),
        )
        
        self._runs[run_id] = run
        
        # Wrap tools to capture calls
        original_tools = self._wrap_crew_tools(crew, run_id)
        
        try:
            # Run the crew
            result = crew.kickoff(inputs=inputs, **kwargs)
            
            # Capture output
            if hasattr(result, "raw"):
                run.output_text = str(result.raw)
            else:
                run.output_text = str(result)
            
            # Capture task outputs
            self._capture_task_outputs(crew, run)
            
            # Send trace
            self._send_trace(run)
            
            return result
            
        finally:
            # Restore original tools
            self._restore_crew_tools(crew, original_tools)
            self._runs.pop(run_id, None)
    
    def trace_task(
        self,
        agent: Agent,
        task: Task,
        context: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Trace a single CrewAI task execution.
        
        Args:
            agent: The agent executing the task
            task: The task to execute
            context: Optional context/background information
            provider: LLM provider name
            model: Model name
            
        Returns:
            Task execution result
        """
        run_id = str(uuid_module.uuid4())
        
        run = _CrewRun(
            run_id=run_id,
            input_text=task.description,
            start_time=time.time(),
            provider=provider or self.default_provider,
            model=model or self._extract_agent_model(agent),
        )
        
        self._runs[run_id] = run
        
        # Wrap agent tools
        original_tools = self._wrap_agent_tools(agent, run_id)
        
        try:
            # Execute task
            result = agent.execute_task(task, context=context)
            
            run.output_text = str(result) if result else ""
            
            # Send trace
            self.cert_client.trace(
                provider=run.provider,
                model=run.model,
                input_text=run.input_text,
                output_text=run.output_text,
                duration_ms=(time.time() - run.start_time) * 1000,
                eval_mode="agentic" if run.tool_calls else "generation",
                context=context,
                tool_calls=run.tool_calls if run.tool_calls else None,
                goal_description=task.description,
                metadata={
                    "crewai_run_id": run.run_id,
                    "agent_role": agent.role,
                    **run.metadata,
                },
            )
            
            if self.auto_flush:
                self.cert_client.flush()
            
            return result
            
        finally:
            self._restore_agent_tools(agent, original_tools)
            self._runs.pop(run_id, None)
    
    def wrap_tool(
        self,
        tool: BaseTool,
        run_id: Optional[str] = None,
    ) -> BaseTool:
        """
        Wrap a CrewAI tool to capture its calls.
        
        Args:
            tool: The tool to wrap
            run_id: Optional run ID to associate calls with
            
        Returns:
            Wrapped tool
        """
        handler = self
        original_run = tool._run if hasattr(tool, "_run") else None
        
        if original_run is None:
            return tool
        
        def traced_run(*args, **kwargs):
            current_run_id = run_id or list(handler._runs.keys())[-1] if handler._runs else None
            
            if current_run_id:
                handler._on_tool_start(current_run_id, tool.name, args, kwargs)
            
            try:
                result = original_run(*args, **kwargs)
                
                if current_run_id:
                    handler._on_tool_end(current_run_id, tool.name, result)
                
                return result
            except Exception as e:
                if current_run_id:
                    handler._on_tool_error(current_run_id, tool.name, e)
                raise
        
        tool._run = traced_run
        return tool
    
    # === Internal Methods ===
    
    def _wrap_crew_tools(
        self,
        crew: Crew,
        run_id: str,
    ) -> Dict[str, Any]:
        """Wrap all tools in a crew and return original functions."""
        originals = {}
        
        for agent in crew.agents:
            agent_originals = self._wrap_agent_tools(agent, run_id)
            originals[agent.role] = agent_originals
        
        return originals
    
    def _restore_crew_tools(
        self,
        crew: Crew,
        originals: Dict[str, Any],
    ) -> None:
        """Restore original tool functions."""
        for agent in crew.agents:
            if agent.role in originals:
                self._restore_agent_tools(agent, originals[agent.role])
    
    def _wrap_agent_tools(
        self,
        agent: Agent,
        run_id: str,
    ) -> Dict[str, Callable]:
        """Wrap all tools for an agent."""
        originals = {}
        
        tools = getattr(agent, "tools", []) or []
        for tool in tools:
            if hasattr(tool, "_run"):
                originals[getattr(tool, "name", str(tool))] = tool._run
                self.wrap_tool(tool, run_id)
        
        return originals
    
    def _restore_agent_tools(
        self,
        agent: Agent,
        originals: Dict[str, Callable],
    ) -> None:
        """Restore original tool functions for an agent."""
        tools = getattr(agent, "tools", []) or []
        for tool in tools:
            name = getattr(tool, "name", str(tool))
            if name in originals:
                tool._run = originals[name]
    
    def _on_tool_start(
        self,
        run_id: str,
        tool_name: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Handle tool start."""
        logger.debug(f"CERT CrewAI: Tool {tool_name} started")
    
    def _on_tool_end(
        self,
        run_id: str,
        tool_name: str,
        result: Any,
    ) -> None:
        """Handle tool end."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        tool_call = {
            "name": tool_name,
            "input": {},  # CrewAI doesn't easily expose inputs
            "output": result if isinstance(result, (dict, list, str, int, float)) else str(result),
        }
        run.tool_calls.append(tool_call)
        logger.debug(f"CERT CrewAI: Tool {tool_name} completed")
    
    def _on_tool_error(
        self,
        run_id: str,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Handle tool error."""
        run = self._runs.get(run_id)
        if run is None:
            return
        
        tool_call = {
            "name": tool_name,
            "input": {},
            "error": str(error),
        }
        run.tool_calls.append(tool_call)
        logger.debug(f"CERT CrewAI: Tool {tool_name} errored")
    
    def _capture_task_outputs(self, crew: Crew, run: _CrewRun) -> None:
        """Capture task outputs from crew."""
        try:
            for task in crew.tasks:
                if hasattr(task, "output") and task.output:
                    run.task_outputs.append({
                        "description": task.description[:100],
                        "output": str(task.output.raw) if hasattr(task.output, "raw") else str(task.output),
                    })
        except Exception as e:
            logger.debug(f"CERT CrewAI: Failed to capture task outputs: {e}")
    
    def _extract_model(self, crew: Crew) -> str:
        """Extract model from crew."""
        try:
            for agent in crew.agents:
                model = self._extract_agent_model(agent)
                if model != self.default_model:
                    return model
            return self.default_model
        except Exception:
            return self.default_model
    
    def _extract_agent_model(self, agent: Agent) -> str:
        """Extract model from agent."""
        try:
            llm = getattr(agent, "llm", None)
            if llm:
                return getattr(llm, "model_name", getattr(llm, "model", self.default_model))
            return self.default_model
        except Exception:
            return self.default_model
    
    def _send_trace(self, run: _CrewRun) -> None:
        """Send trace to CERT."""
        duration_ms = (time.time() - run.start_time) * 1000
        
        # Combine task outputs with tool calls for richer context
        all_tool_calls = list(run.tool_calls)
        for task_output in run.task_outputs:
            all_tool_calls.append({
                "name": f"task_{task_output['description'][:30]}",
                "output": task_output["output"],
            })
        
        self.cert_client.trace(
            provider=run.provider,
            model=run.model,
            input_text=run.input_text,
            output_text=run.output_text or "",
            duration_ms=duration_ms,
            eval_mode="agentic",
            tool_calls=all_tool_calls if all_tool_calls else None,
            goal_description=run.input_text,
            metadata={
                "crewai_run_id": run.run_id,
                "task_count": len(run.task_outputs),
                **run.metadata,
            },
        )
        
        if self.auto_flush:
            self.cert_client.flush()
        
        logger.debug(
            f"CERT CrewAI: Traced crew {run.run_id} "
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
