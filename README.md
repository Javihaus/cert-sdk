# CERT SDK

Python SDK for the [CERT LLM Monitoring Dashboard](https://dashboard.cert-framework.com).

Monitor and evaluate your LLM applications in production with automatic context extraction for agentic pipelines.

## Features

- **Non-blocking**: Background thread, never slows your app
- **Batched**: Efficient bulk sending
- **Resilient**: Fails silently, never crashes your app
- **Automatic Context Extraction**: Tool outputs become context in agentic mode
- **Framework Integrations**: LangChain, AutoGen, CrewAI support
- **Multi-Modal Evaluation**: RAG, Generation, and Agentic modes

## Installation

```bash
# Core SDK
pip install cert-sdk

# With framework integrations
pip install cert-sdk[langchain]  # LangChain support
pip install cert-sdk[autogen]    # AutoGen support
pip install cert-sdk[crewai]     # CrewAI support
pip install cert-sdk[all]        # All integrations
```

## Quick Start

### Basic Usage

```python
from cert import CertClient

# Initialize once
client = CertClient(api_key="cert_xxx")

# Trace an LLM call
client.trace(
    provider="openai",
    model="gpt-4",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris.",
    duration_ms=234,
    prompt_tokens=10,
    completion_tokens=15,
)

# Before exit
client.close()
```

### Context Manager

```python
with CertClient(api_key="cert_xxx") as client:
    client.trace(...)
# Auto-closes and flushes on exit
```

## Evaluation Modes

CERT supports three evaluation modes, each with different metric availability:

| Mode | Context Source | Available Metrics |
|------|---------------|-------------------|
| **RAG** | Explicit context | Semantic similarity, NLI, Grounding, SGI |
| **Agentic** | Tool outputs (auto-extracted) | Same as RAG + tool integration metrics |
| **Generation** | None | Instruction adherence, self-consistency, format |

### RAG Mode

Use when you have explicit context (retrieved documents):

```python
client.trace(
    provider="anthropic",
    model="claude-sonnet-4",
    input_text="What are the symptoms?",
    output_text="Common symptoms include fever and fatigue.",
    duration_ms=456,
    context="Patient guide: Common symptoms include fever, fatigue, and headache.",
    eval_mode="rag",  # Optional: auto-detected when context provided
)
```

### Agentic Mode

Use for agent pipelines with tool calls. **Tool outputs automatically become context:**

```python
client.trace(
    provider="openai",
    model="gpt-4",
    input_text="What's the weather in NYC?",
    output_text="It's currently 72°F and sunny in New York City.",
    duration_ms=1500,
    tool_calls=[
        {
            "name": "weather_api",
            "input": {"city": "NYC"},
            "output": {"temperature": 72, "condition": "sunny", "city": "New York City"}
        }
    ],
    goal_description="Get current weather information",
)
# Context is automatically extracted: "[weather_api]: {"temperature": 72, ...}"
```

### Generation Mode

Use for generation without external context:

```python
client.trace(
    provider="anthropic",
    model="claude-sonnet-4",
    input_text="Write a haiku about spring",
    output_text="Cherry blossoms fall\nGentle breeze carries petals\nNew life awakens",
    duration_ms=890,
    eval_mode="generation",
)
```

## Framework Integrations

### LangChain

```python
from cert import CertClient
from cert.integrations.langchain import CERTLangChainHandler

client = CertClient(api_key="cert_xxx")
handler = CERTLangChainHandler(client)

# Option 1: Use with callbacks
result = agent.invoke(
    {"input": "What's the weather?"},
    config={"callbacks": [handler]}
)

# Option 2: Manual tracing with intermediate steps
result = agent.invoke(
    {"input": "Calculate 2+2"},
    return_intermediate_steps=True
)
handler.trace_from_result(
    input_text="Calculate 2+2",
    result=result,
    provider="openai",
    model="gpt-4"
)
```

### AutoGen

```python
from cert import CertClient
from cert.integrations.autogen import CERTAutoGenHandler

client = CertClient(api_key="cert_xxx")
handler = CERTAutoGenHandler(client)

# Trace a conversation
result = handler.trace_conversation(
    initiator=user_proxy,
    recipient=assistant,
    message="What's the weather in NYC?"
)
```

### CrewAI

```python
from cert import CertClient
from cert.integrations.crewai import CERTCrewAIHandler

client = CertClient(api_key="cert_xxx")
handler = CERTCrewAIHandler(client)

# Trace crew execution
result = handler.trace_crew(
    crew=my_crew,
    inputs={"topic": "AI safety research"}
)
```

## Automatic Context Extraction

In agentic mode, the SDK automatically extracts context from tool outputs. This is critical for computing reliability metrics like SGI (Source-Grounding Index).

```python
# These are equivalent:

# Manual context construction
context = "[search]: {'results': ['doc1', 'doc2']}\n\n[calculate]: 42"
client.trace(context=context, tool_calls=tool_calls, ...)

# Automatic extraction (default)
client.trace(tool_calls=tool_calls, ...)  # Context extracted automatically!
```

To disable automatic extraction:

```python
client = CertClient(api_key="cert_xxx", auto_extract_context=False)
```

## Configuration

```python
client = CertClient(
    api_key="cert_xxx",
    dashboard_url="https://dashboard.cert-framework.com",
    batch_size=10,              # Traces per batch (default: 10)
    flush_interval=5.0,         # Seconds between flushes (default: 5.0)
    max_queue_size=1000,        # Max queued traces (default: 1000)
    timeout=5.0,                # HTTP timeout (default: 5.0)
    auto_extract_context=True,  # Auto-extract context from tools (default: True)
)
```

## Utility Functions

### Manual Context Extraction

```python
from cert import extract_context_from_tool_calls

tool_calls = [
    {"name": "search", "output": {"results": ["doc1", "doc2"]}},
    {"name": "calculate", "output": 42},
    {"name": "api_call", "error": "Connection timeout"},
]

context = extract_context_from_tool_calls(tool_calls)
# Result:
# [search]: {"results": ["doc1", "doc2"]}
#
# [calculate]: 42
#
# [api_call] ERROR: Connection timeout
```

## Statistics

```python
stats = client.get_stats()
print(f"Sent: {stats['traces_sent']}")
print(f"Failed: {stats['traces_failed']}")
print(f"Queued: {stats['traces_queued']}")
```

## Dashboard

View your traces at [dashboard.cert-framework.com](https://dashboard.cert-framework.com):

- Real-time monitoring
- Reliability metrics (SGI, NLI scores, grounding)
- Cost analysis
- EU AI Act compliance reports
- Model comparison

## License

Apache 2.0
