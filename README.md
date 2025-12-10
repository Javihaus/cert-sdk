# CERT SDK

[![CI](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/cert-sdk.svg)](https://badge.fury.io/py/cert-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)


**Production-grade observability for LLM applications.** CERT SDK provides non-blocking telemetry collection for monitoring reliability, latency, and correctness of AI systems in production.

```python
from cert import CertClient

client = CertClient(api_key="cert_xxx", project="my-project")

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris.",
    duration_ms=245.3,
    prompt_tokens=12,
    completion_tokens=8
)
```

View traces at [dashboard.cert-framework.com](https://dashboard.cert-framework.com)

---

## Installation

```bash
pip install cert-sdk
```

**Requirements:** Python 3.9+

---

## Quick Start

### 1. Get Your API Key

Sign up at [dashboard.cert-framework.com](https://dashboard.cert-framework.com) and create an API key.

### 2. Initialize the Client

```python
from cert import CertClient

client = CertClient(
    api_key="cert_xxx",        # Required: Your CERT API key
    project="my-llm-app"       # Required: Project name for organizing traces
)
```

### 3. Trace LLM Calls

```python
import time

start = time.perf_counter()
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
duration_ms = (time.perf_counter() - start) * 1000

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="Hello",
    output_text=response.choices[0].message.content,
    duration_ms=duration_ms,
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens
)
```

### 4. Cleanup

```python
client.flush()   # Send pending traces
client.close()   # Shutdown background worker
```

---

## Core Concepts

### Projects

Traces are organized by **project**. Each project appears separately in the dashboard with its own metrics and visualizations.

```python
# Production traffic
prod_client = CertClient(api_key=key, project="chatbot-prod")

# Development/testing
dev_client = CertClient(api_key=key, project="chatbot-dev")
```

**Best Practice:** Use separate projects for different evaluation paradigms (agentic vs generation) since they have different applicable metrics.

### Evaluation Modes

CERT evaluates traces differently based on their type:

| Mode | When to Use | Key Metrics |
|------|-------------|-------------|
| `agentic` | Agent with tools | SGI, Grounding, Tool Integration |
| `rag` | Retrieval-augmented | Faithfulness, Citation Accuracy |
| `generation` | Direct LLM output | Self-Consistency, Format Compliance |

The SDK **auto-detects** the mode based on what you provide:

```python
# Auto-detected as "agentic" (tool_calls present)
client.trace(..., tool_calls=[{"name": "search", "input": {...}, "output": {...}}])

# Auto-detected as "rag" (context present, no tools)
client.trace(..., context="Retrieved document text...")

# Auto-detected as "generation" (neither present)
client.trace(...)
```

Override with `eval_mode="agentic"` if needed.

---

## API Reference

### `CertClient`

```python
CertClient(
    api_key: str,                    # Required: CERT API key
    project: str = "default",        # Project name
    dashboard_url: str = "https://dashboard.cert-framework.com",
    batch_size: int = 10,            # Traces per batch
    flush_interval: float = 5.0,     # Seconds between auto-flushes
    max_queue_size: int = 1000,      # Max queued traces
    timeout: float = 5.0             # HTTP timeout
)
```

### `trace()`

Log an LLM interaction. **Non-blocking**—returns immediately while trace is queued.

```python
client.trace(
    # === REQUIRED ===
    provider: str,           # LLM provider: "openai", "anthropic", "google", etc.
    model: str,              # Model name: "gpt-4o", "claude-sonnet-4", etc.
    input_text: str,         # User prompt or input
    output_text: str,        # Model response
    duration_ms: float,      # Request latency in milliseconds
    
    # === TOKENS ===
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    # total_tokens computed automatically
    
    # === TIMING ===
    start_time: datetime = None,   # Request start (auto-generated if omitted)
    end_time: datetime = None,     # Request end (auto-generated if omitted)
    
    # === STATUS ===
    status: str = "success",       # "success" or "error"
    error_message: str = None,     # Error details if status="error"
    
    # === DISTRIBUTED TRACING ===
    trace_id: str = None,          # Correlation ID (auto-generated if omitted)
    span_id: str = None,           # Unique span ID (auto-generated if omitted)
    parent_span_id: str = None,    # Parent span for nested calls
    name: str = None,              # Operation name (default: "{provider}.{model}")
    kind: str = "CLIENT",          # Span kind: "CLIENT", "SERVER", "INTERNAL"
    
    # === EVALUATION CONFIG ===
    eval_mode: str = "auto",       # "agentic", "rag", "generation", or "auto"
    context: str = None,           # RAG context / retrieved documents
    tool_calls: list = None,       # Tool invocations (agentic mode)
    goal_description: str = None,  # Task objective for agentic evaluation
    output_schema: dict = None,    # Expected output structure
    
    # === CUSTOM DATA ===
    metadata: dict = None          # Arbitrary key-value pairs
) -> str                           # Returns trace_id
```

### `flush()`

Send all pending traces immediately. Blocks until complete.

```python
client.flush(timeout: float = 10.0)
```

### `close()`

Flush pending traces and shutdown background worker. Call when shutting down your application.

```python
client.close()
```

### `get_stats()`

Get client statistics.

```python
stats = client.get_stats()
# {"traces_sent": 150, "traces_failed": 0, "traces_queued": 3}
```

---

## How to use cert SDK

### Basic Generation

Simple LLM calls without tools or retrieval:

```python
client.trace(
    provider="anthropic",
    model="claude-sonnet-4",
    input_text=user_message,
    output_text=response,
    duration_ms=duration,
    prompt_tokens=usage.input_tokens,
    completion_tokens=usage.output_tokens
)
# eval_mode auto-detected as "generation"
```

### RAG Pipeline

Include retrieved context for faithfulness evaluation:

```python
# Retrieved documents
context = "\n".join([doc.page_content for doc in retrieved_docs])

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text=query,
    output_text=response,
    duration_ms=duration,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    context=context  # Enables faithfulness metrics
)
# eval_mode auto-detected as "rag"
```

### Agentic Pipeline

Capture tool calls for grounding evaluation:

```python
tool_calls = [
    {
        "name": "get_weather",
        "input": {"city": "Paris"},
        "output": {"temperature": 22, "condition": "sunny"}
    },
    {
        "name": "calculator",
        "input": {"expression": "22 * 1.8 + 32"},
        "output": {"result": 71.6}
    }
]

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What's the weather in Paris in Fahrenheit?",
    output_text="It's 72°F and sunny in Paris.",
    duration_ms=1250.5,
    prompt_tokens=150,
    completion_tokens=45,
    tool_calls=tool_calls,
    goal_description="Convert weather to Fahrenheit"
)
# eval_mode auto-detected as "agentic"
```

### Error Tracking

Capture failures for debugging:

```python
try:
    response = llm.invoke(prompt)
    client.trace(
        ...,
        output_text=response.content,
        status="success"
    )
except Exception as e:
    client.trace(
        ...,
        output_text="",
        status="error",
        error_message=str(e)
    )
```

### Distributed Tracing

Correlate multiple spans in a single request:

```python
import uuid

# Parent trace
trace_id = str(uuid.uuid4())

# First LLM call (planning)
client.trace(
    ...,
    trace_id=trace_id,
    span_id="span-plan",
    name="agent.plan"
)

# Second LLM call (execution)
client.trace(
    ...,
    trace_id=trace_id,
    span_id="span-execute",
    parent_span_id="span-plan",
    name="agent.execute"
)
```

### Context Manager

Automatic timing and error capture:

```python
from cert import CertClient, TraceContext

with TraceContext(
    client,
    provider="openai",
    model="gpt-4o",
    input_text=prompt
) as ctx:
    response = llm.invoke(prompt)
    ctx.set_output(response.content)
    ctx.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

# Timing, status, and errors captured automatically
```

---

## Framework Integration

### LangChain / LangGraph

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from cert import CertClient
import time

client = CertClient(api_key=key, project="langchain-app")
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[...])

def run_with_tracing(query: str):
    start = time.perf_counter()
    result = agent.invoke({"messages": [("human", query)]})
    duration = (time.perf_counter() - start) * 1000
    
    # Extract tool calls from message history
    tool_calls = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc["name"],
                    "input": tc["args"],
                    "output": None  # Match with ToolMessage
                })
    
    final_output = result["messages"][-1].content
    
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text=query,
        output_text=final_output,
        duration_ms=duration,
        tool_calls=tool_calls if tool_calls else None
    )
    
    return final_output
```

### OpenAI Direct

```python
import openai
from cert import CertClient
import time

client = CertClient(api_key=key, project="openai-app")

def traced_completion(messages: list, model: str = "gpt-4o"):
    start = time.perf_counter()
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    
    duration = (time.perf_counter() - start) * 1000
    
    client.trace(
        provider="openai",
        model=model,
        input_text=messages[-1]["content"],
        output_text=response.choices[0].message.content,
        duration_ms=duration,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
    
    return response
```

### Anthropic Direct

```python
import anthropic
from cert import CertClient
import time

client = CertClient(api_key=key, project="anthropic-app")
anthropic_client = anthropic.Anthropic()

def traced_message(prompt: str, model: str = "claude-sonnet-4-20250514"):
    start = time.perf_counter()
    
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    duration = (time.perf_counter() - start) * 1000
    
    client.trace(
        provider="anthropic",
        model=model,
        input_text=prompt,
        output_text=response.content[0].text,
        duration_ms=duration,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens
    )
    
    return response
```

---

## Configuration

### Environment Variables

```bash
export CERT_API_KEY="cert_xxx"
export CERT_PROJECT="my-project"
export CERT_DASHBOARD_URL="https://dashboard.cert-framework.com"
```

```python
import os
from cert import CertClient

client = CertClient(
    api_key=os.environ["CERT_API_KEY"],
    project=os.environ.get("CERT_PROJECT", "default")
)
```

### Google Colab

```python
from google.colab import userdata
from cert import CertClient

client = CertClient(
    api_key=userdata.get("CERT_API_KEY"),
    project="colab-experiments"
)
```

### Logging

Enable debug logging to see trace activity:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("cert").setLevel(logging.DEBUG)
```

---

## Best Practices

### Project Organization

| Use Case | Project Name |
|----------|--------------|
| Production traffic | `myapp-prod` |
| Staging environment | `myapp-staging` |
| A/B test variant | `myapp-experiment-v2` |
| Agentic evaluation | `myapp-agentic` |
| RAG evaluation | `myapp-rag` |

### Graceful Shutdown

Always flush and close in production:

```python
import atexit

client = CertClient(api_key=key, project="prod")
atexit.register(client.close)
```

### Batch Processing

For high-throughput applications, adjust batch settings:

```python
client = CertClient(
    api_key=key,
    project="high-volume",
    batch_size=50,           # Larger batches
    flush_interval=2.0,      # More frequent flushes
    max_queue_size=5000      # Larger queue
)
```

### Sensitive Data

Avoid logging PII in traces. Sanitize inputs/outputs if needed:

```python
def sanitize(text: str) -> str:
    # Remove emails, phone numbers, etc.
    return re.sub(r'\S+@\S+', '[EMAIL]', text)

client.trace(
    ...,
    input_text=sanitize(user_input),
    output_text=sanitize(response)
)
```

---

## Troubleshooting

### Traces Not Appearing

1. **Check API key:** Ensure `api_key` is valid
2. **Flush before exit:** Call `client.flush()` before program ends
3. **Check stats:** `client.get_stats()` shows sent/failed counts
4. **Enable logging:** Set `logging.getLogger("cert").setLevel(logging.DEBUG)`

### High Memory Usage

Reduce queue size for memory-constrained environments:

```python
client = CertClient(..., max_queue_size=100)
```

### Network Timeouts

Increase timeout for slow networks:

```python
client = CertClient(..., timeout=15.0)
```

---

## Changelog

### v0.3.0

- Added distributed tracing: `trace_id`, `span_id`, `parent_span_id`
- Added timing fields: `start_time`, `end_time`
- Added status tracking: `status`, `error_message`
- Added `TraceContext` context manager
- Added `total_tokens` auto-computation
- `trace()` now returns `trace_id` for correlation

### v0.2.0

- Added `project` parameter for trace organization
- Added `eval_mode` auto-detection
- Added `tool_calls` for agentic evaluation
- Added `goal_description` for task objectives

### v0.1.0

- Initial release
- Non-blocking trace collection
- Background batch sending

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- **Dashboard:** [dashboard.cert-framework.com](https://dashboard.cert-framework.com)
- **GitHub:** [github.com/Javihaus/cert-sdk](https://github.com/Javihaus/cert-sdk)
- **Issues:** [github.com/Javihaus/cert-sdk/issues](https://github.com/Javihaus/cert-sdk/issues)
