<div align="center">
<img src="docs/logo_cert_a_v3.png" alt="CERT" width="20%" />
</div>

# CERT SDK

[![CI](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/cert-sdk.svg)](https://badge.fury.io/py/cert-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Observability SDK for LLM applications. Non-blocking telemetry with automatic batching.

## Installation

```bash
pip install cert-sdk
```

Or from source:
```bash
pip install git+https://github.com/Javihaus/cert-sdk.git
```

---

## Quick Start

```python
from cert import CertClient

client = CertClient(api_key="cert_xxx", project="my-app")

# Minimal trace - just 4 required params
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris."
)

client.flush()  # Send pending traces before exit
```

That's it. Everything else is optional.

---

## Evaluation Modes

The SDK supports three evaluation modes that determine which metrics are computed:

| Mode | When to Use | Metrics Enabled |
|------|-------------|-----------------|
| `ungrounded` | Pure generation, no external context | Coherence, format compliance, self-consistency |
| `grounded` | RAG, Q&A with context, fact verification | Faithfulness, hallucination detection, citation accuracy |
| `agentic` | Multi-step agents with tool calls | Tool grounding, goal completion, trajectory analysis |

### Auto-Detection

The SDK auto-detects the appropriate mode:

```python
# Detected as "ungrounded" - no context provided
client.trace(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    input_text="Write a haiku about coding",
    output_text="Lines of logic flow\nSilent bugs hide in the dark\nTests bring peace of mind"
)

# Detected as "grounded" - knowledge_base provided
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital?",
    output_text="Paris is the capital of France.",
    knowledge_base="France is a country in Western Europe. Paris is its capital city."
)

# Detected as "agentic" - tool_calls provided
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What's the weather in NYC?",
    output_text="It's currently 72°F and sunny in New York City.",
    tool_calls=[
        {"name": "weather_api", "input": {"city": "NYC"}, "output": {"temp": 72, "condition": "sunny"}}
    ]
)
```

### Explicit Mode

Override auto-detection when needed:

```python
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="Summarize this document",
    output_text="The document discusses...",
    evaluation_mode="ungrounded"  # Force ungrounded even if context exists
)
```

---

## Complete Parameter Reference

### `CertClient` Constructor

```python
client = CertClient(
    api_key="cert_xxx",           # Required: Your CERT API key
    project="my-app",             # Project name (default: "default")
    dashboard_url="https://...",  # Dashboard URL (default: production)
    batch_size=10,                # Traces per HTTP batch (default: 10)
    flush_interval=5.0,           # Seconds between auto-flushes (default: 5.0)
    max_queue_size=1000,          # Max queued traces before dropping (default: 1000)
    timeout=5.0,                  # HTTP timeout in seconds (default: 5.0)
    auto_extract_knowledge=True   # Extract knowledge from tool outputs (default: True)
)
```

### `client.trace()` Method

Returns a trace ID (UUID string). Non-blocking—traces are queued and sent in batches.

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | LLM provider: `"openai"`, `"anthropic"`, `"google"`, `"cohere"`, etc. |
| `model` | `str` | Model identifier: `"gpt-4o"`, `"claude-sonnet-4-20250514"`, `"gemini-1.5-pro"`, etc. |
| `input_text` | `str` | The prompt or input messages sent to the model |
| `output_text` | `str` | The model's response |

#### Timing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_ms` | `float` | `0` | Request latency in milliseconds |
| `start_time` | `datetime` | `now()` | When the LLM call started |
| `end_time` | `datetime` | `now()` | When the LLM call completed |

#### Token Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_tokens` | `int` | `0` | Number of input tokens |
| `completion_tokens` | `int` | `0` | Number of output tokens |

#### Status Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `str` | `"success"` | Trace status: `"success"` or `"error"` |
| `error_message` | `str` | `None` | Error details when `status="error"` |

#### Distributed Tracing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace_id` | `str` | auto-generated | UUID to correlate related spans |
| `span_id` | `str` | auto-generated | Unique identifier for this span |
| `parent_span_id` | `str` | `None` | Parent span for nested operations |
| `name` | `str` | `"{provider}.{model}"` | Human-readable operation name |
| `kind` | `str` | `"CLIENT"` | Span kind: `"CLIENT"`, `"SERVER"`, `"INTERNAL"`, `"PRODUCER"`, `"CONSUMER"` |

#### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluation_mode` | `str` | `"auto"` | `"grounded"`, `"ungrounded"`, `"agentic"`, or `"auto"` |
| `knowledge_base` | `str` | `None` | Context/documents for grounded evaluation |
| `tool_calls` | `list` | `None` | List of tool invocations (see format below) |
| `goal_description` | `str` | `None` | Task objective for agentic evaluation |
| `output_schema` | `dict` | `None` | Expected output structure for validation |

**Aliases for compatibility:**
- `eval_mode` → alias for `evaluation_mode`
- `context` → alias for `knowledge_base`

#### Metadata Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `str` | `None` | Task category: `"qa"`, `"chat"`, `"summarization"`, `"generation"`, etc. |
| `context_source` | `str` | `None` | Context origin: `"retrieval"`, `"tools"`, `"conversation"`, `"user_provided"` |
| `metadata` | `dict` | `None` | Arbitrary key-value pairs for custom data |

---

## Tool Calls Format

For agentic evaluation, `tool_calls` should be a list of dictionaries:

```python
tool_calls = [
    {
        "name": "search_api",              # Required: tool name
        "input": {"query": "weather NYC"}, # Optional: input parameters
        "output": {"temp": 72},            # Optional: tool output (used for grounding)
        "error": None                      # Optional: error message if tool failed
    },
    {
        "name": "calculator",
        "input": {"expression": "2 + 2"},
        "output": 4
    }
]
```

When `auto_extract_context=True` (default), tool outputs are automatically concatenated as the context for grounding evaluation.

---

## Use Examples

### Basic Generation (Ungrounded)

```python
client.trace(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    input_text="Write a product description for wireless headphones",
    output_text="Experience crystal-clear audio with our premium wireless headphones...",
    task_type="generation"
)
```

### RAG / Q&A with Context (Grounded)

```python
# Retrieved documents become the knowledge_base
retrieved_docs = """
Document 1: The Eiffel Tower was completed in 1889.
Document 2: It stands 330 meters tall including antennas.
Document 3: Gustave Eiffel's company designed and built the tower.
"""

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="When was the Eiffel Tower built and how tall is it?",
    output_text="The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    knowledge_base=retrieved_docs,
    evaluation_mode="grounded",
    context_source="retrieval",
    task_type="qa"
)
```

### Multi-Turn Conversation (Grounded)

```python
# Previous conversation becomes the knowledge_base
conversation_history = """
User: I'm looking for a laptop for video editing.
Assistant: I'd recommend at least 16GB RAM and a dedicated GPU. What's your budget?
User: Around $1500.
"""

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What about the MacBook Pro M3?",
    output_text="The MacBook Pro M3 at $1599 is slightly above your $1500 budget but offers excellent video editing performance with 18GB unified memory.",
    knowledge_base=conversation_history,
    evaluation_mode="grounded",
    context_source="conversation",
    task_type="chat"
)
```

### Agentic with Tool Calls

```python
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="Book a flight from NYC to LA for next Friday",
    output_text="I found a flight on United departing at 9 AM for $350. Should I book it?",
    tool_calls=[
        {
            "name": "search_flights",
            "input": {"from": "NYC", "to": "LA", "date": "2024-12-20"},
            "output": {
                "flights": [
                    {"airline": "United", "departure": "09:00", "price": 350},
                    {"airline": "Delta", "departure": "14:30", "price": 420}
                ]
            }
        }
    ],
    evaluation_mode="agentic",
    goal_description="Book the cheapest available flight",
    task_type="booking"
)
```

### Multi-Agent Pipeline

```python
import uuid

# Shared trace_id for the entire pipeline
pipeline_trace_id = str(uuid.uuid4())

# Agent 1: Research
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="Research recent AI developments",
    output_text="Key developments include: 1) GPT-4 Turbo release...",
    trace_id=pipeline_trace_id,
    span_id="span-research",
    task_type="research"
)

# Agent 2: Analysis (child of research)
client.trace(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    input_text="Analyze the impact of these developments",
    output_text="The implications for enterprise adoption are...",
    trace_id=pipeline_trace_id,
    span_id="span-analysis",
    parent_span_id="span-research",
    knowledge_base="Key developments include: 1) GPT-4 Turbo release...",
    evaluation_mode="grounded",
    task_type="analysis"
)

# Agent 3: Summary (child of analysis)
client.trace(
    provider="openai",
    model="gpt-4o-mini",
    input_text="Write an executive summary",
    output_text="Executive Summary: Recent AI advances signal...",
    trace_id=pipeline_trace_id,
    span_id="span-summary",
    parent_span_id="span-analysis",
    task_type="summarization"
)
```

### Error Handling

```python
try:
    response = openai_client.chat.completions.create(...)
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text=prompt,
        output_text=response.choices[0].message.content,
        duration_ms=response.response_ms,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
except Exception as e:
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text=prompt,
        output_text="",
        status="error",
        error_message=str(e)
    )
    raise
```

---

## TraceContext: Automatic Timing

For automatic timing and error capture, use `TraceContext`:

```python
from cert import CertClient, TraceContext

client = CertClient(api_key="cert_xxx", project="my-app")

with TraceContext(client, provider="openai", model="gpt-4o", input_text=prompt) as ctx:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    ctx.set_output(response.choices[0].message.content)
    ctx.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

# Trace is automatically sent with:
# - duration_ms calculated from context enter/exit
# - status="error" if an exception occurred
# - error_message captured from the exception
```

---

## Agentic AI Frameworks Integrations

### LangChain

```python
from cert import CertClient
from cert.integrations.langchain import CERTLangChainHandler

client = CertClient(api_key="cert_xxx", project="langchain-app")
handler = CERTLangChainHandler(client)

# Use as a callback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("Hello", config={"callbacks": [handler]})

# Or with agents
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
result = agent.invoke({"input": "What's the weather?"}, config={"callbacks": [handler]})
```

### AutoGen

```python
from cert import CertClient
from cert.integrations.autogen import CERTAutoGenHandler

client = CertClient(api_key="cert_xxx", project="autogen-app")
handler = CERTAutoGenHandler(client)

# Trace a conversation
result = handler.trace_conversation(
    initiator=user_proxy,
    recipient=assistant,
    message="Analyze this data"
)
```

### CrewAI

```python
from cert import CertClient
from cert.integrations.crewai import CERTCrewAIHandler

client = CertClient(api_key="cert_xxx", project="crewai-app")
handler = CERTCrewAIHandler(client)

# Trace a crew execution
result = handler.trace_crew(
    crew=my_crew,
    inputs={"topic": "AI trends"}
)
```

---

## Production Setup

```python
import os
import atexit
import logging
from cert import CertClient

# Configure logging (optional)
logging.getLogger("cert").setLevel(logging.WARNING)

# Initialize client
client = CertClient(
    api_key=os.environ["CERT_API_KEY"],
    project=os.environ.get("CERT_PROJECT", "production"),
    batch_size=20,        # Larger batches for high-throughput
    flush_interval=10.0,  # Less frequent flushes
    timeout=10.0          # Longer timeout for reliability
)

# Ensure traces are sent on shutdown
atexit.register(client.close)
```

### Checking Client Health

```python
stats = client.get_stats()
print(f"Sent: {stats['traces_sent']}")
print(f"Failed: {stats['traces_failed']}")
print(f"Queued: {stats['traces_queued']}")

if stats['traces_failed'] > 0:
    logging.warning(f"CERT: {stats['traces_failed']} traces failed to send")
```

---

## Debugging

Enable debug logging to see batch sends:

```python
import logging
logging.getLogger("cert").setLevel(logging.DEBUG)
```

Output:
```
DEBUG:cert:CERT: Background worker started
DEBUG:cert:CERT: Sent 10 traces
DEBUG:cert:CERT: Sent 5 traces
```

---

## API Reference Summary

| Method | Description |
|--------|-------------|
| `CertClient(api_key, project, ...)` | Create a new client |
| `client.trace(provider, model, input_text, output_text, ...)` | Log a trace (non-blocking) |
| `client.flush(timeout=10.0)` | Send all pending traces (blocking) |
| `client.close()` | Flush and shutdown background worker |
| `client.get_stats()` | Get send/fail/queue counts |

---

## License

Apache 2.0

---

## Links

- **Cert Framework:** [cert-framework.com](https://cert-framework.com)
- **GitHub:** [github.com/Javihaus/cert-sdk](https://github.com/Javihaus/cert-sdk)
- **Issues:** [github.com/Javihaus/cert-sdk/issues](https://github.com/Javihaus/cert-sdk/issues)
