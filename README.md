<div align="center">
<img src="docs/cert_logo3.png" alt="CERT" width="20%" />
</div>


# CERT SDK

[![PyPI version](https://badge.fury.io/py/cert-sdk.svg)](https://badge.fury.io/py/cert-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml)

**Production-grade observability for LLM applications.** Non-blocking telemetry with automatic batching, two-mode evaluation, and EU AI Act compliance support.

```python
from cert import CertClient

client = CertClient(api_key="cert_xxx", project="my-app")

client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris.",
    duration_ms=245,
    prompt_tokens=12,
    completion_tokens=8
)
```

---

## Installation

```bash
pip install cert-sdk
```

**Requirements:** Python 3.8+

---

## Quick Start

### 1. Get an API Key

Sign up at [dashboard.cert-framework.com](https://dashboard.cert-framework.com) and create an API key.

### 2. Initialize the Client

```python
from cert import CertClient

client = CertClient(
    api_key="cert_xxx",    # Your API key
    project="my-app"       # Project name for organization
)
```

### 3. Trace LLM Calls

```python
# After each LLM call in your application
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text=user_prompt,
    output_text=llm_response,
    duration_ms=response_time,
    prompt_tokens=usage.prompt_tokens,
    completion_tokens=usage.completion_tokens
)
```

### 4. Shutdown Properly

```python
# Before application exit
client.close()
```

---

## Evaluation Modes

CERT uses a two-mode evaluation system based on whether your LLM has access to source knowledge:

| Mode | When to Use | Metrics |
|------|-------------|---------|
| **Grounded** | LLM has access to source documents, tool outputs, or retrieved context | Relevance, Source Accuracy, Faithfulness |
| **Ungrounded** | LLM generates from its training alone | Relevance, Coherence, Completeness |

The SDK auto-detects the appropriate mode, or you can set it explicitly.

### Grounded Evaluation

Use when your LLM response is based on specific source material—retrieved documents, tool outputs, or conversation history.

```python
# RAG Application
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What are our Q3 revenue targets?",
    output_text="According to the planning document, Q3 targets are $2.5M.",
    duration_ms=340,
    knowledge_base="Q3 Planning Doc: Revenue target is $2.5M for Q3...",
    context_source="retrieval"
)
```

```python
# Agent with Tool Calls
client.trace(
    provider="anthropic",
    model="claude-sonnet-4",
    input_text="What's the weather in Paris?",
    output_text="It's currently 18°C and sunny in Paris.",
    duration_ms=1200,
    tool_calls=[
        {
            "name": "weather_api",
            "input": {"city": "Paris"},
            "output": {"temp_c": 18, "condition": "sunny"}
        }
    ]
)
# Context automatically extracted from tool outputs
```

### Ungrounded Evaluation

Use when your LLM generates content from its training without external sources.

```python
# Creative Generation
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="Write a haiku about machine learning",
    output_text="Neural paths converge\nPatterns emerge from chaos\nMachines learn to see",
    duration_ms=890
)
# Automatically evaluated as ungrounded
```

---

## Complete API Reference

### CertClient

```python
from cert import CertClient

client = CertClient(
    api_key: str,                    # Required. Your CERT API key
    project: str = "default",        # Project name for trace grouping
    dashboard_url: str = "https://dashboard.cert-framework.com",
    batch_size: int = 10,            # Traces per HTTP batch
    flush_interval: float = 5.0,     # Seconds between auto-flushes
    max_queue_size: int = 1000,      # Max queued traces before dropping
    timeout: float = 5.0,            # HTTP request timeout
    auto_extract_knowledge: bool = True  # Extract knowledge from tool outputs
)
```

### client.trace()

Log an LLM interaction. **Non-blocking**—traces are queued and sent in batches.

```python
trace_id = client.trace(
    # === REQUIRED ===
    provider: str,           # "openai", "anthropic", "google", "bedrock", etc.
    model: str,              # "gpt-4o", "claude-sonnet-4", "gemini-pro", etc.
    input_text: str,         # User prompt or messages
    output_text: str,        # LLM response
    duration_ms: float,      # Request latency in milliseconds
    
    # === TOKENS ===
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    
    # === EVALUATION ===
    evaluation_mode: str = "auto",      # "grounded", "ungrounded", or "auto"
    knowledge_base: str = None,         # Source documents for grounded eval
    context_source: str = None,         # "retrieval", "tools", "conversation", "user_provided"
    
    # === TOOL CALLS ===
    tool_calls: List[dict] = None,      # [{"name": str, "input": dict, "output": any}]
    goal_description: str = None,       # Task objective
    
    # === DISTRIBUTED TRACING ===
    trace_id: str = None,               # Correlation ID (auto-generated if not provided)
    span_id: str = None,                # Span ID (auto-generated if not provided)
    parent_span_id: str = None,         # Parent span for nested traces
    name: str = None,                   # Operation name (defaults to provider.model)
    
    # === STATUS ===
    status: str = "success",            # "success" or "error"
    error_message: str = None,          # Error details when status="error"
    
    # === CUSTOM DATA ===
    metadata: dict = None               # Arbitrary key-value pairs
)
```

**Returns:** `str` — The trace ID for correlation.

### client.flush()

Send all pending traces immediately. Blocks until complete.

```python
client.flush(timeout: float = 10.0)
```

### client.close()

Flush pending traces and shutdown the background worker. **Always call before application exit.**

```python
client.close()
```

### client.get_stats()

Get client statistics for monitoring.

```python
stats = client.get_stats()
# {"traces_sent": 142, "traces_failed": 0, "traces_queued": 3}
```

---

## Context Manager

Use `TraceContext` for automatic timing and error capture:

```python
from cert import CertClient, TraceContext

client = CertClient(api_key="cert_xxx")

with TraceContext(
    client,
    provider="openai",
    model="gpt-4o",
    input_text="Explain quantum computing"
) as ctx:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain quantum computing"}]
    )
    ctx.set_output(response.choices[0].message.content)
    ctx.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

# Trace automatically sent with timing and status
```

### TraceContext Methods

```python
ctx.set_output(text: str)                    # Set response text
ctx.set_tokens(prompt: int, completion: int) # Set token counts
ctx.set_tool_calls(calls: List[dict])        # Add tool calls
ctx.set_knowledge_base(kb: str, source: str) # Set knowledge base
```

---

## Framework Integrations

### LangChain

```python
from cert import CertClient
from cert.integrations.langchain import CertCallbackHandler

client = CertClient(api_key="cert_xxx")
handler = CertCallbackHandler(client)

# Use with any LangChain chain
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = chain.invoke({"query": "..."}, config={"callbacks": [handler]})
```

### AutoGen

```python
from cert import CertClient
from cert.integrations.autogen import AutoGenInstrumentor

client = CertClient(api_key="cert_xxx")
instrumentor = AutoGenInstrumentor(client)

# Instrument your agents
instrumentor.instrument(assistant_agent)
instrumentor.instrument(user_proxy)
```

### CrewAI

```python
from cert import CertClient
from cert.integrations.crewai import CrewAIInstrumentor

client = CertClient(api_key="cert_xxx")
instrumentor = CrewAIInstrumentor(client)

# Instrument your crew
instrumentor.instrument(crew)
```

---

## Production Best Practices

### Environment Configuration

```python
import os
from cert import CertClient

client = CertClient(
    api_key=os.environ["CERT_API_KEY"],
    project=os.environ.get("CERT_PROJECT", "production"),
    batch_size=20,           # Larger batches for high-volume
    flush_interval=10.0,     # Less frequent flushes
    max_queue_size=5000      # Larger buffer
)
```

### Graceful Shutdown

```python
import atexit
import signal

client = CertClient(api_key=os.environ["CERT_API_KEY"])

# Register cleanup handlers
atexit.register(client.close)

def handle_signal(signum, frame):
    client.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
```

### Error Handling

```python
try:
    response = llm.invoke(prompt)
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text=prompt,
        output_text=response.content,
        duration_ms=elapsed,
        status="success"
    )
except Exception as e:
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text=prompt,
        output_text="",
        duration_ms=elapsed,
        status="error",
        error_message=str(e)
    )
    raise
```

### Health Monitoring

```python
# Periodic health check
stats = client.get_stats()

if stats["traces_failed"] > stats["traces_sent"] * 0.01:
    logger.warning(f"High trace failure rate: {stats}")

if stats["traces_queued"] > 500:
    logger.warning(f"Trace queue backing up: {stats['traces_queued']}")
```

---

## Debugging

Enable debug logging to troubleshoot issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("cert").setLevel(logging.DEBUG)
```

**Common Issues:**

| Symptom | Cause | Solution |
|---------|-------|----------|
| `traces_failed` increasing | Network or auth issues | Check API key and network connectivity |
| `traces_queued` growing | Slow network or high volume | Increase `batch_size` or `flush_interval` |
| Traces not appearing | Missing `close()` call | Always call `client.close()` on shutdown |

---

## Data Types

### ToolCall

```python
{
    "name": str,           # Required. Tool/function name
    "input": dict,         # Tool input arguments
    "output": any,         # Tool return value
    "error": str           # Error message if tool failed
}
```

### EvaluationMode

```python
"grounded"    # Has knowledge_base → full metric suite
"ungrounded"  # No knowledge_base → basic metrics
"auto"        # Auto-detect based on provided data (default)
```

### ContextSource

```python
"retrieval"      # From RAG/vector search
"tools"          # From tool/function outputs
"conversation"   # From conversation history
"user_provided"  # Explicitly provided by user
```

---

## Changelog

### v0.4.0 (Current)

- **New:** Two-mode evaluation architecture (grounded/ungrounded)
- **New:** `knowledge_base` parameter replaces `context`
- **New:** `context_source` parameter for analytics
- **New:** Automatic knowledge extraction from tool outputs
- **Deprecated:** `eval_mode` parameter (use `evaluation_mode`)
- **Deprecated:** `context` parameter (use `knowledge_base`)

### v0.3.x

- Initial release with three-mode evaluation (rag/generation/agentic)

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Links

| Resource | URL |
|----------|-----|
| **Dashboard** | [dashboard.cert-framework.com](https://dashboard.cert-framework.com) |
| **Documentation** | [docs.cert-framework.com](https://docs.cert-framework.com) |
| **GitHub** | [github.com/Javihaus/cert-sdk](https://github.com/Javihaus/cert-sdk) |
| **Issues** | [github.com/Javihaus/cert-sdk/issues](https://github.com/Javihaus/cert-sdk/issues) |
| **PyPI** | [pypi.org/project/cert-sdk](https://pypi.org/project/cert-sdk) |

---

## Support

- **Bug Reports:** [GitHub Issues](https://github.com/Javihaus/cert-sdk/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/Javihaus/cert-sdk/discussions)
- **Security Issues:** security@cert-framework.com

---

<p align="center">
  <strong>Built for production LLM applications.</strong><br>
  <a href="https://cert-framework.com">cert-framework.com</a>
</p>
