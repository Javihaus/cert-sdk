# CERT SDK

[![PyPI version](https://badge.fury.io/py/cert-sdk.svg)](https://pypi.org/project/cert-sdk/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python SDK for [CERT](https://cert-framework-production.up.railway.app) — LLM monitoring, evaluation, and EU AI Act compliance.

## Features

- **Non-blocking** — Background thread handles HTTP, never slows your app
- **Batched** — Efficient bulk sending reduces network overhead
- **Resilient** — Fails silently, never crashes your application
- **Simple** — 4 lines to instrument any LLM call
- **Typed** — Full type hints for IDE autocomplete

## Installation

```bash
pip install cert-sdk
```

## Quick Start

```python
from cert import CertClient

# Initialize client with your API key
client = CertClient(
    api_key="cert_xxx",      # Get from dashboard settings
    project="my-app",        # Organize traces by project
)

# After each LLM call, log a trace
client.trace(
    type="generation",       # "rag", "generation", or "agentic"
    provider="openai",
    model="gpt-4o",
    input_text="Explain quantum computing",
    output_text="Quantum computing uses quantum mechanical phenomena...",
    duration_ms=1523,
    prompt_tokens=15,
    completion_tokens=150,
)

# Important: flush before exit (especially in notebooks/scripts)
client.flush()
client.close()
```

## Trace Types

CERT supports three trace types for different LLM use cases:

### RAG (Retrieval-Augmented Generation)

```python
client.trace(
    type="rag",
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    input_text="What is our refund policy?",
    output_text="Our refund policy allows returns within 30 days...",
    duration_ms=1234,
    context="[Policy Document] Returns are accepted within 30 days of purchase...",
)
```

### Generation (Chat, Completion, Structured Output)

```python
client.trace(
    type="generation",
    provider="openai",
    model="gpt-4o",
    input_text="Write a haiku about Python",
    output_text="Code flows like water\nIndentation guides the way\nBeautiful and clear",
    duration_ms=890,
    output_schema={"type": "string", "format": "haiku"},  # Optional
)
```

### Agentic (Tool Use, Multi-step)

```python
client.trace(
    type="agentic",
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    input_text="What's the weather in Tokyo and book a flight there",
    output_text="The weather in Tokyo is 22°C. I've booked flight JL123...",
    duration_ms=4521,
    tool_calls=[
        {"name": "get_weather", "input": {"city": "Tokyo"}, "output": {"temp": 22}},
        {"name": "book_flight", "input": {"destination": "Tokyo"}, "output": {"flight": "JL123"}},
    ],
    goal_description="Check weather and book travel",
)
```

## Context Manager

Auto-close with context manager:

```python
with CertClient(api_key="cert_xxx", project="my-app") as client:
    client.trace(type="generation", ...)
# Automatically flushes and closes
```

## Notebooks & Scripts

In Jupyter notebooks or short-lived scripts, always call `flush()` before exit:

```python
client = CertClient(api_key="cert_xxx")

# ... your traces ...

# Critical: ensures all traces are sent before kernel/process exits
client.flush()
```

## Configuration

```python
client = CertClient(
    api_key="cert_xxx",         # Required: API key from dashboard
    project="my-app",           # Project name (default: "default")
    dashboard_url=None,         # Custom URL (default: CERT production)
    batch_size=10,              # Traces per batch (default: 10)
    flush_interval=5.0,         # Max seconds between sends (default: 5.0)
    max_queue_size=1000,        # Max queued traces (default: 1000)
    timeout=10.0,               # HTTP timeout seconds (default: 10.0)
)
```

## Monitoring

Check client health:

```python
stats = client.stats()
print(stats)
# {'traces_sent': 42, 'traces_failed': 0, 'traces_dropped': 0, 'traces_queued': 3}
```

Enable debug logging:

```python
import logging
logging.getLogger("cert").setLevel(logging.DEBUG)
```

## API Reference

### `CertClient.trace()`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | `"rag"` \| `"generation"` \| `"agentic"` | Yes | Trace classification |
| `provider` | `str` | Yes | LLM provider name |
| `model` | `str` | Yes | Model identifier |
| `input_text` | `str` | Yes | User prompt/input |
| `output_text` | `str` | Yes | Model response |
| `duration_ms` | `float` | Yes | Request duration |
| `prompt_tokens` | `int` | No | Input token count |
| `completion_tokens` | `int` | No | Output token count |
| `context` | `str` | No | RAG: retrieved documents |
| `output_schema` | `dict` | No | Generation: expected schema |
| `tool_calls` | `list` | No | Agentic: tool invocations |
| `goal_description` | `str` | No | Agentic: task description |
| `metadata` | `dict` | No | Custom key-value pairs |

### `CertClient.flush(timeout=10.0)`

Force-send all pending traces. Returns count sent.

### `CertClient.close()`

Flush and stop background worker. Call on shutdown.

### `CertClient.stats()`

Returns dict with `traces_sent`, `traces_failed`, `traces_dropped`, `traces_queued`.

## Dashboard

View traces, run evaluations, and generate compliance reports:

**[cert-framework-production.up.railway.app](https://cert-framework-production.up.railway.app)**

- Real-time trace monitoring
- Automated quality evaluation (NLI-based)
- Cost and latency analytics
- EU AI Act compliance documentation

## Requirements

- Python 3.8+
- `requests` library

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Links

- [Dashboard](https://cert-framework-production.up.railway.app)
- [GitHub](https://github.com/Javihaus/cert-sdk)
- [Issues](https://github.com/Javihaus/cert-sdk/issues)
