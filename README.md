# CERT SDK

[![CI](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Javihaus/cert-sdk/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/cert-sdk.svg)](https://badge.fury.io/py/cert-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Observability SDK for LLM applications. Non-blocking telemetry collection with automatic batching.

## Install
```bash
pip install cert-sdk
```

## Usage
```python
from cert import CertClient

client = CertClient(
    api_key="cert_xxx",      # Get one at cert-framework.com
    project="my-app"
)

# Trace any LLM call
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris.",
    duration_ms=245.3,
    prompt_tokens=12,
    completion_tokens=8
)

# Always flush before exit
client.flush()
client.close()
```

View traces at [cert-framework.com](https://cert-framework.com)

## Evaluation Modes

The SDK auto-detects evaluation mode based on what you provide:

| You provide | Mode detected | Metrics enabled |
|-------------|---------------|-----------------|
| `tool_calls=[...]` | `agentic` | Tool grounding, goal completion |
| `context="..."` | `rag` | Faithfulness, citation accuracy |
| Neither | `generation` | Self-consistency, format compliance |
```python
# Agentic: pass tool_calls
client.trace(
    ...,
    tool_calls=[{"name": "search", "input": {"q": "paris"}, "output": {"result": "..."}}],
    goal_description="Find weather information"
)

# RAG: pass context
client.trace(
    ...,
    context="Retrieved: Paris is the capital of France..."
)

# Generation: just the basics (default)
client.trace(...)
```

## API

### `CertClient(api_key, project, **kwargs)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | Your CERT API key |
| `project` | `"default"` | Project name for grouping traces |
| `batch_size` | `10` | Traces per HTTP request |
| `flush_interval` | `5.0` | Seconds between auto-flushes |
| `max_queue_size` | `1000` | Max queued traces before dropping |
| `timeout` | `5.0` | HTTP timeout in seconds |

### `client.trace(**kwargs) -> str`

Returns `trace_id`. Non-blocking—queues trace for background send.

**Required:**
- `provider` — LLM provider (`"openai"`, `"anthropic"`, `"google"`, etc.)
- `model` — Model name (`"gpt-4o"`, `"claude-sonnet-4"`, etc.)
- `input_text` — The prompt
- `output_text` — The response
- `duration_ms` — Latency in milliseconds

**Optional:**
- `prompt_tokens`, `completion_tokens` — Token counts
- `status` — `"success"` (default) or `"error"`
- `error_message` — Error details when `status="error"`
- `trace_id`, `span_id`, `parent_span_id` — Distributed tracing correlation
- `context` — Retrieved documents (enables RAG metrics)
- `tool_calls` — List of tool invocations (enables agentic metrics)
- `goal_description` — Task objective for agentic evaluation
- `metadata` — Arbitrary key-value pairs

### `client.flush(timeout=10.0)`

Send all pending traces. Blocks until complete.

### `client.close()`

Flush + shutdown background worker. Call on application exit.

### `client.get_stats() -> dict`

Returns `{"traces_sent": int, "traces_failed": int, "traces_queued": int}`.

## Production Setup
```python
import atexit
from cert import CertClient

client = CertClient(
    api_key=os.environ["CERT_API_KEY"],
    project=os.environ.get("CERT_PROJECT", "prod")
)
atexit.register(client.close)  # Ensure flush on shutdown
```

## Debug
```python
import logging
logging.getLogger("cert").setLevel(logging.DEBUG)
```

## License

Apache 2.0

## Links

- **Dashboard:** [dashboard.cert-framework.com](https://dashboard.cert-framework.com)
- **GitHub:** [github.com/Javihaus/cert-sdk](https://github.com/Javihaus/cert-sdk)
- **Issues:** [github.com/Javihaus/cert-sdk/issues](https://github.com/Javihaus/cert-sdk/issues)
