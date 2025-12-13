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

## Quick Start (4 lines!)
```python
from cert import CertClient

client = CertClient(api_key="cert_xxx", project="my-app")

# That's it! Just 4 required params:
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital of France?",
    output_text="The capital of France is Paris."
)

client.flush()
```

## Evaluation Modes

The SDK auto-detects evaluation mode based on what you provide:

| You provide | Mode detected | Metrics enabled |
|-------------|---------------|-----------------|
| `knowledge_base="..."` | `grounded` | Faithfulness, hallucination detection |
| `tool_calls=[...]` | `agentic` | Tool grounding, goal completion |
| Neither | `ungrounded` | Self-consistency, format compliance |

```python
# Grounded: verify response against context
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What is the capital?",
    output_text="Paris is the capital.",
    knowledge_base="France is a country. Paris is its capital.",
    evaluation_mode="grounded"  # or auto-detected
)

# Agentic: multi-step with tool calls
client.trace(
    provider="openai",
    model="gpt-4o",
    input_text="What's the weather?",
    output_text="It's 72°F in NYC.",
    tool_calls=[{"name": "weather_api", "input": {"city": "NYC"}, "output": {"temp": 72}}],
    evaluation_mode="agentic"
)

# Ungrounded: pure generation (default)
client.trace(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    input_text="Write a haiku about coding",
    output_text="Lines of logic flow..."
)
```

## API Reference

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

**Required (only 4!):**
- `provider` — LLM provider (`"openai"`, `"anthropic"`, `"google"`, etc.)
- `model` — Model name (`"gpt-4o"`, `"claude-sonnet-4-20250514"`, etc.)
- `input_text` — The prompt
- `output_text` — The response

**Optional:**
- `duration_ms` — Latency in milliseconds (defaults to 0)
- `prompt_tokens`, `completion_tokens` — Token counts
- `status` — `"success"` (default) or `"error"`
- `error_message` — Error details when `status="error"`
- `trace_id`, `span_id`, `parent_span_id` — Distributed tracing correlation
- `evaluation_mode` — `"grounded"`, `"ungrounded"`, `"agentic"`, or `"auto"` (default)
- `knowledge_base` — Context/documents for grounded evaluation (alias: `context`)
- `tool_calls` — List of tool invocations for agentic evaluation
- `goal_description` — Task objective for agentic evaluation
- `task_type` — Task category (e.g., `"qa"`, `"chat"`, `"summarization"`)
- `context_source` — Context origin (e.g., `"retrieval"`, `"conversation"`)
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
