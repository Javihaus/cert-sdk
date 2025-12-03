# CERT SDK

Python SDK for the [CERT LLM Monitoring Dashboard](https://dashboard.cert-framework.com).

Monitor your LLM applications in production with zero performance impact.

## Features

- **Non-blocking**: Background thread, never slows your app
- **Batched**: Efficient bulk sending
- **Resilient**: Fails silently, never crashes your app
- **Simple**: 3 lines to instrument

## Installation

```bash
pip install cert-sdk
```

## Quick Start

```python
from cert import CertClient

# Initialize once
client = CertClient(api_key="cert_xxx")  # Get key from dashboard.cert-framework.com

# After each LLM call
response = openai_client.chat.completions.create(...)

client.trace(
    provider="openai",
    model="gpt-4",
    input_text=prompt,
    output_text=response.choices[0].message.content,
    duration_ms=duration,
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
)

# Before exit
client.close()
```

## Context Manager

```python
with CertClient(api_key="cert_xxx") as client:
    # Use client
    client.trace(...)
# Auto-closes on exit
```

## Configuration

```python
client = CertClient(
    api_key="cert_xxx",
    batch_size=10,          # Traces per batch (default: 10)
    flush_interval=5.0,     # Seconds between flushes (default: 5.0)
    max_queue_size=1000,    # Max queued traces (default: 1000)
    timeout=5.0,            # HTTP timeout (default: 5.0)
)
```

## Dashboard

View your traces at [dashboard.cert-framework.com](https://dashboard.cert-framework.com):
- Real-time monitoring
- Cost analysis
- Quality evaluation
- Compliance reports

## License

Apache 2.0
