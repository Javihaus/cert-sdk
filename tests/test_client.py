"""Tests for CERT SDK client."""

import time
from unittest.mock import Mock, patch

import pytest

from cert import CertClient, TraceType, EvalMode, ToolCall


# =============================================================================
# Initialization Tests
# =============================================================================


def test_client_initialization():
    """Client initializes with required api_key."""
    client = CertClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.project == "default"
    assert CertClient.DEFAULT_URL in client.endpoint
    client.close()


def test_client_initialization_with_project():
    """Client accepts project parameter."""
    client = CertClient(api_key="test_key", project="my-project")
    assert client.project == "my-project"
    client.close()


def test_client_initialization_with_custom_url():
    """Client accepts custom dashboard URL."""
    client = CertClient(api_key="test_key", dashboard_url="https://custom.example.com")
    assert "custom.example.com" in client.endpoint
    client.close()


def test_client_initialization_strips_trailing_slash():
    """Client strips trailing slash from URL."""
    client = CertClient(api_key="test_key", dashboard_url="https://example.com/")
    assert client.endpoint == "https://example.com/api/v1/traces"
    client.close()


def test_client_requires_api_key():
    """Client raises ValueError if api_key is empty."""
    with pytest.raises(ValueError, match="api_key is required"):
        CertClient(api_key="")


def test_client_repr():
    """Client has useful repr."""
    client = CertClient(api_key="test_key", project="test-proj")
    assert "test-proj" in repr(client)
    client.close()


# =============================================================================
# Trace Type Validation Tests
# =============================================================================


def test_trace_requires_type():
    """trace() requires type parameter."""
    client = CertClient(api_key="test_key")
    with pytest.raises(TypeError):
        client.trace(
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
    client.close()


def test_trace_validates_type():
    """trace() rejects invalid type values."""
    client = CertClient(api_key="test_key")
    with pytest.raises(ValueError, match="type must be"):
        client.trace(
            type="invalid",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
    client.close()


def test_trace_accepts_rag_type():
    """trace() accepts 'rag' type."""
    client = CertClient(api_key="test_key")
    trace_id = client.trace(
        type="rag",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )
    assert trace_id is not None
    client.close()


def test_trace_accepts_generation_type():
    """trace() accepts 'generation' type."""
    client = CertClient(api_key="test_key")
    trace_id = client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )
    assert trace_id is not None
    client.close()


def test_trace_accepts_agentic_type():
    """trace() accepts 'agentic' type."""
    client = CertClient(api_key="test_key")
    trace_id = client.trace(
        type="agentic",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )
    assert trace_id is not None
    client.close()


# =============================================================================
# Non-blocking Behavior Tests
# =============================================================================


def test_trace_is_nonblocking():
    """trace() returns immediately without blocking."""
    client = CertClient(api_key="test_key")

    start = time.time()
    client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )
    elapsed = time.time() - start

    # Should return in < 10ms (no HTTP call)
    assert elapsed < 0.01

    stats = client.stats()
    assert stats["traces_queued"] >= 1
    client.close()


def test_trace_returns_uuid():
    """trace() returns a valid UUID string."""
    client = CertClient(api_key="test_key")
    trace_id = client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )
    # UUID format: 8-4-4-4-12 hex chars
    assert len(trace_id) == 36
    assert trace_id.count("-") == 4
    client.close()


# =============================================================================
# Batch Sending Tests
# =============================================================================


@patch("cert.client.requests.post")
def test_batch_sends_on_size(mock_post):
    """Batch sends when batch_size is reached."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=2, flush_interval=999)

    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.5)

    assert mock_post.call_count >= 1
    call_args = mock_post.call_args
    assert len(call_args.kwargs["json"]["traces"]) >= 2
    client.close()


@patch("cert.client.requests.post")
def test_batch_sends_on_interval(mock_post):
    """Batch sends when flush_interval elapses."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=100, flush_interval=0.3)

    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.6)

    assert mock_post.call_count >= 1
    client.close()


@patch("cert.client.requests.post")
def test_close_sends_all_traces(mock_post):
    """close() sends all pending traces including worker's batch."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    # close() signals worker to flush its batch and exit
    client.close()

    # All traces should be sent
    stats = client.stats()
    assert stats["traces_sent"] == 2
    assert stats["traces_queued"] == 0


@patch("cert.client.requests.post")
def test_close_flushes(mock_post):
    """close() flushes remaining traces."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    client.close()

    assert mock_post.call_count >= 1


# =============================================================================
# Payload Structure Tests
# =============================================================================


@patch("cert.client.requests.post")
def test_trace_includes_type_field(mock_post):
    """Trace payload includes user-declared type."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        type="rag",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert trace["type"] == "rag"
    client.close()


@patch("cert.client.requests.post")
def test_trace_includes_project_name(mock_post):
    """Trace payload includes project name from client."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", project="my-project", batch_size=1)
    client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert trace["projectName"] == "my-project"
    client.close()


@patch("cert.client.requests.post")
def test_rag_trace_includes_context(mock_post):
    """RAG trace includes context field."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        type="rag",
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=10,
        context="retrieved documents here",
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert trace["context"] == "retrieved documents here"
    assert trace["evalMode"] == "rag"
    client.close()


@patch("cert.client.requests.post")
def test_generation_trace_includes_schema(mock_post):
    """Generation trace includes output_schema field."""
    mock_post.return_value = Mock(status_code=200)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="prompt",
        output_text='{"name": "test"}',
        duration_ms=10,
        output_schema=schema,
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert trace["outputSchema"] == schema
    client.close()


@patch("cert.client.requests.post")
def test_agentic_trace_includes_tool_calls(mock_post):
    """Agentic trace includes tool_calls field."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "search", "input": {"q": "test"}, "output": "results"}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        type="agentic",
        provider="test",
        model="m",
        input_text="task",
        output_text="completed",
        duration_ms=10,
        tool_calls=tool_calls,
        goal_description="find information",
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert trace["toolCalls"] == tool_calls
    assert trace["goalDescription"] == "find information"
    assert trace["evalMode"] == "agentic"
    client.close()


@patch("cert.client.requests.post")
def test_optional_fields_excluded_when_none(mock_post):
    """Optional fields are not included when None."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        type="generation",
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )

    time.sleep(0.3)
    trace = mock_post.call_args.kwargs["json"]["traces"][0]

    assert "context" not in trace
    assert "outputSchema" not in trace
    assert "toolCalls" not in trace
    assert "goalDescription" not in trace
    client.close()


# =============================================================================
# Tool Calls Validation Tests
# =============================================================================


def test_tool_calls_requires_name():
    """tool_calls entries must have 'name' field."""
    client = CertClient(api_key="test_key")

    with pytest.raises(ValueError, match="missing required 'name' field"):
        client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[{"input": {}}],
        )
    client.close()


def test_tool_calls_name_must_be_string():
    """tool_calls name must be a string."""
    client = CertClient(api_key="test_key")

    with pytest.raises(ValueError, match="name must be a string"):
        client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[{"name": 123}],
        )
    client.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


@patch("cert.client.requests.post")
def test_http_error_increments_failed(mock_post):
    """HTTP errors increment failed counter, don't raise."""
    mock_post.return_value = Mock(status_code=500, text="Internal error")

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.3)

    stats = client.stats()
    assert stats["traces_failed"] >= 1
    client.close()


@patch("cert.client.requests.post")
def test_timeout_increments_failed(mock_post):
    """Timeouts increment failed counter, don't raise."""
    import requests
    mock_post.side_effect = requests.exceptions.Timeout()

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.3)

    stats = client.stats()
    assert stats["traces_failed"] >= 1
    client.close()


@patch("cert.client.requests.post")
def test_exception_increments_failed(mock_post):
    """General exceptions increment failed counter, don't crash."""
    mock_post.side_effect = Exception("Network error")

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.3)

    stats = client.stats()
    assert stats["traces_failed"] >= 1
    client.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


@patch("cert.client.requests.post")
def test_context_manager(mock_post):
    """Context manager flushes and closes."""
    mock_post.return_value = Mock(status_code=200)

    with CertClient(api_key="test_key") as client:
        client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    # Should have auto-flushed
    assert mock_post.call_count >= 1


# =============================================================================
# Stats Tests
# =============================================================================


def test_stats_returns_all_counters():
    """stats() returns all expected counters."""
    client = CertClient(api_key="test_key")

    stats = client.stats()

    assert "traces_sent" in stats
    assert "traces_failed" in stats
    assert "traces_dropped" in stats
    assert "traces_queued" in stats
    client.close()


@patch("cert.client.requests.post")
def test_stats_tracks_sent(mock_post):
    """stats() accurately tracks sent traces."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.5)

    stats = client.stats()
    assert stats["traces_sent"] == 2
    client.close()


# =============================================================================
# API Key Header Tests
# =============================================================================


@patch("cert.client.requests.post")
def test_api_key_header(mock_post):
    """API key is sent in X-API-Key header."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="my_secret_key", batch_size=1)
    client.trace(type="generation", provider="p", model="m", input_text="i", output_text="o", duration_ms=10)

    time.sleep(0.3)

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "my_secret_key"
    client.close()
