"""Tests for CERT client."""

import time
import warnings
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from cert import (
    CertClient,
    TraceContext,
    EvalMode,
    EvaluationMode,
    ContextSource,
    SpanKind,
    TraceStatus,
    ToolCall,
)
from cert.client import _validate_tool_calls


def test_client_initialization():
    """Test client can be created."""
    client = CertClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.project == "default"
    assert "cert-framework.com" in client.endpoint
    client.close()


def test_client_initialization_with_project():
    """Test client can be created with custom project."""
    client = CertClient(api_key="test_key", project="my-project")
    assert client.project == "my-project"
    client.close()


def test_trace_queues_data():
    """Test that trace() queues data without blocking."""
    client = CertClient(api_key="test_key")

    # Should return immediately (non-blocking)
    start = time.time()
    trace_id = client.trace(
        provider="test",
        model="test-model",
        input_text="input",
        output_text="output",
        duration_ms=100.0,
    )
    elapsed = time.time() - start

    # Should be < 10ms (non-blocking)
    assert elapsed < 0.01

    # Should return a trace_id
    assert trace_id is not None
    assert len(trace_id) == 36  # UUID format

    # Should have queued one trace
    stats = client.get_stats()
    assert stats["traces_queued"] >= 1

    client.close()


@patch("cert.client.requests.post")
def test_batch_sending(mock_post):
    """Test that batches are sent to API."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(
        api_key="test_key",
        batch_size=2,
        flush_interval=0.1,
    )

    # Send 2 traces (fills batch)
    client.trace(
        provider="test", model="m1",
        input_text="i1", output_text="o1", duration_ms=10
    )
    client.trace(
        provider="test", model="m2",
        input_text="i2", output_text="o2", duration_ms=20
    )

    # Wait for batch to send
    time.sleep(0.5)

    # Should have called API once with 2 traces
    assert mock_post.call_count >= 1
    call_args = mock_post.call_args
    sent_data = call_args.kwargs["json"]
    assert len(sent_data["traces"]) >= 2

    client.close()


@patch("cert.client.requests.post")
def test_close_sends_remaining(mock_post):
    """Test close() sends remaining traces in batch."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(
        api_key="test_key",
        batch_size=100,  # Won't fill batch
        flush_interval=999,  # Won't auto-flush
    )

    client.trace(
        provider="test", model="m",
        input_text="i", output_text="o", duration_ms=10
    )

    # Close forces remaining traces to be sent
    client.close()

    # Should have sent trace on shutdown
    assert mock_post.call_count >= 1


@patch("cert.client.requests.post")
def test_error_handling(mock_post):
    """Test that errors don't crash."""
    # Simulate API error
    mock_post.side_effect = Exception("API down")

    client = CertClient(api_key="test_key", batch_size=1)

    # Should not raise
    client.trace(
        provider="test", model="m",
        input_text="i", output_text="o", duration_ms=10
    )

    time.sleep(0.5)

    # Should have incremented failed count
    stats = client.get_stats()
    assert stats["traces_failed"] >= 1

    client.close()


def test_context_manager():
    """Test context manager support."""
    with CertClient(api_key="test_key") as client:
        client.trace(
            provider="test", model="m",
            input_text="i", output_text="o", duration_ms=10
        )
    # Should auto-close without error


# ============================================================================
# New v0.3.0 Payload Tests
# ============================================================================


@patch("cert.client.requests.post")
def test_trace_returns_trace_id(mock_post):
    """Test trace() returns trace_id."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    trace_id = client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    assert trace_id is not None
    assert len(trace_id) == 36  # UUID format
    client.close()


@patch("cert.client.requests.post")
def test_auto_generates_span_id(mock_post):
    """Test that span_id is auto-generated."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["span_id"].startswith("span-")
    client.close()


@patch("cert.client.requests.post")
def test_total_tokens_computed(mock_post):
    """Test that total_tokens is computed from prompt + completion."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        prompt_tokens=100,
        completion_tokens=50
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["total_tokens"] == 150
    client.close()


@patch("cert.client.requests.post")
def test_provider_maps_to_vendor(mock_post):
    """Test that provider is mapped to vendor in payload."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="anthropic",
        model="claude-sonnet",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["llm_vendor"] == "anthropic"
    assert "provider" not in trace
    assert "vendor" not in trace
    client.close()


@patch("cert.client.requests.post")
def test_input_output_field_names(mock_post):
    """Test that input/output map to inputText/outputText."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello input",
        output_text="Hi output!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["input_text"] == "Hello input"
    assert trace["output_text"] == "Hi output!"
    client.close()


@patch("cert.client.requests.post")
def test_timing_fields_included(mock_post):
    """Test that timing fields are included in payload."""
    mock_post.return_value = Mock(status_code=200)

    start = datetime.now(timezone.utc)
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        start_time=start
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert "start_time" in trace
    assert "end_time" in trace
    assert trace["start_time"] == start.isoformat()
    client.close()


@patch("cert.client.requests.post")
def test_status_and_error(mock_post):
    """Test status and error_message fields."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="",
        duration_ms=100,
        status="error",
        error_message="API timeout"
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["status"] == "error"
    assert trace["error_message"] == "API timeout"
    client.close()


@patch("cert.client.requests.post")
def test_operation_name_auto_generated(mock_post):
    """Test operation name is auto-generated from provider.model."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["name"] == "openai.gpt-4o"
    client.close()


@patch("cert.client.requests.post")
def test_custom_operation_name(mock_post):
    """Test custom operation name can be provided."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        name="my_custom_operation"
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["name"] == "my_custom_operation"
    client.close()


@patch("cert.client.requests.post")
def test_source_field(mock_post):
    """Test source field is set to cert-sdk."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["source"] == "cert-sdk"
    client.close()


@patch("cert.client.requests.post")
def test_project_field(mock_post):
    """Test project field is included."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", project="my-project", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["project_name"] == "my-project"
    client.close()


@patch("cert.client.requests.post")
def test_span_kind_field(mock_post):
    """Test kind field defaults to CLIENT."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["kind"] == "CLIENT"
    client.close()


@patch("cert.client.requests.post")
def test_custom_span_kind(mock_post):
    """Test custom span kind."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        kind="SERVER"
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["kind"] == "SERVER"
    client.close()


@patch("cert.client.requests.post")
def test_parent_span_id(mock_post):
    """Test parent_span_id is included when provided."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        parent_span_id="parent-span-123"
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["parent_span_id"] == "parent-span-123"
    client.close()


@patch("cert.client.requests.post")
def test_trace_id_can_be_provided(mock_post):
    """Test custom trace_id can be provided."""
    mock_post.return_value = Mock(status_code=200)

    custom_trace_id = "custom-trace-id-123"
    client = CertClient(api_key="test_key", batch_size=1)
    returned_id = client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100,
        trace_id=custom_trace_id
    )

    assert returned_id == custom_trace_id

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["trace_id"] == custom_trace_id
    client.close()



@patch("cert.client.requests.post")
def test_grounded_mode_with_knowledge_base(mock_post):
    """Test grounded mode with knowledge_base."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        knowledge_base="some retrieved context",
        context_source="retrieval",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["knowledge_base"] == "some retrieved context"
    assert trace["context_source"] == "retrieval"
    # Also check backwards compatibility field
    assert trace["context"] == "some retrieved context"
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_grounded_mode(mock_post):
    """Test auto-detection resolves to grounded when knowledge_base is provided."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="auto",
        knowledge_base="some context",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    client.close()


@patch("cert.client.requests.post")
def test_ungrounded_mode(mock_post):
    """Test ungrounded mode."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="ungrounded",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "ungrounded"
    assert "knowledge_base" not in trace
    assert "context" not in trace
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_ungrounded_mode(mock_post):
    """Test auto-detection resolves to ungrounded when no knowledge_base or tool_calls."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="auto",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "ungrounded"
    client.close()


@patch("cert.client.requests.post")
def test_ungrounded_mode_with_output_schema(mock_post):
    """Test ungrounded mode with output_schema."""
    mock_post.return_value = Mock(status_code=200)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text='{"name": "test"}',
        duration_ms=100,
        evaluation_mode="ungrounded",
        output_schema=schema,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "ungrounded"
    assert trace["output_schema"] == schema
    client.close()


@patch("cert.client.requests.post")
def test_grounded_mode_with_tool_calls(mock_post):
    """Test grounded mode with tool_calls."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "search", "input": {"query": "test"}, "output": "results"}
    ]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
        context_source="tools",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["tool_calls"] == tool_calls
    assert trace["context_source"] == "tools"
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_grounded_with_tool_calls(mock_post):
    """Test auto-detection resolves to grounded when tool_calls have outputs."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "get_weather", "input": {"city": "NYC"}, "output": {"temp": 72}}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="auto",
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["context_source"] == "tools"
    # Knowledge should be auto-extracted
    assert "[get_weather]:" in trace["knowledge_base"]
    client.close()


@patch("cert.client.requests.post")
def test_grounded_with_goal_description(mock_post):
    """Test grounded mode with goal_description."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "search", "input": {}, "output": "data"}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
        goal_description="Find and summarize information",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["goal_description"] == "Find and summarize information"
    client.close()


# ============================================================================
# Legacy API Tests (Backwards Compatibility)
# ============================================================================


@patch("cert.client.requests.post")
def test_legacy_rag_mode(mock_post):
    """Test legacy eval_mode='rag' maps to evaluation_mode='grounded'."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client.trace(
            provider="test",
            model="m",
            input_text="question",
            output_text="answer",
            duration_ms=100,
            eval_mode="rag",
            context="some retrieved context",
        )
        assert any("eval_mode is deprecated" in str(warning.message) for warning in w)
        assert any("context is deprecated" in str(warning.message) for warning in w)

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Should be mapped to new values
    assert trace["evaluation_mode"] == "grounded"
    assert trace["knowledge_base"] == "some retrieved context"
    # Also check backwards compat field
    assert trace["context"] == "some retrieved context"
    client.close()


@patch("cert.client.requests.post")
def test_legacy_generation_mode(mock_post):
    """Test legacy eval_mode='generation' maps to evaluation_mode='ungrounded'."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        client.trace(
            provider="test",
            model="m",
            input_text="question",
            output_text="answer",
            duration_ms=100,
            eval_mode="generation",
        )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "ungrounded"
    client.close()


@patch("cert.client.requests.post")
def test_legacy_agentic_mode(mock_post):
    """Test legacy eval_mode='agentic' maps to evaluation_mode='grounded'."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "search", "input": {}, "output": "results"}]
    client = CertClient(api_key="test_key", batch_size=1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        client.trace(
            provider="test",
            model="m",
            input_text="question",
            output_text="answer",
            duration_ms=100,
            eval_mode="agentic",
            tool_calls=tool_calls,
        )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["context_source"] == "tools"
    # Legacy eval_mode should be included for backward compatibility
    assert trace["eval_mode"] == "agentic"
    client.close()


# ============================================================================
# Edge Cases
# ============================================================================


@patch("cert.client.requests.post")
def test_empty_tool_calls_resolves_to_ungrounded(mock_post):
    """Test empty tool_calls list resolves to ungrounded, not grounded."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="auto",
        tool_calls=[],  # Empty list
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "ungrounded"
    client.close()


@patch("cert.client.requests.post")
def test_whitespace_knowledge_base_resolves_to_grounded(mock_post):
    """Test whitespace knowledge_base still counts as grounded."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="auto",
        knowledge_base="   ",  # Whitespace only (but still a value)
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Note: whitespace is still treated as a knowledge_base
    assert trace["evaluation_mode"] == "grounded"
    client.close()


@patch("cert.client.requests.post")
def test_explicit_knowledge_base_takes_precedence_over_tool_calls(mock_post):
    """Test that explicit knowledge_base is used when both provided."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "retrieve", "input": {}, "output": "auto extracted"}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
        knowledge_base="EXPLICIT KNOWLEDGE",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Explicit knowledge base should be used
    assert trace["knowledge_base"] == "EXPLICIT KNOWLEDGE"
    assert "auto extracted" not in trace["knowledge_base"]
    client.close()


# ============================================================================
# Validation Tests
# ============================================================================


def test_validate_tool_calls_valid():
    """Test valid tool_calls passes validation."""
    tool_calls = [
        {"name": "search", "input": {"query": "test"}},
        {"name": "calculator", "input": {"expression": "1+1"}, "output": 2},
    ]
    # Should not raise
    _validate_tool_calls(tool_calls)


def test_validate_tool_calls_missing_name():
    """Test tool_calls without name raises ValueError."""
    tool_calls = [{"input": {"query": "test"}}]  # Missing 'name'

    with pytest.raises(ValueError, match="missing required 'name' field"):
        _validate_tool_calls(tool_calls)


def test_validate_tool_calls_non_string_name():
    """Test tool_calls with non-string name raises ValueError."""
    tool_calls = [{"name": 123, "input": {}}]  # name is not a string

    with pytest.raises(ValueError, match="name must be a string"):
        _validate_tool_calls(tool_calls)


def test_trace_validates_tool_calls():
    """Test that trace() validates tool_calls before queuing."""
    client = CertClient(api_key="test_key")

    with pytest.raises(ValueError, match="missing required 'name' field"):
        client.trace(
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[{"input": {}}],  # Missing 'name'
        )

    client.close()


@patch("cert.client.requests.post")
def test_optional_fields_not_included_when_none(mock_post):
    """Test that optional fields are not included when None."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="i",
        output_text="o",
        duration_ms=10,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # These optional fields should not be present
    assert "knowledge_base" not in trace
    assert "context" not in trace
    assert "output_schema" not in trace
    assert "tool_calls" not in trace
    assert "goal_description" not in trace
    assert "parent_span_id" not in trace
    assert "error_message" not in trace
    assert "context_source" not in trace

    # These required fields should be present
    assert "evaluation_mode" in trace
    assert trace["evaluation_mode"] == "ungrounded"

    client.close()


# ============================================================================
# Automatic Knowledge Extraction Tests
# ============================================================================


def test_extract_knowledge_from_tool_calls_basic():
    """Test basic knowledge extraction from tool calls."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {"name": "search", "output": {"results": ["doc1", "doc2"]}},
        {"name": "calculate", "output": 42},
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "[search]:" in knowledge
    assert '["doc1", "doc2"]' in knowledge
    assert "[calculate]: 42" in knowledge


def test_extract_knowledge_from_tool_calls_with_error():
    """Test knowledge extraction includes errors."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {"name": "api_call", "error": "Connection timeout"},
        {"name": "search", "output": "results"},
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "[api_call] ERROR: Connection timeout" in knowledge
    assert "[search]: results" in knowledge


def test_extract_knowledge_from_tool_calls_empty():
    """Test knowledge extraction with empty list."""
    from cert.client import extract_knowledge_from_tool_calls

    knowledge = extract_knowledge_from_tool_calls([])

    assert knowledge == ""


def test_extract_knowledge_skips_none_output():
    """Test knowledge extraction skips tools with no output or error."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {"name": "pending_tool"},  # No output or error
        {"name": "completed", "output": "done"},
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "pending_tool" not in knowledge
    assert "[completed]: done" in knowledge


@patch("cert.client.requests.post")
def test_grounded_auto_extracts_knowledge(mock_post):
    """Test that grounded mode automatically extracts knowledge from tool_calls."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "weather", "input": {"city": "NYC"}, "output": {"temp": 72, "condition": "sunny"}},
    ]

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="What's the weather?",
        output_text="It's 72°F and sunny",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
        # Note: NO knowledge_base provided - should be auto-extracted!
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Knowledge should be auto-extracted
    assert "knowledge_base" in trace
    assert "[weather]:" in trace["knowledge_base"]
    assert '"temp": 72' in trace["knowledge_base"]
    assert trace["evaluation_mode"] == "grounded"
    assert trace["context_source"] == "tools"

    client.close()


@patch("cert.client.requests.post")
def test_explicit_knowledge_base_takes_precedence(mock_post):
    """Test that explicit knowledge_base overrides auto-extraction."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "search", "output": "auto-extracted content"},
    ]

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
        knowledge_base="EXPLICIT KNOWLEDGE",  # Should take precedence
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Explicit knowledge should be used, not auto-extracted
    assert trace["knowledge_base"] == "EXPLICIT KNOWLEDGE"
    assert "auto-extracted" not in trace["knowledge_base"]

    client.close()


@patch("cert.client.requests.post")
def test_auto_extract_disabled(mock_post):
    """Test that auto_extract_knowledge=False disables extraction."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "search", "output": "results"},
    ]

    client = CertClient(
        api_key="test_key",
        batch_size=1,
        auto_extract_knowledge=False,  # Disable auto-extraction
    )
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        evaluation_mode="grounded",
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Knowledge should NOT be present
    assert "knowledge_base" not in trace

    client.close()


@patch("cert.client.requests.post")
def test_auto_mode_with_tools_extracts_knowledge(mock_post):
    """Test that auto mode + tools auto-extracts knowledge."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "calc", "output": 100},
    ]

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="calculate",
        output_text="100",
        duration_ms=100,
        evaluation_mode="auto",  # Auto mode
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Should resolve to grounded and auto-extract knowledge
    assert trace["evaluation_mode"] == "grounded"
    assert "knowledge_base" in trace
    assert "[calc]: 100" in trace["knowledge_base"]
    assert trace["context_source"] == "tools"

    client.close()


def test_extract_knowledge_complex_json():
    """Test knowledge extraction handles complex nested JSON."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {
            "name": "api_response",
            "output": {
                "data": {
                    "users": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25},
                    ],
                    "total": 2,
                },
                "status": "success",
            },
        },
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "[api_response]:" in knowledge
    assert "Alice" in knowledge
    assert "Bob" in knowledge
    assert '"total": 2' in knowledge


def test_extract_knowledge_string_output():
    """Test knowledge extraction handles string outputs."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {"name": "read_file", "output": "File content here\nWith multiple lines"},
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "[read_file]: File content here\nWith multiple lines" in knowledge


def test_extract_knowledge_numeric_output():
    """Test knowledge extraction handles numeric outputs."""
    from cert.client import extract_knowledge_from_tool_calls

    tool_calls = [
        {"name": "calculate", "output": 3.14159},
        {"name": "count", "output": 42},
    ]

    knowledge = extract_knowledge_from_tool_calls(tool_calls)

    assert "[calculate]: 3.14159" in knowledge
    assert "[count]: 42" in knowledge


# ============================================================================
# TraceContext Tests
# ============================================================================


@patch("cert.client.requests.post")
def test_trace_context_automatic_timing(mock_post):
    """Test TraceContext captures timing automatically."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)

    with TraceContext(client, provider="openai", model="gpt-4", input_text="test") as ctx:
        time.sleep(0.01)  # Small delay to ensure measurable duration
        ctx.set_output("response")
        ctx.set_tokens(10, 20)

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["duration_ms"] > 0
    assert "start_time" in trace
    assert "end_time" in trace
    assert trace["prompt_tokens"] == 10
    assert trace["completion_tokens"] == 20
    assert trace["output_text"] == "response"
    assert trace["status"] == "success"

    client.close()


@patch("cert.client.requests.post")
def test_trace_context_error_capture(mock_post):
    """Test TraceContext captures errors automatically."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)

    with pytest.raises(ValueError):
        with TraceContext(client, provider="openai", model="gpt-4", input_text="test") as ctx:
            raise ValueError("Test error")

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["status"] == "error"
    assert "Test error" in trace["error_message"]

    client.close()


@patch("cert.client.requests.post")
def test_trace_context_with_extra_kwargs(mock_post):
    """Test TraceContext passes extra kwargs to trace()."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    tool_calls = [{"name": "search", "output": "results"}]

    with TraceContext(
        client,
        provider="openai",
        model="gpt-4",
        input_text="test",
        evaluation_mode="grounded",
        tool_calls=tool_calls
    ) as ctx:
        ctx.set_output("response")

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evaluation_mode"] == "grounded"
    assert trace["tool_calls"] == tool_calls

    client.close()


@patch("cert.client.requests.post")
def test_trace_context_provides_trace_id(mock_post):
    """Test TraceContext provides trace_id and span_id."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)

    with TraceContext(client, provider="openai", model="gpt-4", input_text="test") as ctx:
        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 36
        assert ctx.span_id.startswith("span-")
        ctx.set_output("response")

    client.close()


@patch("cert.client.requests.post")
def test_trace_context_set_knowledge_base(mock_post):
    """Test TraceContext set_knowledge_base method."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)

    with TraceContext(client, provider="openai", model="gpt-4", input_text="test") as ctx:
        ctx.set_knowledge_base("Retrieved document content", source="retrieval")
        ctx.set_output("response")

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["knowledge_base"] == "Retrieved document content"
    assert trace["context_source"] == "retrieval"
    assert trace["evaluation_mode"] == "grounded"

    client.close()


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


@patch("cert.client.requests.post")
def test_minimal_trace_backward_compatible(mock_post):
    """Test minimal trace call still works."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)

    # This is how code would call trace() with minimal args
    client.trace(
        provider="openai",
        model="gpt-4",
        input_text="Hello",
        output_text="Hi!",
        duration_ms=100
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Should have auto-generated all required fields
    assert "trace_id" in trace
    assert "span_id" in trace
    assert "name" in trace
    assert trace["status"] == "success"
    assert trace["evaluation_mode"] == "ungrounded"

    client.close()


@patch("cert.client.requests.post")
def test_legacy_auto_extract_context_parameter(mock_post):
    """Test legacy auto_extract_context parameter with deprecation warning."""
    mock_post.return_value = Mock(status_code=200)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = CertClient(
            api_key="test_key",
            batch_size=1,
            auto_extract_context=False,  # DEPRECATED
        )
        assert any("auto_extract_context is deprecated" in str(warning.message) for warning in w)

    # Should still work
    assert client.auto_extract_knowledge is False
    client.close()


# ============================================================================
# Type Imports Tests
# ============================================================================


def test_type_imports():
    """Test that all types can be imported from cert module."""
    from cert import EvalMode, EvaluationMode, ContextSource, SpanKind, TraceStatus, ToolCall

    # Types should be available
    assert EvalMode is not None
    assert EvaluationMode is not None
    assert ContextSource is not None
    assert SpanKind is not None
    assert TraceStatus is not None
    assert ToolCall is not None


def test_version_import():
    """Test that version can be imported."""
    from cert import __version__

    assert __version__ == "1.0.0"


# Backwards compatibility test for extract_context_from_tool_calls
def test_extract_context_from_tool_calls_alias():
    """Test that extract_context_from_tool_calls is an alias for extract_knowledge_from_tool_calls."""
    from cert import extract_context_from_tool_calls, extract_knowledge_from_tool_calls

    tool_calls = [{"name": "test", "output": "result"}]

    assert extract_context_from_tool_calls(tool_calls) == extract_knowledge_from_tool_calls(tool_calls)
