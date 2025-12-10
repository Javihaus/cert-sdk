"""Tests for CERT client."""

import time
from unittest.mock import Mock, patch

import pytest

from cert import CertClient, EvalMode, ToolCall
from cert.client import _validate_tool_calls


def test_client_initialization():
    """Test client can be created."""
    client = CertClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert "cert-framework.com" in client.endpoint
    client.close()


def test_trace_queues_data():
    """Test that trace() queues data without blocking."""
    client = CertClient(api_key="test_key")

    # Should return immediately (non-blocking)
    start = time.time()
    client.trace(
        provider="test",
        model="test-model",
        input_text="input",
        output_text="output",
        duration_ms=100.0,
    )
    elapsed = time.time() - start

    # Should be < 10ms (non-blocking)
    assert elapsed < 0.01

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
# Eval Mode Tests
# ============================================================================


@patch("cert.client.requests.post")
def test_explicit_rag_mode(mock_post):
    """Test explicit RAG mode with context."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="rag",
        context="some retrieved context",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "rag"
    assert trace["context"] == "some retrieved context"
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_rag_mode(mock_post):
    """Test auto-detection resolves to RAG when context is provided."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
        context="some context",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "rag"
    client.close()


@patch("cert.client.requests.post")
def test_explicit_generation_mode(mock_post):
    """Test explicit generation mode."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
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

    assert trace["evalMode"] == "generation"
    assert "context" not in trace
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_generation_mode(mock_post):
    """Test auto-detection resolves to generation when no context or tool_calls."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "generation"
    client.close()


@patch("cert.client.requests.post")
def test_generation_mode_with_output_schema(mock_post):
    """Test generation mode with output_schema."""
    mock_post.return_value = Mock(status_code=200)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text='{"name": "test"}',
        duration_ms=100,
        eval_mode="generation",
        output_schema=schema,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "generation"
    assert trace["outputSchema"] == schema
    client.close()


@patch("cert.client.requests.post")
def test_explicit_agentic_mode(mock_post):
    """Test explicit agentic mode with tool_calls."""
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
        eval_mode="agentic",
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "agentic"
    assert trace["toolCalls"] == tool_calls
    client.close()


@patch("cert.client.requests.post")
def test_auto_detect_agentic_mode(mock_post):
    """Test auto-detection resolves to agentic when tool_calls provided."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "get_weather", "input": {"city": "NYC"}}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "agentic"
    client.close()


@patch("cert.client.requests.post")
def test_agentic_with_goal_description(mock_post):
    """Test agentic mode with goal_description."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "search", "input": {}}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="agentic",
        tool_calls=tool_calls,
        goal_description="Find and summarize information",
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "agentic"
    assert trace["goalDescription"] == "Find and summarize information"
    client.close()


# ============================================================================
# Edge Cases
# ============================================================================


@patch("cert.client.requests.post")
def test_empty_tool_calls_resolves_to_generation(mock_post):
    """Test empty tool_calls list resolves to generation, not agentic."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
        tool_calls=[],  # Empty list
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "generation"
    client.close()


@patch("cert.client.requests.post")
def test_whitespace_context_resolves_to_generation(mock_post):
    """Test whitespace-only context resolves to generation, not rag."""
    mock_post.return_value = Mock(status_code=200)

    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
        context="   ",  # Whitespace only
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    assert trace["evalMode"] == "generation"
    client.close()


@patch("cert.client.requests.post")
def test_tool_calls_takes_precedence_over_context(mock_post):
    """Test tool_calls takes precedence over context in auto mode."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [{"name": "retrieve", "input": {}}]
    client = CertClient(api_key="test_key", batch_size=1)
    client.trace(
        provider="test",
        model="m",
        input_text="question",
        output_text="answer",
        duration_ms=100,
        eval_mode="auto",
        tool_calls=tool_calls,
        context="some context",  # Both provided
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # tool_calls wins over context
    assert trace["evalMode"] == "agentic"
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
    assert "context" not in trace
    assert "outputSchema" not in trace
    assert "toolCalls" not in trace
    assert "goalDescription" not in trace

    # These required fields should be present
    assert "evalMode" in trace
    assert trace["evalMode"] == "generation"

    client.close()


# ============================================================================
# Automatic Context Extraction Tests
# ============================================================================


def test_extract_context_from_tool_calls_basic():
    """Test basic context extraction from tool calls."""
    from cert.client import extract_context_from_tool_calls
    
    tool_calls = [
        {"name": "search", "output": {"results": ["doc1", "doc2"]}},
        {"name": "calculate", "output": 42},
    ]
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "[search]:" in context
    assert '["doc1", "doc2"]' in context
    assert "[calculate]: 42" in context


def test_extract_context_from_tool_calls_with_error():
    """Test context extraction includes errors."""
    from cert.client import extract_context_from_tool_calls
    
    tool_calls = [
        {"name": "api_call", "error": "Connection timeout"},
        {"name": "search", "output": "results"},
    ]
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "[api_call] ERROR: Connection timeout" in context
    assert "[search]: results" in context


def test_extract_context_from_tool_calls_empty():
    """Test context extraction with empty list."""
    from cert.client import extract_context_from_tool_calls
    
    context = extract_context_from_tool_calls([])
    
    assert context == ""


def test_extract_context_skips_none_output():
    """Test context extraction skips tools with no output or error."""
    from cert.client import extract_context_from_tool_calls
    
    tool_calls = [
        {"name": "pending_tool"},  # No output or error
        {"name": "completed", "output": "done"},
    ]
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "pending_tool" not in context
    assert "[completed]: done" in context


@patch("cert.client.requests.post")
def test_agentic_auto_extracts_context(mock_post):
    """Test that agentic mode automatically extracts context from tool_calls."""
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
        eval_mode="agentic",
        tool_calls=tool_calls,
        # Note: NO context provided - should be auto-extracted!
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Context should be auto-extracted
    assert "context" in trace
    assert "[weather]:" in trace["context"]
    assert '"temp": 72' in trace["context"]
    assert trace["evalMode"] == "agentic"
    
    client.close()


@patch("cert.client.requests.post")
def test_agentic_explicit_context_takes_precedence(mock_post):
    """Test that explicit context overrides auto-extraction."""
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
        eval_mode="agentic",
        tool_calls=tool_calls,
        context="EXPLICIT CONTEXT",  # Should take precedence
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Explicit context should be used, not auto-extracted
    assert trace["context"] == "EXPLICIT CONTEXT"
    assert "auto-extracted" not in trace["context"]
    
    client.close()


@patch("cert.client.requests.post")
def test_auto_extract_disabled(mock_post):
    """Test that auto_extract_context=False disables extraction."""
    mock_post.return_value = Mock(status_code=200)

    tool_calls = [
        {"name": "search", "output": "results"},
    ]
    
    client = CertClient(
        api_key="test_key",
        batch_size=1,
        auto_extract_context=False,  # Disable auto-extraction
    )
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

    # Context should NOT be present
    assert "context" not in trace
    
    client.close()


@patch("cert.client.requests.post")
def test_auto_mode_with_tools_extracts_context(mock_post):
    """Test that auto mode + tools auto-extracts context."""
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
        eval_mode="auto",  # Auto mode
        tool_calls=tool_calls,
    )

    time.sleep(0.3)
    call_args = mock_post.call_args
    trace = call_args.kwargs["json"]["traces"][0]

    # Should resolve to agentic and auto-extract context
    assert trace["evalMode"] == "agentic"
    assert "context" in trace
    assert "[calc]: 100" in trace["context"]
    
    client.close()


def test_extract_context_complex_json():
    """Test context extraction handles complex nested JSON."""
    from cert.client import extract_context_from_tool_calls
    
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
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "[api_response]:" in context
    assert "Alice" in context
    assert "Bob" in context
    assert '"total": 2' in context


def test_extract_context_string_output():
    """Test context extraction handles string outputs."""
    from cert.client import extract_context_from_tool_calls
    
    tool_calls = [
        {"name": "read_file", "output": "File content here\nWith multiple lines"},
    ]
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "[read_file]: File content here\nWith multiple lines" in context


def test_extract_context_numeric_output():
    """Test context extraction handles numeric outputs."""
    from cert.client import extract_context_from_tool_calls
    
    tool_calls = [
        {"name": "calculate", "output": 3.14159},
        {"name": "count", "output": 42},
    ]
    
    context = extract_context_from_tool_calls(tool_calls)
    
    assert "[calculate]: 3.14159" in context
    assert "[count]: 42" in context
