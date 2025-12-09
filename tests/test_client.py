"""Comprehensive tests for CERT SDK client."""

import time
import uuid
from unittest.mock import Mock, patch

import pytest

from cert import CertClient, EvalMode, ToolCall, TraceType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a basic client for testing."""
    c = CertClient(api_key="test_key")
    yield c
    c.close()


@pytest.fixture
def client_with_project():
    """Create a client with custom project."""
    c = CertClient(api_key="test_key", project="test-project")
    yield c
    c.close()


@pytest.fixture
def fast_flush_client():
    """Create a client with short flush interval."""
    c = CertClient(api_key="test_key", batch_size=100, flush_interval=0.2)
    yield c
    c.close()


@pytest.fixture
def small_batch_client():
    """Create a client with small batch size."""
    c = CertClient(api_key="test_key", batch_size=2, flush_interval=999)
    yield c
    c.close()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestClientInitialization:
    """Tests for CertClient initialization."""

    def test_initialization_with_api_key(self):
        """Client initializes with required api_key."""
        client = CertClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.project == "default"
        assert CertClient.DEFAULT_URL in client.endpoint
        client.close()

    def test_initialization_with_project(self):
        """Client accepts project parameter."""
        client = CertClient(api_key="test_key", project="my-project")
        assert client.project == "my-project"
        client.close()

    def test_initialization_with_custom_url(self):
        """Client accepts custom dashboard URL."""
        client = CertClient(api_key="test_key", dashboard_url="https://custom.example.com")
        assert "custom.example.com" in client.endpoint
        client.close()

    def test_initialization_strips_trailing_slash(self):
        """Client strips trailing slash from URL."""
        client = CertClient(api_key="test_key", dashboard_url="https://example.com/")
        assert client.endpoint == "https://example.com/api/v1/traces"
        client.close()

    def test_initialization_with_custom_batch_size(self):
        """Client accepts custom batch_size."""
        client = CertClient(api_key="test_key", batch_size=50)
        assert client.batch_size == 50
        client.close()

    def test_initialization_with_custom_flush_interval(self):
        """Client accepts custom flush_interval."""
        client = CertClient(api_key="test_key", flush_interval=10.0)
        assert client.flush_interval == 10.0
        client.close()

    def test_initialization_with_custom_timeout(self):
        """Client accepts custom timeout."""
        client = CertClient(api_key="test_key", timeout=30.0)
        assert client.timeout == 30.0
        client.close()

    def test_requires_api_key(self):
        """Client raises ValueError if api_key is empty."""
        with pytest.raises(ValueError, match="api_key is required"):
            CertClient(api_key="")

    def test_requires_api_key_not_none(self):
        """Client raises appropriate error for None api_key."""
        with pytest.raises((ValueError, TypeError)):
            CertClient(api_key=None)  # type: ignore

    def test_repr(self, client_with_project):
        """Client has useful repr."""
        assert "test-project" in repr(client_with_project)

    def test_repr_format(self, client):
        """Client repr follows expected format."""
        assert repr(client) == "CertClient(project='default')"


# =============================================================================
# Trace Type Validation Tests
# =============================================================================


class TestTraceTypeValidation:
    """Tests for trace type validation."""

    def test_trace_requires_type(self, client):
        """trace() requires type parameter."""
        with pytest.raises(TypeError):
            client.trace(
                provider="test",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )

    def test_trace_validates_type(self, client):
        """trace() rejects invalid type values."""
        with pytest.raises(ValueError, match="type must be"):
            client.trace(
                type="invalid",  # type: ignore
                provider="test",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )

    def test_trace_accepts_rag_type(self, client):
        """trace() accepts 'rag' type."""
        trace_id = client.trace(
            type="rag",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        assert trace_id is not None

    def test_trace_accepts_generation_type(self, client):
        """trace() accepts 'generation' type."""
        trace_id = client.trace(
            type="generation",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        assert trace_id is not None

    def test_trace_accepts_agentic_type(self, client):
        """trace() accepts 'agentic' type."""
        trace_id = client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        assert trace_id is not None

    def test_all_valid_types(self, client):
        """All VALID_TYPES are accepted."""
        for trace_type in CertClient.VALID_TYPES:
            trace_id = client.trace(
                type=trace_type,
                provider="test",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )
            assert trace_id is not None


# =============================================================================
# Non-blocking Behavior Tests
# =============================================================================


class TestNonBlockingBehavior:
    """Tests for non-blocking trace behavior."""

    def test_trace_is_nonblocking(self, client):
        """trace() returns immediately without blocking."""
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

    def test_trace_returns_uuid(self, client):
        """trace() returns a valid UUID string."""
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
        # Validate it's a proper UUID
        uuid.UUID(trace_id)

    def test_trace_returns_unique_ids(self, client):
        """Each trace() call returns a unique ID."""
        ids = set()
        for _ in range(100):
            trace_id = client.trace(
                type="generation",
                provider="test",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )
            ids.add(trace_id)
        assert len(ids) == 100

    def test_trace_queues_immediately(self, client):
        """trace() queues trace immediately."""
        client.trace(
            type="generation",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        stats = client.stats()
        assert stats["traces_queued"] >= 1


# =============================================================================
# Batch Sending Tests
# =============================================================================


class TestBatchSending:
    """Tests for batch sending behavior."""

    @patch("cert.client.requests.post")
    def test_batch_sends_on_size(self, mock_post):
        """Batch sends when batch_size is reached."""
        mock_post.return_value = Mock(status_code=200)

        # Create client after mock is applied
        client = CertClient(api_key="test_key", batch_size=2, flush_interval=999)

        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        # Use flush to ensure traces are sent
        client.flush(timeout=2.0)

        assert mock_post.call_count >= 1
        # Check that at least 2 traces were sent in total
        total_traces = sum(len(call.kwargs["json"]["traces"]) for call in mock_post.call_args_list)
        assert total_traces >= 2
        client.close()

    @patch("cert.client.requests.post")
    def test_batch_sends_on_interval(self, mock_post, fast_flush_client):
        """Batch sends when flush_interval elapses."""
        mock_post.return_value = Mock(status_code=200)

        fast_flush_client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.5)

        assert mock_post.call_count >= 1

    @patch("cert.client.requests.post")
    def test_close_sends_all_traces(self, mock_post):
        """close() sends all pending traces including worker's batch."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        # close() signals worker to flush its batch and exit
        client.close()

        # All traces should be sent
        stats = client.stats()
        assert stats["traces_sent"] == 2
        assert stats["traces_queued"] == 0

    @patch("cert.client.requests.post")
    def test_close_flushes(self, mock_post):
        """close() flushes remaining traces."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        client.close()

        assert mock_post.call_count >= 1


# =============================================================================
# Flush Method Tests
# =============================================================================


class TestFlushMethod:
    """Tests for the flush() method."""

    @patch("cert.client.requests.post")
    def test_flush_sends_pending_traces(self, mock_post):
        """flush() sends all pending traces."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        sent_count = client.flush()

        assert sent_count == 2
        assert mock_post.call_count >= 1
        client.close()

    @patch("cert.client.requests.post")
    def test_flush_returns_count(self, mock_post):
        """flush() returns number of traces sent."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=100, flush_interval=999)

        for _ in range(5):
            client.trace(
                type="generation",
                provider="p",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )

        sent_count = client.flush()

        assert sent_count == 5
        client.close()

    def test_flush_after_close_returns_zero(self):
        """flush() returns 0 after client is closed."""
        client = CertClient(api_key="test_key")
        client.close()

        sent_count = client.flush()
        assert sent_count == 0


# =============================================================================
# Payload Structure Tests
# =============================================================================


class TestPayloadStructure:
    """Tests for trace payload structure."""

    @patch("cert.client.requests.post")
    def test_trace_includes_type_field(self, mock_post):
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
    def test_trace_includes_project_name(self, mock_post):
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
    def test_rag_trace_includes_context(self, mock_post):
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
    def test_generation_trace_includes_schema(self, mock_post):
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
    def test_agentic_trace_includes_tool_calls(self, mock_post):
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
    def test_trace_includes_token_counts(self, mock_post):
        """Trace includes token count fields when provided."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            prompt_tokens=100,
            completion_tokens=50,
        )

        time.sleep(0.3)
        trace = mock_post.call_args.kwargs["json"]["traces"][0]

        assert trace["promptTokens"] == 100
        assert trace["completionTokens"] == 50
        client.close()

    @patch("cert.client.requests.post")
    def test_trace_includes_metadata(self, mock_post):
        """Trace includes metadata field when provided."""
        mock_post.return_value = Mock(status_code=200)

        metadata = {"user_id": "123", "session_id": "abc"}
        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            metadata=metadata,
        )

        time.sleep(0.3)
        trace = mock_post.call_args.kwargs["json"]["traces"][0]

        assert trace["metadata"] == metadata
        client.close()

    @patch("cert.client.requests.post")
    def test_optional_fields_excluded_when_none(self, mock_post):
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
        assert "promptTokens" not in trace
        assert "completionTokens" not in trace
        assert "metadata" not in trace
        client.close()

    @patch("cert.client.requests.post")
    def test_trace_includes_required_fields(self, mock_post):
        """Trace includes all required fields."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="openai",
            model="gpt-4",
            input_text="test input",
            output_text="test output",
            duration_ms=123.45,
        )

        time.sleep(0.3)
        trace = mock_post.call_args.kwargs["json"]["traces"][0]

        assert "id" in trace
        assert trace["type"] == "generation"
        assert trace["provider"] == "openai"
        assert trace["model"] == "gpt-4"
        assert trace["input"] == "test input"
        assert trace["output"] == "test output"
        assert trace["durationMs"] == 123.45
        assert "timestamp" in trace
        assert trace["evalMode"] == "generation"
        assert "projectName" in trace
        client.close()


# =============================================================================
# Tool Calls Validation Tests
# =============================================================================


class TestToolCallsValidation:
    """Tests for tool_calls validation."""

    def test_tool_calls_requires_name(self, client):
        """tool_calls entries must have 'name' field."""
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

    def test_tool_calls_name_must_be_string(self, client):
        """tool_calls name must be a string."""
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

    def test_tool_calls_validates_all_entries(self, client):
        """tool_calls validates all entries in list."""
        with pytest.raises(ValueError, match="tool_calls\\[1\\]"):
            client.trace(
                type="agentic",
                provider="test",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
                tool_calls=[{"name": "valid"}, {"input": {}}],
            )

    def test_tool_calls_accepts_valid_structure(self, client):
        """tool_calls accepts valid structure."""
        trace_id = client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[
                {"name": "tool1", "input": {"a": 1}, "output": "result1"},
                {"name": "tool2", "input": {"b": 2}, "output": "result2"},
            ],
        )
        assert trace_id is not None

    def test_tool_calls_accepts_minimal_structure(self, client):
        """tool_calls accepts minimal structure with just name."""
        trace_id = client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[{"name": "minimal_tool"}],
        )
        assert trace_id is not None

    def test_empty_tool_calls_list_accepted(self, client):
        """Empty tool_calls list is accepted."""
        trace_id = client.trace(
            type="agentic",
            provider="test",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
            tool_calls=[],
        )
        assert trace_id is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @patch("cert.client.requests.post")
    def test_http_error_increments_failed(self, mock_post):
        """HTTP errors increment failed counter, don't raise."""
        mock_post.return_value = Mock(status_code=500, text="Internal error")

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        stats = client.stats()
        assert stats["traces_failed"] >= 1
        client.close()

    @patch("cert.client.requests.post")
    def test_timeout_increments_failed(self, mock_post):
        """Timeouts increment failed counter, don't raise."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        stats = client.stats()
        assert stats["traces_failed"] >= 1
        client.close()

    @patch("cert.client.requests.post")
    def test_exception_increments_failed(self, mock_post):
        """General exceptions increment failed counter, don't crash."""
        mock_post.side_effect = Exception("Network error")

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        stats = client.stats()
        assert stats["traces_failed"] >= 1
        client.close()

    @patch("cert.client.requests.post")
    def test_http_4xx_increments_failed(self, mock_post):
        """HTTP 4xx errors increment failed counter."""
        mock_post.return_value = Mock(status_code=401, text="Unauthorized")

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        stats = client.stats()
        assert stats["traces_failed"] >= 1
        client.close()


# =============================================================================
# Queue Overflow Tests
# =============================================================================


class TestQueueOverflow:
    """Tests for queue overflow handling."""

    @patch("cert.client.requests.post")
    def test_queue_overflow_drops_traces(self, mock_post):
        """Queue overflow drops traces and increments counter."""

        # Make the HTTP request slow to block the worker
        def slow_post(*args, **kwargs):
            time.sleep(0.5)
            return Mock(status_code=200)

        mock_post.side_effect = slow_post

        # Create client with very small queue and batch_size=1 to trigger sending
        client = CertClient(
            api_key="test_key",
            max_queue_size=3,
            batch_size=1,
            flush_interval=999,
        )

        # Rapidly fill queue while worker is blocked on first send
        for _ in range(10):
            client.trace(
                type="generation",
                provider="p",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )

        stats = client.stats()

        # Some traces should be dropped since queue is only size 3
        assert stats["traces_dropped"] > 0
        client.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager support."""

    @patch("cert.client.requests.post")
    def test_context_manager_flushes_and_closes(self, mock_post):
        """Context manager flushes and closes."""
        mock_post.return_value = Mock(status_code=200)

        with CertClient(api_key="test_key") as client:
            client.trace(
                type="generation",
                provider="p",
                model="m",
                input_text="i",
                output_text="o",
                duration_ms=10,
            )

        # Should have auto-flushed
        assert mock_post.call_count >= 1

    def test_context_manager_returns_client(self):
        """Context manager returns client instance."""
        with CertClient(api_key="test_key") as client:
            assert isinstance(client, CertClient)

    @patch("cert.client.requests.post")
    def test_context_manager_handles_exception(self, mock_post):
        """Context manager closes even on exception."""
        mock_post.return_value = Mock(status_code=200)

        with pytest.raises(RuntimeError):
            with CertClient(api_key="test_key") as client:
                client.trace(
                    type="generation",
                    provider="p",
                    model="m",
                    input_text="i",
                    output_text="o",
                    duration_ms=10,
                )
                raise RuntimeError("Test error")

        # Should still have attempted to flush
        time.sleep(0.2)


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for stats() method."""

    def test_stats_returns_all_counters(self, client):
        """stats() returns all expected counters."""
        stats = client.stats()

        assert "traces_sent" in stats
        assert "traces_failed" in stats
        assert "traces_dropped" in stats
        assert "traces_queued" in stats

    def test_stats_initial_values(self, client):
        """stats() returns zero for all counters initially."""
        stats = client.stats()

        assert stats["traces_sent"] == 0
        assert stats["traces_failed"] == 0
        assert stats["traces_dropped"] == 0
        assert stats["traces_queued"] == 0

    @patch("cert.client.requests.post")
    def test_stats_tracks_sent(self, mock_post):
        """stats() accurately tracks sent traces."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.5)

        stats = client.stats()
        assert stats["traces_sent"] == 2
        client.close()

    def test_stats_tracks_queued(self, client):
        """stats() accurately tracks queued traces."""
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        stats = client.stats()
        assert stats["traces_queued"] >= 1


# =============================================================================
# API Key Header Tests
# =============================================================================


class TestApiKeyHeader:
    """Tests for API key header handling."""

    @patch("cert.client.requests.post")
    def test_api_key_header(self, mock_post):
        """API key is sent in X-API-Key header."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="my_secret_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-API-Key"] == "my_secret_key"
        client.close()

    @patch("cert.client.requests.post")
    def test_content_type_header(self, mock_post):
        """Content-Type is set to application/json."""
        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=1)
        client.trace(
            type="generation",
            provider="p",
            model="m",
            input_text="i",
            output_text="o",
            duration_ms=10,
        )

        time.sleep(0.3)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        client.close()


# =============================================================================
# Type Annotation Tests
# =============================================================================


class TestTypeAnnotations:
    """Tests for type annotation coverage."""

    def test_trace_type_literal(self):
        """TraceType is a Literal type."""
        # This just verifies the type is importable
        assert TraceType is not None

    def test_eval_mode_literal(self):
        """EvalMode is a Literal type."""
        assert EvalMode is not None

    def test_tool_call_typeddict(self):
        """ToolCall is a TypedDict."""
        assert ToolCall is not None


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for thread safety and concurrency."""

    @patch("cert.client.requests.post")
    def test_concurrent_traces(self, mock_post):
        """Client handles concurrent trace calls."""
        import threading

        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=100)
        results = []

        def trace_worker():
            for _ in range(10):
                trace_id = client.trace(
                    type="generation",
                    provider="p",
                    model="m",
                    input_text="i",
                    output_text="o",
                    duration_ms=10,
                )
                results.append(trace_id)

        threads = [threading.Thread(target=trace_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 50 traces should have unique IDs
        assert len(results) == 50
        assert len(set(results)) == 50

        client.close()

    @patch("cert.client.requests.post")
    def test_concurrent_stats(self, mock_post):
        """stats() is thread-safe."""
        import threading

        mock_post.return_value = Mock(status_code=200)

        client = CertClient(api_key="test_key", batch_size=1)
        errors = []

        def stats_worker():
            try:
                for _ in range(100):
                    stats = client.stats()
                    assert isinstance(stats["traces_sent"], int)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stats_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        client.close()
