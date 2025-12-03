"""Tests for CERT client."""

import time
from unittest.mock import Mock, patch

import pytest

from cert import CertClient


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
