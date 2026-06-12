"""Unit tests for the synchronous peer HTTP client.

Uses httpx's MockTransport so the real request/response + error-mapping code
runs with no network and no server.
"""

from __future__ import annotations

import httpx
import pytest

from eugene_plexus_coordinator.engine.errors import PeerError, PeerUnavailable
from eugene_plexus_coordinator.engine.peer_client import PeerClient


def _client(handler, *, token: str | None = None) -> PeerClient:
    transport = httpx.MockTransport(handler)
    return PeerClient(service_token=token, client=httpx.Client(transport=transport))


def test_get_dataset_parses_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/data/datasets/abc"
        return httpx.Response(200, json={"datasetId": "abc", "status": "ready"})

    client = _client(handler)
    body = client.get_dataset("http://data", "abc")
    assert body["status"] == "ready"


def test_list_unwraps_collection_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"tokenizers": [{"tokenizerId": "t1"}, {"tokenizerId": "t2"}]}
        )

    client = _client(handler)
    tokenizers = client.list_tokenizers("http://data")
    assert [t["tokenizerId"] for t in tokenizers] == ["t1", "t2"]


def test_trailing_slash_in_base_url_is_handled() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={})

    client = _client(handler)
    client.get_dataset("http://data/", "abc")
    assert seen["path"] == "/v1/data/datasets/abc"


def test_non_2xx_raises_peer_error_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no such dataset", "title": "Not found"})

    client = _client(handler)
    with pytest.raises(PeerError) as exc:
        client.get_dataset("http://data", "missing")
    assert exc.value.status_code == 404
    # the Problem detail is surfaced in the message
    assert "no such dataset" in str(exc.value)


def test_connect_error_raises_peer_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(PeerUnavailable):
        client.healthz("http://trainer")


def test_auth_header_present_when_token_set() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler, token="svc-token")
    client.healthz("http://trainer")
    assert seen["auth"] == "Bearer svc-token"


def test_no_auth_header_when_token_absent() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    client = _client(handler)
    client.healthz("http://trainer")
    assert seen["auth"] is None


def test_empty_body_returns_empty_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)  # no content

    client = _client(handler)
    assert client.cancel_training_run("http://trainer", "run-1") == {}


def test_start_training_run_posts_json_body() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["content"] = request.content
        return httpx.Response(201, json={"runId": "r1", "status": "queued"})

    client = _client(handler)
    result = client.start_training_run("http://trainer", {"projectId": "p1"})
    assert seen["method"] == "POST"
    assert b"projectId" in seen["content"]  # type: ignore[operator]
    assert result["runId"] == "r1"
