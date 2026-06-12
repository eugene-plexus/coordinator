"""Synchronous HTTP client for the platform's peer components.

The coordinator is pure orchestration: every pipeline stage delegates to a
peer component (`data`, `trainer`, `eval`, `inference`) over HTTP+JSON. This
module is the single seam through which those calls go.

Design notes
------------
* **Synchronous.** Pipeline runs execute in background worker threads (see
  ``engine.PipelineEngine``); a worker drives its stages with blocking calls
  and ``time.sleep`` between status polls. A sync client is the natural fit
  and keeps the worker code linear.
* **Schema-light.** Per the platform's "components share schemas, not code"
  rule, the coordinator does NOT import peer-specific request/response models
  (those live behind each peer's own spec). It builds request payloads as
  plain dicts and reads only the fields it needs out of the JSON responses.
  The shared `common.yaml` types it *does* own (TrainingRunRequest, etc.) are
  serialized to dicts before they reach this layer.
* **Auth.** The coordinator presents its long-lived service token
  (``aud: service:coordinator``) as a bearer on every outbound call, threaded
  in by the watchdog at spawn time.

Errors map to the engine taxonomy: a reachable peer returning non-2xx raises
`PeerError` (with the status code); an unreachable peer (connect/read
timeout, DNS, refused) raises `PeerUnavailable`.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from .errors import PeerError, PeerUnavailable

# Default per-call HTTP timeout. Long enough to tolerate a peer doing real
# work before it returns its "accepted" response, short enough that a hung
# peer surfaces as PeerUnavailable rather than wedging a worker forever.
DEFAULT_TIMEOUT_S = 30.0


def _extract_detail(response: httpx.Response) -> str:
    """Pull a human-readable error detail out of a peer error response.

    Peer components return RFC-7807 `Problem` bodies (`{title, detail, ...}`)
    on error. Fall back to the raw text, then to the reason phrase.
    """
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text or response.reason_phrase or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("title")
        if isinstance(detail, str) and detail:
            return detail
    return response.reason_phrase or f"HTTP {response.status_code}"


class PeerClientProtocol(Protocol):
    """Structural type the engine depends on (real client or a test fake)."""

    def healthz(self, base_url: str) -> dict[str, Any]: ...

    def get_dataset(self, data_url: str, dataset_id: str) -> dict[str, Any]: ...

    def pretokenize_dataset(
        self, data_url: str, dataset_id: str, *, tokenizer_id: str, block_size: int
    ) -> dict[str, Any]: ...

    def list_tokenizers(self, data_url: str) -> list[dict[str, Any]]: ...

    def start_training_run(self, trainer_url: str, request: dict[str, Any]) -> dict[str, Any]: ...

    def get_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]: ...

    def list_run_checkpoints(self, trainer_url: str, run_id: str) -> list[dict[str, Any]]: ...

    def cancel_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]: ...

    def start_eval_run(
        self, eval_url: str, *, suite_id: str, checkpoint_id: str
    ) -> dict[str, Any]: ...

    def get_eval_result(self, eval_url: str, eval_run_id: str) -> dict[str, Any]: ...

    def create_endpoint(self, inference_url: str, endpoint: dict[str, Any]) -> dict[str, Any]: ...

    def load_endpoint(
        self, inference_url: str, endpoint_id: str, *, checkpoint_id: str
    ) -> dict[str, Any]: ...

    def list_endpoints(self, inference_url: str) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


class PeerClient:
    """Sync httpx-backed implementation of `PeerClientProtocol`."""

    def __init__(
        self,
        *,
        service_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self._token = service_token
        # follow_redirects so a trailing-slash / auth-proxy 3xx resolves
        # transparently instead of slipping past the >=400 check and failing
        # later as a confusing JSON-decode error.
        self._client = (
            client if client is not None else httpx.Client(timeout=timeout, follow_redirects=True)
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ----------------------------------------------------------------- #
    # Low-level request
    # ----------------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = base_url.rstrip("/") + path
        try:
            response = self._client.request(
                method, url, json=json, params=params, headers=self._headers()
            )
        except httpx.RequestError as e:
            raise PeerUnavailable(f"{method} {url} failed to connect: {e}") from e

        if response.status_code >= 400:
            raise PeerError(
                f"{method} {url} -> {response.status_code}: {_extract_detail(response)}",
                status_code=response.status_code,
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as e:
            raise PeerError(f"{method} {url}: response was not valid JSON: {e}") from e

    @staticmethod
    def _as_list(body: Any, key: str) -> list[dict[str, Any]]:
        """Unwrap a ``{key: [...]}`` collection response into the list."""
        if isinstance(body, dict):
            items = body.get(key)
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
        return []

    # ----------------------------------------------------------------- #
    # meta
    # ----------------------------------------------------------------- #

    def healthz(self, base_url: str) -> dict[str, Any]:
        body = self._request("GET", base_url, "/healthz")
        return body if isinstance(body, dict) else {}

    # ----------------------------------------------------------------- #
    # data component
    # ----------------------------------------------------------------- #

    def get_dataset(self, data_url: str, dataset_id: str) -> dict[str, Any]:
        body = self._request("GET", data_url, f"/v1/data/datasets/{dataset_id}")
        return body if isinstance(body, dict) else {}

    def pretokenize_dataset(
        self, data_url: str, dataset_id: str, *, tokenizer_id: str, block_size: int
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            data_url,
            f"/v1/data/datasets/{dataset_id}/pretokenize",
            json={"tokenizerId": tokenizer_id, "blockSize": block_size},
        )
        return body if isinstance(body, dict) else {}

    def list_tokenizers(self, data_url: str) -> list[dict[str, Any]]:
        return self._as_list(self._request("GET", data_url, "/v1/data/tokenizers"), "tokenizers")

    # ----------------------------------------------------------------- #
    # trainer component
    # ----------------------------------------------------------------- #

    def start_training_run(self, trainer_url: str, request: dict[str, Any]) -> dict[str, Any]:
        body = self._request("POST", trainer_url, "/v1/trainer/runs", json=request)
        return body if isinstance(body, dict) else {}

    def get_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]:
        body = self._request("GET", trainer_url, f"/v1/trainer/runs/{run_id}")
        return body if isinstance(body, dict) else {}

    def list_run_checkpoints(self, trainer_url: str, run_id: str) -> list[dict[str, Any]]:
        return self._as_list(
            self._request("GET", trainer_url, f"/v1/trainer/runs/{run_id}/checkpoints"),
            "checkpoints",
        )

    def cancel_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]:
        body = self._request("POST", trainer_url, f"/v1/trainer/runs/{run_id}/cancel")
        return body if isinstance(body, dict) else {}

    # ----------------------------------------------------------------- #
    # eval component
    # ----------------------------------------------------------------- #

    def start_eval_run(self, eval_url: str, *, suite_id: str, checkpoint_id: str) -> dict[str, Any]:
        body = self._request(
            "POST",
            eval_url,
            "/v1/eval/runs",
            json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id},
        )
        return body if isinstance(body, dict) else {}

    def get_eval_result(self, eval_url: str, eval_run_id: str) -> dict[str, Any]:
        body = self._request("GET", eval_url, f"/v1/eval/runs/{eval_run_id}")
        return body if isinstance(body, dict) else {}

    # ----------------------------------------------------------------- #
    # inference component
    # ----------------------------------------------------------------- #

    def create_endpoint(self, inference_url: str, endpoint: dict[str, Any]) -> dict[str, Any]:
        body = self._request("POST", inference_url, "/v1/inference/endpoints", json=endpoint)
        return body if isinstance(body, dict) else {}

    def load_endpoint(
        self, inference_url: str, endpoint_id: str, *, checkpoint_id: str
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            inference_url,
            f"/v1/inference/endpoints/{endpoint_id}/load",
            json={"checkpointId": checkpoint_id},
        )
        return body if isinstance(body, dict) else {}

    def list_endpoints(self, inference_url: str) -> list[dict[str, Any]]:
        return self._as_list(
            self._request("GET", inference_url, "/v1/inference/endpoints"), "endpoints"
        )
