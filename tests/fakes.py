"""In-memory fake of the peer-component HTTP surface for engine tests.

`FakePeerClient` structurally satisfies `PeerClientProtocol`, so the engine
drives it exactly as it would the real `PeerClient` — but deterministically
and without a network. Behavior is tunable per test: training can complete,
fail, or hang (so a cancel can be exercised); a dataset can be marked empty;
peers can be made unreachable for the /healthz probe.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

from eugene_plexus_coordinator.engine.errors import PeerUnavailable


class FakePeerClient:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

        # data component
        self.dataset_status: dict[str, str] = {}  # id -> status; default "ready"
        self.tokenizer_ids: set[str] = set()
        self.tokenizer_status: dict[str, str] = {}  # id -> status; default "ready"

        # trainer
        self.training_mode = "complete"  # "complete" | "fail" | "hang"
        self.training_error = "boom"
        self._runs: dict[str, dict[str, Any]] = {}  # runId -> {polls, checkpointId}

        # inference
        # "ready" -> load returns ready immediately; "loading_then_ready" ->
        # load returns loading, the first list poll flips it to ready;
        # "error" -> load returns error.
        self.inference_mode = "ready"
        self._endpoints: dict[str, dict[str, Any]] = {}

        # meta
        self.unreachable: set[str] = set()  # base URLs that fail /healthz

    # ----------------------------------------------------------------- #
    # test helpers
    # ----------------------------------------------------------------- #

    def register_dataset(self, dataset_id: str, *, status: str = "ready") -> None:
        with self._lock:
            self.dataset_status[dataset_id] = status

    def register_tokenizer(self, tokenizer_id: str, *, status: str = "ready") -> None:
        with self._lock:
            self.tokenizer_ids.add(tokenizer_id)
            self.tokenizer_status[tokenizer_id] = status

    def was_called(self, method: str) -> bool:
        with self._lock:
            return any(name == method for name, _ in self.calls)

    def _record(self, method: str, *args: Any) -> None:
        with self._lock:
            self.calls.append((method, args))

    # ----------------------------------------------------------------- #
    # meta
    # ----------------------------------------------------------------- #

    def healthz(self, base_url: str) -> dict[str, Any]:
        self._record("healthz", base_url)
        if base_url in self.unreachable:
            raise PeerUnavailable(f"{base_url} refused connection")
        return {"status": "ok"}

    # ----------------------------------------------------------------- #
    # data component
    # ----------------------------------------------------------------- #

    def get_dataset(self, data_url: str, dataset_id: str) -> dict[str, Any]:
        self._record("get_dataset", dataset_id)
        with self._lock:
            status = self.dataset_status.get(dataset_id, "ready")
        return {
            "datasetId": dataset_id,
            "name": f"dataset-{dataset_id[:8]}",
            "status": status,
            "vocabFingerprint": "fp" if status == "ready" else None,
        }

    def pretokenize_dataset(
        self, data_url: str, dataset_id: str, *, tokenizer_id: str, block_size: int
    ) -> dict[str, Any]:
        self._record("pretokenize_dataset", dataset_id, tokenizer_id, block_size)
        with self._lock:
            # Pretokenization makes the dataset ready (idempotent here).
            self.dataset_status.setdefault(dataset_id, "ready")
        return {"datasetId": dataset_id, "status": "pretokenizing"}

    def list_tokenizers(self, data_url: str) -> list[dict[str, Any]]:
        self._record("list_tokenizers", data_url)
        with self._lock:
            return [
                {"tokenizerId": tid, "status": self.tokenizer_status.get(tid, "ready"), "name": tid}
                for tid in self.tokenizer_ids
            ]

    # ----------------------------------------------------------------- #
    # trainer
    # ----------------------------------------------------------------- #

    def start_training_run(self, trainer_url: str, request: dict[str, Any]) -> dict[str, Any]:
        self._record("start_training_run", request)
        run_id = str(uuid4())
        with self._lock:
            self._runs[run_id] = {"polls": 0, "checkpointId": str(uuid4())}
        return {"runId": run_id, "projectId": request.get("projectId"), "status": "queued"}

    def get_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]:
        self._record("get_training_run", run_id)
        with self._lock:
            state = self._runs.get(run_id, {"polls": 0})
            state["polls"] = state.get("polls", 0) + 1
            polls = state["polls"]
            self._runs[run_id] = state
            mode = self.training_mode
        if mode == "hang":
            return {"runId": run_id, "status": "running", "currentStep": polls}
        if polls < 2:
            return {"runId": run_id, "status": "running", "currentStep": polls}
        if mode == "fail":
            return {"runId": run_id, "status": "failed", "lastError": self.training_error}
        return {"runId": run_id, "status": "completed", "currentStep": 100}

    def list_run_checkpoints(self, trainer_url: str, run_id: str) -> list[dict[str, Any]]:
        self._record("list_run_checkpoints", run_id)
        with self._lock:
            cp = self._runs.get(run_id, {}).get("checkpointId", str(uuid4()))
        return [
            {
                "checkpointId": cp,
                "runId": run_id,
                "step": 100,
                "isLatest": True,
                "isBest": True,
            }
        ]

    def cancel_training_run(self, trainer_url: str, run_id: str) -> dict[str, Any]:
        self._record("cancel_training_run", run_id)
        return {"runId": run_id, "status": "cancelled"}

    # ----------------------------------------------------------------- #
    # eval
    # ----------------------------------------------------------------- #

    def start_eval_run(self, eval_url: str, *, suite_id: str, checkpoint_id: str) -> dict[str, Any]:
        self._record("start_eval_run", suite_id, checkpoint_id)
        return {
            "evalRunId": str(uuid4()),
            "evalSuiteId": suite_id,
            "checkpointId": checkpoint_id,
            "status": "completed",
            "valLoss": 1.5,
            "perplexity": 4.48,
        }

    def get_eval_result(self, eval_url: str, eval_run_id: str) -> dict[str, Any]:
        self._record("get_eval_result", eval_run_id)
        return {"evalRunId": eval_run_id, "status": "completed", "valLoss": 1.5, "perplexity": 4.48}

    # ----------------------------------------------------------------- #
    # inference
    # ----------------------------------------------------------------- #

    def create_endpoint(self, inference_url: str, endpoint: dict[str, Any]) -> dict[str, Any]:
        self._record("create_endpoint", endpoint)
        with self._lock:
            stored = {**endpoint, "status": "unloaded"}
            self._endpoints[endpoint["endpointId"]] = stored
        return stored

    def load_endpoint(
        self, inference_url: str, endpoint_id: str, *, checkpoint_id: str
    ) -> dict[str, Any]:
        self._record("load_endpoint", endpoint_id, checkpoint_id)
        with self._lock:
            ep = self._endpoints.setdefault(endpoint_id, {"endpointId": endpoint_id})
            ep["checkpointId"] = checkpoint_id
            if self.inference_mode == "error":
                ep["status"] = "error"
            elif self.inference_mode == "loading_then_ready":
                ep["status"] = "loading"
            else:
                ep["status"] = "ready"
            return dict(ep)

    def list_endpoints(self, inference_url: str) -> list[dict[str, Any]]:
        self._record("list_endpoints", inference_url)
        with self._lock:
            # In loading_then_ready mode, the first poll observes the endpoint
            # flipping from loading to ready.
            for ep in self._endpoints.values():
                if ep.get("status") == "loading":
                    ep["status"] = "ready"
            return [dict(ep) for ep in self._endpoints.values()]

    def close(self) -> None:
        self._record("close")
