"""Tests for the config protocol endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_coordinator_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "coordinator"
    keys = {f["key"] for f in body["fields"]}
    # `port` is not here — owned by the watchdog topology via
    # EUGENE_PLEXUS_CRD_BIND_PORT.
    assert keys == {
        "trainerUrl",
        "dataUrl",
        "evalUrl",
        "inferenceUrl",
        "projectStorePath",
        "logLevel",
    }


def test_peer_url_fields_carry_component_kind_hints(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    fields = {f["key"]: f for f in response.json()["fields"]}
    assert fields["trainerUrl"]["componentKindHint"] == "trainer"
    assert fields["dataUrl"]["componentKindHint"] == "data"
    assert fields["evalUrl"]["componentKindHint"] == "eval"
    assert fields["inferenceUrl"]["componentKindHint"] == "inference"


def test_get_config_returns_defaults(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert "port" not in body
    assert body["logLevel"] == "INFO"
    # Seeded by the conftest fixture.
    assert body["trainerUrl"] == "http://127.0.0.1:8087"
    assert body["dataUrl"] == ""


def test_patch_config_validates_per_field(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "dataUrl": "http://127.0.0.1:8088",  # valid url
            "logLevel": "DEBUG",  # valid enum, requiresRestart
            "logLevel_typo": "DEBUG",  # unknown field
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["applied"]) == {"dataUrl", "logLevel"}
    rejected = {r["key"] for r in body["rejected"]}
    assert rejected == {"logLevel_typo"}
    # logLevel is requiresRestart
    assert body["requiresRestart"] is True
    assert "logLevel" in body["pendingRestart"]


def test_patch_config_rejects_bad_enum(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"logLevel": "VERBOSE"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "logLevel"


def test_patch_config_rejects_unknown_field(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"madeUpKey": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "madeUpKey"
    assert "unknown field" in body["rejected"][0]["message"]


def test_config_test_reports_configured_peers(client: TestClient) -> None:
    response = client.post("/v1/config/test", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "coordinator"
    # The conftest fixture configures trainerUrl, so the test passes.
    assert body["ok"] is True
    assert "latencyMs" in body
    assert "trainer" in body["summary"]
