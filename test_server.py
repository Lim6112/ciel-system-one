"""Tests for the /api/predict endpoint with a stubbed router (no model load).

uv run pytest test_server.py
"""

from __future__ import annotations

import io
import time
import urllib.error

import pytest
from fastapi.testclient import TestClient

import decision_lab_proxy
import server as server_module


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a stub router that echoes the request it received."""

    class StubRouter:
        def predict(self, state, questions, **kwargs):
            self.last_call = {"state": state, "questions": questions, "kwargs": kwargs}
            return {
                "answers": {
                    qid: {"type": q["type"], "confidence": 0.9}
                    for qid, q in questions.items()
                },
                "routing": {"model": kwargs.get("model", "english")},
            }

    stub = StubRouter()
    monkeypatch.setattr(server_module._RouterState, "router", stub)
    # Deliberately NOT a `with` block: entering the TestClient context would run
    # the app lifespan and replace the stub with the real (slow) router.
    yield TestClient(server_module.app), stub


VALID_CHOICE = {
    "type": "choice",
    "instructions": "Which department?",
    "criteria": {"billing": "payments", "other": "everything else"},
}
VALID_NOUL = {"type": "noul", "instructions": "Churn risk?"}


def test_predict_success(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"subject": "Duplicate charge"},
        "questions": {"department": VALID_CHOICE, "churn_risk": VALID_NOUL},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["routing"]["model"] == "english"  # auto -> no override
    assert set(body["result"]["answers"]) == {"department", "churn_risk"}
    assert body["latency_ms"] >= 0
    assert stub.last_call["state"] == {"subject": "Duplicate charge"}


def test_predict_forwards_model_override(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "typed-decisions",
    })
    assert response.status_code == 200
    assert stub.last_call["kwargs"] == {"model": "typed-decisions"}


def test_predict_strips_criteria_for_noul(client):
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
    })
    assert response.status_code == 200
    assert stub.last_call["questions"]["q"] == {"type": "noul", "instructions": "Churn risk?"}


def test_rejects_choice_without_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {}}},
    })
    assert response.status_code == 422


def test_rejects_score_without_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "score", "instructions": "how bad", "criteria": []}},
    })
    assert response.status_code == 422


def test_rejects_noul_with_criteria(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "noul", "instructions": "yes?", "criteria": ["x"]}},
    })
    assert response.status_code == 422


def test_rejects_unknown_question_type(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": {"type": "essay", "instructions": "write"}},
    })
    assert response.status_code == 422


def test_rejects_empty_state_and_questions(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={"state": {}, "questions": {"q": VALID_NOUL}})
    assert response.status_code == 422
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"}, "questions": {}})
    assert response.status_code == 422


def test_rejects_unknown_model(client):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "gpt-4",
    })
    assert response.status_code == 422


def test_health_ready(client):
    test_client, _ = client
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_index_served(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "CJet web UI" in response.text


def test_concurrent_predictions_are_serialized(client, monkeypatch):
    """router.predict() must never run concurrently (MPS/Metal crashes if it does)."""
    import threading

    test_client, stub = client
    overlap = []

    def slow_predict(state, questions, **kwargs):
        inside = {"entered": True}
        overlap.append(inside)
        time.sleep(0.05)
        overlap.remove(inside)
        return {"answers": {}, "routing": {}}

    monkeypatch.setattr(stub, "predict", slow_predict)
    threads = [
        threading.Thread(
            target=test_client.post,
            args=("/api/predict",),
            kwargs={"json": {"state": {"body": "hi"}, "questions": {"q": VALID_NOUL}}},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlap, "router.predict ran concurrently"


# ---- jev ---------------------------------------------------------------------
# These stub jev.predict, so the suite needs neither a network nor an API key.

JEV_REPLY = {
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "assign": {
            "type": "choice",
            "choice": "bus_b",
            "probabilities": {"bus_a": 0.3, "bus_b": 0.55, "bus_c": 0.15},
            "confidence": 0.42,
        }
    },
    "usage": {"input_tokens": 373, "output_tokens": 42},
    "cost_usd": 1.5666e-05,
}


@pytest.fixture()
def jev_stub(monkeypatch):
    """Stub the Jev client and hand back the calls it received."""
    calls = []

    def fake_predict(state, questions, **kwargs):
        calls.append({"state": state, "questions": questions, **kwargs})
        return dict(JEV_REPLY)

    monkeypatch.setattr(server_module.jev, "predict", fake_predict)
    return calls


def test_jev_branch_returns_laya_shaped_result(client, jev_stub):
    """Jev answers through the same envelope, so no client needs a branch."""
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["answers"]["assign"]["choice"] == "bus_b"
    assert body["result"]["cost_usd"] == 1.5666e-05
    assert body["routing"]["model"] == "jev"
    # The laya router is never touched on this path.
    assert not hasattr(stub, "last_call")
    assert jev_stub[0]["state"] == {"body": "hi"}
    assert jev_stub[0]["questions"]["assign"]["criteria"] == VALID_CHOICE["criteria"]


def test_jev_answer_has_no_act_probability(client, jev_stub):
    """Jev omits action.act_probability; clients must tolerate its absence."""
    test_client, _ = client
    body = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    }).json()
    answer = body["result"]["answers"]["assign"]
    assert "action" not in answer
    # fleet.html reads it as (answer.action || {}).act_probability ?? 1
    assert (answer.get("action") or {}).get("act_probability", 1) == 1


def test_jev_works_while_the_laya_router_is_missing(client, jev_stub, monkeypatch):
    """Jev never touches the local model, so it must not 503 on a cold router."""
    test_client, _ = client
    monkeypatch.setattr(server_module._RouterState, "router", None)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 200


def test_jev_does_not_take_the_predict_lock(client, jev_stub, monkeypatch):
    """The lock exists for MPS/Metal; a ~1s network call must not serialise on it."""
    import threading

    test_client, _ = client
    overlap = []

    def slow_predict(state, questions, **kwargs):
        overlap.append(1)
        time.sleep(0.05)
        overlap.remove(1)
        return dict(JEV_REPLY)

    monkeypatch.setattr(server_module.jev, "predict", slow_predict)
    threads = [
        threading.Thread(
            target=test_client.post,
            args=("/api/predict",),
            kwargs={"json": {"state": {"body": "hi"},
                             "questions": {"assign": VALID_CHOICE},
                             "model": "jev"}},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlap, "jev.predict ran while holding predict_lock"


def test_jev_error_becomes_a_502_with_the_upstream_body(client, monkeypatch):
    test_client, _ = client

    def boom(state, questions, **kwargs):
        raise server_module.jev.JevError("Jev request failed (402)", status=402,
                                         body='{"error":{"message":"insufficient credit"}}')

    monkeypatch.setattr(server_module.jev, "predict", boom)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "jev",
    })
    assert response.status_code == 502
    assert "insufficient credit" in response.json()["detail"]


def test_jev_rejects_unknown_model_still(client, jev_stub):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "gpt-4",
    })
    assert response.status_code == 422


def test_jev_requires_a_key(monkeypatch):
    """A missing key must name the variable rather than fail opaquely."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(server_module.jev.JevError, match="OPENROUTER_API_KEY"):
        server_module.jev.api_key()


# ---- ciel (decision_lab dynamic head) -----------------------------------------
# These stub ciel.predict, so the suite needs neither a network nor the service.

CIEL_REPLY = {
    "model": "ciel_decision_model_v1",
    "provider": "decision_lab",
    "answers": {
        "assign": {
            "type": "choice",
            "choice": "bus_b",
            "probabilities": {"bus_a": 0.3, "bus_b": 0.55, "bus_c": 0.15},
            "confidence": 0.55,
        },
        "urgency": {
            "type": "score",
            "score": 1.73,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.05, "1": 0.27, "2": 0.68},
            "confidence": 0.68,
        },
        "refund": {"type": "noul", "noul": 0.91, "confidence": 0.91},
    },
    "usage": {"input_tokens": 0, "output_tokens": 0},
}

VALID_SCORE = {
    "type": "score",
    "instructions": "urgency",
    "criteria": ["low", "medium", "high"],
}


@pytest.fixture()
def ciel_stub(monkeypatch):
    """Stub the Ciel client and hand back the calls it received."""
    calls = []

    def fake_predict(state, questions, **kwargs):
        calls.append({"state": state, "questions": questions, **kwargs})
        return dict(CIEL_REPLY)

    monkeypatch.setattr(server_module.ciel, "predict", fake_predict)
    return calls


def test_ciel_branch_returns_laya_shaped_result(client, ciel_stub):
    """Ciel answers through the same envelope, so no client needs a branch."""
    test_client, stub = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "ciel_decision_model_v1",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["answers"]["assign"]["choice"] == "bus_b"
    assert body["routing"]["model"] == "ciel_decision_model_v1"
    assert body["routing"]["repo"] == "decision_lab/dynamic-head"
    # No sub-path in MODEL_PATHS, so the service's own default head answers — reported
    # explicitly rather than left absent, because `index.html` renders this field and a
    # missing one would read as "no path" instead of "the default path".
    assert body["routing"]["path"] == "(service default)"
    # The laya router is never touched on this path.
    assert not hasattr(stub, "last_call")
    assert ciel_stub[0]["state"] == {"body": "hi"}
    assert ciel_stub[0]["questions"]["assign"]["criteria"] == VALID_CHOICE["criteria"]


def test_ciel_routing_reports_the_sub_path_that_answered(client, ciel_stub):
    """Each sub-path arm names its own head, so the UI can show which one replied.

    The value the page offers is what `resolve_path` maps; this pins the pairing
    end to end, so a dropdown option cannot be added that silently lands on the
    default head while its label claims a trained one.
    """
    test_client, _ = client
    for model, expected in [("ciel_warehouse_v1", "warehouse"),
                            ("ciel_typed_v1", "typed")]:
        body = test_client.post("/api/predict", json={
            "state": {"body": "hi"},
            "questions": {"assign": VALID_CHOICE},
            "model": model,
        }).json()
        assert body["routing"]["model"] == model
        assert body["routing"]["path"] == expected


def test_ciel_answer_shapes_match_laya(client, ciel_stub):
    """choice/score/noul answers use laya's per-type fields, not a uniform shape."""
    test_client, _ = client
    body = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE, "urgency": VALID_SCORE, "refund": VALID_NOUL},
        "model": "ciel_decision_model_v1",
    }).json()
    answers = body["result"]["answers"]
    score = answers["urgency"]
    assert score["legend"]["0"] == "low"
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert isinstance(score["score"], float)
    noul = answers["refund"]
    assert "action" not in noul
    # fleet.html reads it as (answer.action || {}).act_probability ?? 1
    assert (noul.get("action") or {}).get("act_probability", 1) == 1


def test_ciel_has_no_cost(client, ciel_stub):
    """The service is local and free; the UI's cost line must not appear."""
    test_client, _ = client
    body = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "ciel_decision_model_v1",
    }).json()
    assert "cost_usd" not in body["result"]


def test_ciel_works_while_the_laya_router_is_missing(client, ciel_stub, monkeypatch):
    """Ciel never touches the local model, so it must not 503 on a cold router."""
    test_client, _ = client
    monkeypatch.setattr(server_module._RouterState, "router", None)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "ciel_decision_model_v1",
    })
    assert response.status_code == 200


def test_ciel_does_not_take_the_predict_lock(client, ciel_stub, monkeypatch):
    """The lock exists for MPS/Metal; a network call must not serialise on it."""
    import threading

    test_client, _ = client
    overlap = []

    def slow_predict(state, questions, **kwargs):
        overlap.append(1)
        time.sleep(0.05)
        overlap.remove(1)
        return dict(CIEL_REPLY)

    monkeypatch.setattr(server_module.ciel, "predict", slow_predict)
    threads = [
        threading.Thread(
            target=test_client.post,
            args=("/api/predict",),
            kwargs={"json": {"state": {"body": "hi"},
                             "questions": {"assign": VALID_CHOICE},
                             "model": "ciel_decision_model_v1"}},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlap, "ciel.predict ran while holding predict_lock"


def test_ciel_error_becomes_a_502_with_the_upstream_body(client, monkeypatch):
    test_client, _ = client

    def boom(state, questions, **kwargs):
        raise server_module.ciel.CielError("Decision Lab request failed (400)", status=400,
                                           body='{"detail":"dynamic head not trained"}')

    monkeypatch.setattr(server_module.ciel, "predict", boom)
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"assign": VALID_CHOICE},
        "model": "ciel_decision_model_v1",
    })
    assert response.status_code == 502
    assert "head not trained" in response.json()["detail"]


def test_ciel_rejects_unknown_model_still(client, ciel_stub):
    test_client, _ = client
    response = test_client.post("/api/predict", json={
        "state": {"body": "hi"},
        "questions": {"q": VALID_NOUL},
        "model": "gpt-4",
    })
    assert response.status_code == 422

# ---- /api/transcribe -------------------------------------------------------------
#
# The endpoint is exercised with asr.transcribe stubbed, so these run with no model, no
# network and no microphone — same contract the rest of this file holds to. What is
# tested here is the WIRING: validation, status codes, and the fact that transcription
# does not go through laya's predict_lock.


def _wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A minimal mono 16-bit PCM WAV, built without numpy so this file needs no import."""
    import io
    import math
    import struct
    import wave

    n = int(seconds * rate)
    pcm = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / rate)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture()
def asr_stub(monkeypatch):
    """Replace asr.transcribe with a recorder that returns a fixed transcript."""
    calls = []

    def fake_transcribe(wav_bytes, lang="en"):
        calls.append({"bytes": len(wav_bytes), "lang": lang})
        return {"transcript": "turn left", "model": f"stub-{lang}", "lang": lang,
                "duration_s": 1.0}

    monkeypatch.setattr(server_module.asr, "transcribe", fake_transcribe)
    return calls


def test_transcribe_returns_the_transcript(client, asr_stub):
    test_client, _ = client
    response = test_client.post(
        "/api/transcribe?lang=en", content=_wav(),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "turn left"
    assert body["lang"] == "en"
    assert body["latency_ms"] >= 0
    assert asr_stub[0]["lang"] == "en"


def test_transcribe_defaults_to_english(client, asr_stub):
    test_client, _ = client
    response = test_client.post("/api/transcribe", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 200
    assert asr_stub[0]["lang"] == "en"


def test_transcribe_passes_the_language_through(client, asr_stub):
    test_client, _ = client
    response = test_client.post("/api/transcribe?lang=ko", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 200
    assert asr_stub[0]["lang"] == "ko"


def test_transcribe_rejects_an_unsupported_language(client, asr_stub):
    test_client, _ = client
    response = test_client.post("/api/transcribe?lang=fr", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 422
    assert "unsupported lang" in response.json()["detail"]
    assert not asr_stub, "the stub ran despite an invalid language"


def test_transcribe_rejects_an_empty_body(client, asr_stub):
    test_client, _ = client
    response = test_client.post("/api/transcribe", content=b"",
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 422
    assert not asr_stub


def test_transcribe_rejects_an_oversized_body(client, asr_stub):
    test_client, _ = client
    big = b"\x00" * (server_module.MAX_AUDIO_BYTES + 1)
    response = test_client.post("/api/transcribe", content=big,
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 413
    assert not asr_stub, "an oversized body reached the model"


def test_transcribe_maps_a_bad_recording_to_400(client, monkeypatch):
    """A bad-body failure carries no status, so it must surface as a client error."""
    test_client, _ = client

    def boom(wav_bytes, lang="en"):
        raise server_module.asr.AsrError("not a readable WAV: nope")

    monkeypatch.setattr(server_module.asr, "transcribe", boom)
    response = test_client.post("/api/transcribe", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 400
    assert "not a readable WAV" in response.json()["detail"]


def test_transcribe_maps_a_load_failure_to_502(client, monkeypatch):
    """A load failure carries a status, so it must surface as upstream, not as the
    caller's fault — that is what lets the UI say "check your connection"."""
    test_client, _ = client

    def boom(wav_bytes, lang="en"):
        raise server_module.asr.AsrError("could not load the checkpoint", status=502)

    monkeypatch.setattr(server_module.asr, "transcribe", boom)
    response = test_client.post("/api/transcribe", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 502
    assert "could not load" in response.json()["detail"]


def test_transcribe_works_when_the_laya_router_is_cold(monkeypatch, asr_stub):
    """Voice must not wait on laya. The router is None here; the request still succeeds.

    This is the same property jev and ciel have, and it is the reason /api/transcribe
    does not take predict_lock or 503 on a cold router.
    """
    monkeypatch.setattr(server_module._RouterState, "router", None)
    test_client = TestClient(server_module.app)
    response = test_client.post("/api/transcribe", content=_wav(),
                                headers={"content-type": "application/octet-stream"})
    assert response.status_code == 200
    assert response.json()["transcript"] == "turn left"


def test_health_is_unchanged_by_the_new_endpoint(client):
    """The penguin demo must not change what /api/health reports."""
    test_client, _ = client
    assert test_client.get("/api/health").json() == {"ready": True}


def test_transcribe_rejects_an_oversized_content_length_before_reading(client, asr_stub):
    """REGRESSION. The guard used to run only after `await request.body()` had buffered
    the whole upload, so it rejected large bodies but did not bound memory. The declared
    length must be refused up front."""
    test_client, _ = client
    response = test_client.post(
        "/api/transcribe", content=b"",
        headers={"content-type": "application/octet-stream",
                 "content-length": str(server_module.MAX_AUDIO_BYTES + 1)},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]
    assert not asr_stub, "an oversized body reached the model"


def test_transcribe_still_checks_the_real_length_when_the_header_lies(client, asr_stub):
    """A chunked upload declares no length. The post-read check must still catch it."""
    test_client, _ = client
    big = b"\x00" * (server_module.MAX_AUDIO_BYTES + 1)
    response = test_client.post(
        "/api/transcribe", content=big,
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 413
    assert not asr_stub


def test_transcribe_accepts_a_usable_content_length(client, asr_stub):
    """The pre-check must not reject legitimate uploads."""
    test_client, _ = client
    wav = _wav()
    response = test_client.post(
        "/api/transcribe", content=wav,
        headers={"content-type": "application/octet-stream",
                 "content-length": str(len(wav))},
    )
    assert response.status_code == 200


def test_transcribe_language_is_normalised_like_asr_does(client, asr_stub):
    """asr folds case and whitespace; the endpoint must not be stricter than the module
    it fronts."""
    test_client, _ = client
    for lang in ["en", "EN", " en "]:
        asr_stub.clear()
        response = test_client.post(
            f"/api/transcribe?lang={lang}", content=_wav(),
            headers={"content-type": "application/octet-stream"},
        )
        assert response.status_code == 200, f"{lang!r} was rejected"


def test_a_truncated_wav_body_returns_400_not_500(client):
    """End to end through the real decoder (not a stub): a 4-byte body must be a client
    error, not an unhandled EOFError."""
    test_client, _ = client
    response = test_client.post(
        "/api/transcribe", content=b"RIFF",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400, response.text
    assert "not a readable WAV" in response.json()["detail"]


def test_decision_lab_page_is_served():
    """The embedded Decision Lab page rides the catch-all static mount, so a typo in its
    filename (or dropping it from static/) would only show up in a browser. Check it is
    reachable and still carries the iframe the page is built around."""
    response = TestClient(server_module.app).get("/decision_lab.html")
    assert response.status_code == 200, response.text
    assert 'id="lab"' in response.text
    # Same-origin through the /lab relay, never the service's own port: that address
    # is exactly what a browser elsewhere cannot reach, and reaching it from an https
    # page would be blocked anyway.
    assert "location.origin}/lab" in response.text
    assert ":8000" not in response.text


# ---- /lab relay (Decision Lab's UI, served same-origin) -----------------------
# These stub the relay's single network seam, so the suite needs neither the
# service nor a browser.


def _relay_response(body: bytes, content_type: str | None = "application/json", status: int = 200):
    """A stand-in for urlopen's return value, read in chunks like the real one."""

    class FakeResponse:
        def __init__(self):
            self.status = status
            self.headers = {"content-type": content_type} if content_type else {}
            self._body = io.BytesIO(body)

        def read(self, size: int = -1) -> bytes:
            return self._body.read(size)

        def close(self) -> None:
            pass

    return FakeResponse()


@pytest.fixture()
def lab(monkeypatch):
    """Stub the relay's network seam and record what it was asked to send."""
    calls = []
    state = {"response": _relay_response(b'{"ready":true}')}

    def fake_open(request, timeout):
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": {name.lower(): value for name, value in request.header_items()},
            "body": request.data,
            "timeout": timeout,
        })
        if isinstance(state["response"], Exception):
            raise state["response"]
        return state["response"]

    monkeypatch.setattr(decision_lab_proxy, "_open", fake_open)
    return calls, state


def test_lab_relays_status_to_the_configured_service(client, lab):
    """The page asks this server; this server asks the service. Same address reader as
    the ciel adapter, so the UI and the predictions cannot disagree about where the
    head lives."""
    calls, _ = lab
    test_client, _ = client
    response = test_client.get("/lab/api/status")
    assert response.status_code == 200
    assert response.json() == {"ready": True}
    assert calls[0]["url"] == f"{server_module.ciel.base_url()}/api/status"
    assert response.headers["content-type"].startswith("application/json")


def test_lab_forwards_the_query_string(client, lab):
    calls, _ = lab
    test_client, _ = client
    test_client.get("/lab/api/label/queue?include_labeled=1")
    assert calls[0]["url"].endswith("/api/label/queue?include_labeled=1")


def test_lab_forwards_a_post_body_and_the_traffic_capture_opt_in(client, lab):
    """X-Decision-Lab-Log is what turns a call into training data, so the relay must
    carry it through rather than eating it as an unknown header."""
    calls, _ = lab
    test_client, _ = client
    test_client.post(
        "/lab/api/label",
        json={"item_id": "i1", "gold_idx": 1, "source": "accepted"},
        headers={"X-Decision-Lab-Log": "1"},
    )
    assert calls[0]["headers"]["x-decision-lab-log"] == "1"
    assert calls[0]["method"] == "POST"
    assert b"item_id" in calls[0]["body"]


def test_lab_drops_the_headers_the_transport_owns(client, lab):
    """REGRESSION-shaped: forwarding Host would point the upstream request at *this*
    server, and a forwarded Content-Length would contradict the body urllib sends.
    accept-encoding goes too — bodies are relayed verbatim, so a gzip response would
    arrive without the header needed to decode it."""
    calls, _ = lab
    test_client, _ = client
    test_client.post(
        "/lab/api/label", content=b"{}",
        headers={"content-type": "application/json", "accept-encoding": "gzip, deflate"},
    )
    assert "host" not in calls[0]["headers"]
    assert "content-length" not in calls[0]["headers"]
    assert "accept-encoding" not in calls[0]["headers"]
    assert calls[0]["headers"]["content-type"] == "application/json"


def test_lab_passes_an_upstream_error_through_unchanged(client, lab):
    """The service's own 4xx is information the labeling UI shows verbatim. Turning it
    into a relay error would replace \"no pending items to label\" with a shrug."""
    calls, state = lab
    state["response"] = urllib.error.HTTPError(
        "http://127.0.0.1:8000/api/label/auto", 400, "Bad Request", {},
        io.BytesIO(b'{"detail":"no pending items to label"}'),
    )
    test_client, _ = client
    response = test_client.post("/lab/api/label/auto", json={})
    assert response.status_code == 400
    assert "no pending items to label" in response.text


def test_lab_maps_an_unreachable_service_to_a_502_naming_the_address(client, lab):
    """502, not 500: the caller's request was fine, the thing behind it was not. The
    address is in the detail because that is the one fact that makes it fixable."""
    calls, state = lab
    state["response"] = urllib.error.URLError(ConnectionRefusedError("Connection refused"))
    test_client, _ = client
    response = test_client.get("/lab/api/status")
    assert response.status_code == 502
    assert server_module.ciel.base_url() in response.json()["detail"]


def test_lab_streams_the_auto_label_batch_for_real(client, lab):
    """One progress event per labeled item is only worth anything if it arrives while
    the batch is still running, so an SSE response must not be buffered."""
    calls, state = lab
    state["response"] = _relay_response(
        b'event: progress\ndata: {"ok": true}\n\n', content_type="text/event-stream",
    )
    test_client, _ = client
    response = test_client.get("/lab/api/label/auto-all")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in response.text
    assert response.headers.get("x-accel-buffering") == "no"


def test_lab_relays_the_pages_own_assets(client, lab):
    """The embedded UI loads its CSS and JS from the service. Without those coming
    through too, the frame arrives unstyled and inert."""
    calls, state = lab
    state["response"] = _relay_response(b".tab{color:red}", content_type="text/css")
    test_client, _ = client
    response = test_client.get("/lab/static/style.css")
    assert response.status_code == 200
    assert calls[0]["url"].endswith("/static/style.css")
    assert response.headers["content-type"].startswith("text/css")


def test_lab_opens_the_label_view_without_touching_the_upstream_checkout(client, lab):
    """The page is embedded on its Label view. Doing that rewrite in the relay keeps
    this repo the only place that knows about it, so an upstream checkout whose app.js
    predates `?tab=` still lands on the view this page exists to show."""
    calls, state = lab
    state["response"] = _relay_response(
        b"<html><head><title>Decision Lab</title></head><body></body></html>",
        content_type="text/html; charset=utf-8",
    )
    test_client, _ = client
    response = test_client.get("/lab/")
    assert response.status_code == 200
    assert 'set("tab", "label")' in response.text
    assert calls[0]["url"].endswith("/")
    # The upstream UI is light-styled; without this a dark-mode browser paints its
    # controls dark against its own light background.
    assert "<style>:root{color-scheme:light}</style>" in response.text


def test_lab_leaves_an_explicit_view_alone(client, lab):
    """The injected script checks before rewriting, and so does the relay — a redirect
    on every load would be a reload loop."""
    calls, state = lab
    state["response"] = _relay_response(b"<head></head>", content_type="text/html")
    test_client, _ = client
    response = test_client.get("/lab/?tab=eval")
    assert 'set("tab", "label")' not in response.text


def test_lab_does_not_rewrite_a_page_that_cannot_take_it(client, lab):
    """A body with no head is returned untouched: a working page beats a \"rewritten\"
    one whose scripts never ran."""
    calls, state = lab
    state["response"] = _relay_response(b"<html>no head here</html>", content_type="text/html")
    test_client, _ = client
    assert test_client.get("/lab/").text == "<html>no head here</html>"


def test_lab_ui_answers_with_and_without_a_trailing_slash(client, lab):
    """Both spellings, and neither by redirect: the frame's own reload and a hand-typed
    URL disagree about the slash, and a redirect inside a frame is a needless flash."""
    calls, state = lab
    state["response"] = _relay_response(b"<head></head>", content_type="text/html")
    test_client, _ = client
    for url in ("/lab", "/lab/"):
        response = test_client.get(url, follow_redirects=False)
        assert response.status_code == 200, f"{url} did not serve directly"


def test_lab_works_while_the_laya_router_is_cold(client, lab, monkeypatch):
    """Labeling must not wait on the local model — the same property jev, ciel and
    /api/transcribe have, and the reason this takes no predict_lock either."""
    calls, _ = lab
    test_client, _ = client
    monkeypatch.setattr(server_module._RouterState, "router", None)
    assert test_client.get("/lab/api/status").status_code == 200
