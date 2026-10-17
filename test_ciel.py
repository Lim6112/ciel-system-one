"""Unit tests for the ciel adapter's mapping helpers (no network, no model).

The mapping between laya's question/answer shapes and decision_lab's
positional-answers API is where this integration can silently drift, so these
tests pin it directly.

uv run pytest test_ciel.py
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

import ciel


CHOICE_QUESTION = {
    "type": "choice",
    "instructions": "Which department?",
    "criteria": {"billing": "payments", "other": "everything else"},
}
SCORE_QUESTION = {
    "type": "score",
    "instructions": "urgency",
    "criteria": ["low", "medium", "high"],
}
NOUL_QUESTION = {"type": "noul", "instructions": "Is this actionable?"}


def test_state_to_custom_text_serializes_json():
    """The state goes upstream as JSON — the same string the head trained on.

    This replaced a bare space-joined-values encoding. Both the old rule and
    this one exist for the same reason: the head was trained on one format and
    asked about another, and every documented failure of this adapter has been
    exactly that. The current corpus (decision_lab states/warehouse.py) renders
    flat JSON whose keys are the field names, so the keys are signal now.
    """
    assert ciel._state_to_custom_text({"b": 2, "a": 1}) == '{"b": 2, "a": 1}'


def test_state_to_custom_text_keeps_key_order():
    """Key order is load-bearing: it decides the leaf order the field splitter
    produces, and the head attends over that set."""
    assert ciel._state_to_custom_text({"role": "robot", "color": "red"}) == (
        '{"role": "robot", "color": "red"}')


def test_state_to_custom_text_is_flat_json():
    """Flat and scalar only: server.py str()s each top-level value, so a nested
    object would be sent as a Python repr the head has never seen."""
    text = ciel._state_to_custom_text({"n": 3, "s": "x"})
    parsed = json.loads(text)
    assert all(isinstance(v, (str, int, float, bool)) for v in parsed.values())


def test_resolve_path_defaults_to_none():
    """No mapping and no env var => None, i.e. the service's own default head.
    This is what every caller got before sub-paths existed."""
    assert ciel.resolve_path(None) is None
    assert ciel.resolve_path("ciel_decision_model_v1") is None


def test_resolve_path_maps_the_warehouse_model():
    assert ciel.resolve_path("ciel_warehouse_v1") == "warehouse"


def test_resolve_path_honors_env(monkeypatch):
    monkeypatch.setenv("DECISION_LAB_PATH", "warehouse")
    assert ciel.resolve_path(None) == "warehouse"
    # an explicit model mapping still wins over the env default
    monkeypatch.setenv("DECISION_LAB_PATH", "typed")
    assert ciel.resolve_path("ciel_warehouse_v1") == "warehouse"


def test_to_lab_questions_maps_each_type():
    lab, qids = ciel._to_lab_questions({
        "dept": CHOICE_QUESTION,
        "urgency": SCORE_QUESTION,
        "actionable": NOUL_QUESTION,
    })
    assert qids == ["dept", "urgency", "actionable"]
    # dict insertion order must survive into the options list
    assert lab[0] == {
        "type": "choice", "question": "Which department?",
        "options": ["billing", "other"],
    }
    assert lab[1] == {"type": "score", "question": "urgency", "levels": ["low", "medium", "high"]}
    assert lab[2] == {"type": "noul", "question": "Is this actionable?"}


def test_to_lab_questions_rejects_unknown_type():
    with pytest.raises(ciel.CielError, match="essay"):
        ciel._to_lab_questions({"q": {"type": "essay", "instructions": "write"}})


def test_from_lab_answers_choice():
    answers = ciel._from_lab_answers(
        ["dept"], {"dept": CHOICE_QUESTION},
        [{"predicted": "other",
          "distribution": {"billing": 0.3, "other": 0.7}, "confidence": 0.7}],
    )
    assert answers["dept"] == {
        "type": "choice", "choice": "other",
        "probabilities": {"billing": 0.3, "other": 0.7}, "confidence": 0.7,
    }


def test_from_lab_answers_score_remaps_labels_to_indices():
    answers = ciel._from_lab_answers(
        ["urgency"], {"urgency": SCORE_QUESTION},
        [{"predicted": 2, "expected": 1.73,
          "distribution": {"low": 0.05, "medium": 0.27, "high": 0.68},
          "confidence": 0.68}],
    )
    answer = answers["urgency"]
    assert answer == {
        "type": "score", "score": 1.73,
        "legend": {"0": "low", "1": "medium", "2": "high"},
        "probabilities": {"0": 0.05, "1": 0.27, "2": 0.68},
        "confidence": 0.68,
    }


def test_from_lab_answers_score_falls_back_to_predicted_index():
    answers = ciel._from_lab_answers(
        ["urgency"], {"urgency": SCORE_QUESTION},
        [{"predicted": 1, "confidence": 0.5}],
    )
    answer = answers["urgency"]
    assert answer["score"] == 1.0
    assert answer["probabilities"] == {"1": 1.0}


def test_from_lab_answers_noul_uses_true_probability():
    answers = ciel._from_lab_answers(
        ["q"], {"q": NOUL_QUESTION},
        [{"predicted": True, "distribution": {"true": 0.91, "false": 0.09},
          "confidence": 0.91}],
    )
    assert answers["q"] == {"type": "noul", "noul": 0.91, "confidence": 0.91}


def test_from_lab_answers_noul_without_distribution():
    answers = ciel._from_lab_answers(
        ["q"], {"q": NOUL_QUESTION}, [{"predicted": False, "confidence": 0.6}]
    )
    assert answers["q"] == {"type": "noul", "noul": 0.0, "confidence": 0.6}


def test_from_lab_answers_rejects_count_mismatch():
    with pytest.raises(ciel.CielError, match="1 answers for 2 questions"):
        ciel._from_lab_answers(
            ["a", "b"], {"a": NOUL_QUESTION, "b": NOUL_QUESTION},
            [{"predicted": True, "confidence": 0.5}],
        )


def test_base_url_honors_env(monkeypatch):
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:9000/")
    assert ciel.base_url() == "http://127.0.0.1:9000"


class _LabHandler(http.server.BaseHTTPRequestHandler):
    """A canned decision_lab: echoes the request, returns one score answer."""

    def do_GET(self):
        payload = json.dumps({"ready": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _LabHandler.received = json.loads(self.rfile.read(length))
        payload = json.dumps({
            "answers": [{"predicted": 1, "expected": 1.4,
                         "distribution": {"low": 0.2, "medium": 0.6, "high": 0.2},
                         "confidence": 0.6}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture()
def lab_server(monkeypatch):
    server = http.server.HTTPServer(("127.0.0.1", 0), _LabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DECISION_LAB_URL", f"http://127.0.0.1:{server.server_port}")
    yield server
    server.shutdown()


def test_predict_round_trip(lab_server):
    assert ciel.health() is True
    result = ciel.predict({"body": "hello"}, {"urgency": SCORE_QUESTION})
    assert _LabHandler.received == {
        "custom_text": '{"body": "hello"}',
        "questions": [{"type": "score", "question": "urgency",
                       "levels": ["low", "medium", "high"]}],
    }
    assert result["model"] == "ciel_decision_model_v1"
    assert result["answers"]["urgency"]["probabilities"] == {"0": 0.2, "1": 0.6, "2": 0.2}


def test_predict_sends_the_sub_path(lab_server):
    """A problem-specific head is only reachable if the path goes upstream.
    Without it the service resolves to web.default_path — silently, since an
    unknown path 400s but a missing one does not."""
    ciel.predict({"body": "hello"}, {"urgency": SCORE_QUESTION},
                 model="ciel_warehouse_v1")
    assert _LabHandler.received["path"] == "warehouse"


def test_predict_omits_path_when_unset(lab_server):
    """No mapping => no path key at all, so existing callers keep hitting the
    service's default head."""
    ciel.predict({"body": "hello"}, {"urgency": SCORE_QUESTION})
    assert "path" not in _LabHandler.received


def test_predict_reports_an_unreachable_service(monkeypatch):
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:1")
    with pytest.raises(ciel.CielError, match="could not reach Decision Lab"):
        ciel.predict({"body": "hi"}, {"q": NOUL_QUESTION})


def test_health_false_when_unreachable(monkeypatch):
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:1")
    assert ciel.health() is False
