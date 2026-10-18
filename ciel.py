"""Ciel decision client, shaped to look exactly like laya.

`ciel_decision_model_v1` is a trained dynamic decision head served by the local
"decision_lab" HTTP service: one POST to /api/decide-dynamic answers any set of
typed questions — `{type, instructions, criteria}` — about a text state. This module
exists so that swap is invisible to everything downstream: `predict()` returns the
same envelope `Agent.system_one()` returns, so the browser and the A/B harness read
`result.answers.<qid>` either way and neither has a branch on who answered.

Two differences from jev, both handled here rather than in the callers:

- The service takes a single `custom_text` string, while laya's state is a dict of
  fields. Joined here as the bare values, space-separated, in insertion order —
  NOT "key: value" lines. The head embeds the state text and scores options
  against it, and it was trained on prose whose field names live inside the
  sentences; a "situation:" / "bricks:" prefix is out-of-distribution and
  measurably flips correct answers to garbage (confirmed against the live
  service: identical state, prefix-only difference, right → wrong).
- Upstream answers come back as an array parallel to the request questions, not
  keyed by qid. The positional zip happens here, so the envelope stays keyed.

No API key — the service is local and free, so (unlike jev) there is no `cost_usd`.
Uses stdlib urllib rather than adding an HTTP dependency for one POST.

The service defaults to port 8000, which is also `uvicorn server:app`'s default;
point DECISION_LAB_URL elsewhere if both run on one machine (see README).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping

MODEL_NAME = "ciel_decision_model_v1"
SERVICE_NAME = "decision_lab/dynamic-head"
URL_ENV = "DECISION_LAB_URL"
DEFAULT_URL = "http://127.0.0.1:8000"
DECIDE_PATH = "/api/decide-dynamic"
# Sub-path selection. The service serves several heads off one backbone and
# routes on the request's `path`; without this the request resolves to
# web.default_path ("doc") and a problem's own head is unreachable — silently,
# since the service only 400s on an *unknown* path, never a missing one.
PATH_ENV = "DECISION_LAB_PATH"

# Model overrides that select a problem-specific head rather than the doc bank.
#
# Every name here must correspond to a key in the service's `web.paths`. `typed`
# was served but missing from this map for a while, which made it unreachable:
# a request that omits `path` resolves to the service's default and 400s only on
# an *unknown* path, never a missing one, so the omission was silent — the exact
# failure the resolve_path docstring below warns about.
MODEL_PATHS = {
    "ciel_warehouse_v1": "warehouse",
    "ciel_typed_v1": "typed",
}


def resolve_path(model: str | None = None) -> str | None:
    """Sub-path for this call: explicit model mapping, then env, then None.

    None means "let the service use web.default_path" — the behaviour every
    existing caller got, so nothing that ran before this needs to change.
    """
    if model and model in MODEL_PATHS:
        return MODEL_PATHS[model]
    path = os.environ.get(PATH_ENV, "").strip()
    return path or None
# A local inference round trip is fast, but the service also runs an embedding
# backbone; 30s is generous but bounded, so a hung upstream surfaces as an error
# the UI can show rather than a request that never returns.
DEFAULT_TIMEOUT = 30.0


class CielError(RuntimeError):
    """An upstream failure, carrying enough of the response to diagnose it."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def base_url() -> str:
    url = os.environ.get(URL_ENV, DEFAULT_URL).strip()
    return url.rstrip("/")


def health(timeout: float = 5.0) -> bool:
    """True once the service reports ready — used by the live smoke check."""
    try:
        with urllib.request.urlopen(f"{base_url()}/api/status", timeout=timeout) as response:
            return json.load(response).get("ready") is True
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return False


def _state_to_custom_text(state: Mapping[str, Any]) -> str:
    """Serialize the state as JSON — the same string the head trained on.

    This replaced a bare space-joined-values encoding, and the reason is worth
    keeping: the head was trained on one format and asked about another. Every
    documented failure of this adapter has been exactly that, so the fix is to
    stop having two formats rather than to keep tuning one of them.

    The earlier "join the values, drop the field names" rule came from serving a
    *prose* head, where a "situation:" / "bricks:" prefix was out of
    distribution and measurably flipped answers. That head no longer exists: the
    warehouse corpus (decision_lab states/warehouse.py) renders flat JSON whose
    keys are the field names, and split_state_fields() recovers those keys as
    its leaf set on the way in — so the keys are the signal now, not noise.

    Flat and scalar only, deliberately. server.py's _stringify_state calls str()
    on each top-level value, so a nested object arrives as a Python repr; a list
    is rendered by _flatten_json as one joined string. Either way the head is
    shown text no training state contains.
    """
    return json.dumps(dict(state), ensure_ascii=False)


def _to_lab_questions(questions: Mapping[str, Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[str]]:
    """Translate laya questions to the service's schema, keeping the qid order.

    Returns:
        (lab_questions, qids) — parallel lists, so answers can be zipped back by
        position.
    """
    lab_questions: list[Dict[str, Any]] = []
    qids: list[str] = []
    for qid, question in questions.items():
        qtype = question.get("type")
        lab: Dict[str, Any] = {"type": qtype, "question": question.get("instructions", "")}
        if qtype == "choice":
            criteria = question.get("criteria") or {}
            # dict insertion order preserves the option order; the model only
            # sees the labels, so the descriptions are dropped here.
            lab["options"] = [str(option) for option in criteria.keys()]
        elif qtype == "score":
            lab["levels"] = [str(level) for level in question.get("criteria") or []]
        elif qtype != "noul":
            raise CielError(f"unsupported question type {qtype!r} for qid {qid!r}")
        lab_questions.append(lab)
        qids.append(qid)
    return lab_questions, qids


def _probabilities_by_index(
    distribution: Mapping[str, Any], levels: list[str], predicted: int
) -> Dict[str, float]:
    """Remap a {level_label: prob} distribution to {str(index): prob}."""
    probabilities: Dict[str, float] = {}
    for label, prob in distribution.items():
        if label in levels:
            probabilities[str(levels.index(label))] = float(prob)
        elif label.isdigit():
            probabilities[label] = float(prob)
    if not probabilities:
        # Distribution keys matched nothing we sent — fall back to the argmax.
        probabilities[str(predicted)] = 1.0
    return probabilities


def _from_lab_answer(
    question: Mapping[str, Any], answer: Mapping[str, Any]
) -> Dict[str, Any]:
    """Map one service answer to laya's type-specific answer shape."""
    qtype = question.get("type")
    confidence = float(answer.get("confidence") or 0.0)
    if qtype == "noul":
        distribution = answer.get("distribution") or {}
        p_true = float(distribution["true"]) if "true" in distribution else (
            1.0 if answer.get("predicted") else 0.0
        )
        return {"type": "noul", "noul": p_true, "confidence": confidence}
    if qtype == "score":
        levels = [str(level) for level in question.get("criteria") or []]
        predicted = int(answer.get("predicted") or 0)
        expected = answer.get("expected")
        return {
            "type": "score",
            "score": float(expected) if expected is not None else float(predicted),
            "legend": {str(i): level for i, level in enumerate(levels)},
            "probabilities": _probabilities_by_index(
                answer.get("distribution") or {}, levels, predicted
            ),
            "confidence": confidence,
        }
    # choice (and anything else with label-shaped answers)
    distribution = answer.get("distribution") or {}
    return {
        "type": "choice",
        "choice": str(answer.get("predicted", "")),
        "probabilities": {str(label): float(prob) for label, prob in distribution.items()},
        "confidence": confidence,
    }


def _from_lab_answers(
    qids: list[str],
    questions: Mapping[str, Dict[str, Any]],
    lab_answers: list[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if len(lab_answers) != len(qids):
        raise CielError(
            f"Decision Lab returned {len(lab_answers)} answers for {len(qids)} questions",
            body=json.dumps(lab_answers)[:500],
        )
    return {
        qid: _from_lab_answer(questions[qid], answer)
        for qid, answer in zip(qids, lab_answers)
    }


def predict(
    state: Mapping[str, Any],
    questions: Mapping[str, Dict[str, Any]],
    timeout: float = DEFAULT_TIMEOUT,
    model: str | None = None,
) -> Dict[str, Any]:
    """Ask the decision_lab service, and return laya's result envelope.

    Args:
        model: the caller's model override. It is not sent upstream (the
            service has no such field) but it selects the sub-path, so a
            caller that asked for a problem-specific head gets it.

    Returns:
        {"model", "provider", "answers", "usage"} — the same shape
        `laya.Agent.system_one` produces, minus cost (the service is free).
    """
    lab_questions, qids = _to_lab_questions(questions)
    payload = {
        "custom_text": _state_to_custom_text(state),
        "questions": lab_questions,
    }
    path = resolve_path(model)
    if path is not None:
        payload["path"] = path
    request = urllib.request.Request(
        base_url() + DECIDE_PATH,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # Opt this client into Decision Lab's traffic capture so its calls
            # can be labeled and used as real training data. The server only
            # logs when web.log_traffic is also enabled.
            "X-Decision-Lab-Log": "1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise CielError(f"Decision Lab request failed ({exc.code})", status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise CielError(f"could not reach Decision Lab: {exc.reason}") from exc

    answers = _from_lab_answers(qids, questions, body.get("answers") or [])
    return {
        "model": MODEL_NAME,
        "provider": "decision_lab",
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
