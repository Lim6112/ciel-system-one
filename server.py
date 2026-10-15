"""FastAPI server exposing the laya router to the web UI.

Serves static/index.html and a POST /api/predict endpoint that takes
{state, questions, model} and returns the laya prediction result. The router
is preloaded once at startup (see laya_startup.py and README "Startup
performance").

Run with: uv run uvicorn server:app

`model="jev"` swaps in TypeSafe's Jev through OpenRouter instead. Jev shares laya's
typed-question interface and its answer shape (see jev.py), so this endpoint's contract
and every client are unchanged — the only visible differences are latency, a reported
cost, and the absence of laya's `act_probability`.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from laya_startup import build_router

import jev

import ciel

import asr

QuestionType = Literal["choice", "score", "noul"]
ModelOverride = Literal[
    "auto", "english", "multilingual", "typed-decisions", "jev", "ciel_decision_model_v1"
]

STATIC_DIR = Path(__file__).parent / "static"


class Question(BaseModel):
    """One typed question, matching laya's question schema.

    choice -> criteria is a {label: description} dict; score -> criteria is a
    [labels] list; noul takes no criteria.
    """

    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: Any = None

    @field_validator("criteria")
    @classmethod
    def check_criteria(cls, value: Any, info) -> Any:
        qtype = info.data.get("type")
        if qtype == "choice":
            if not isinstance(value, dict) or not value:
                raise ValueError("choice questions need a non-empty criteria object")
            if not all(isinstance(k, str) and k for k in value):
                raise ValueError("choice criteria keys must be non-empty strings")
        elif qtype == "score":
            if not isinstance(value, list) or not value:
                raise ValueError("score questions need a non-empty criteria array")
        elif value is not None:
            raise ValueError("noul questions take no criteria")
        return value


class PredictRequest(BaseModel):
    state: dict[str, Any] = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)
    model: ModelOverride = "auto"


class _RouterState:
    router: Any = None
    # FastAPI runs sync endpoints in a threadpool, so concurrent requests can
    # call router.predict() at once — which crashes MPS/Metal ("a command
    # encoder is already encoding to this command buffer"). Serialize them.
    predict_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _RouterState.router = build_router()
    yield


app = FastAPI(title="CJet web UI", lifespan=lifespan)


def _stringify_state(state: dict[str, Any]) -> dict[str, str]:
    """laya accepts arbitrary state values; coerce them to strings for the API contract."""
    return {k: str(v) for k, v in state.items()}


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    questions = {
        name: q.model_dump(exclude_none=True) for name, q in request.questions.items()
    }
    start = time.perf_counter()

    # Jev is a network call that never touches the local model, so it deliberately does
    # NOT take predict_lock (that lock exists because concurrent encoder calls crash
    # MPS/Metal) and deliberately does not 503 when the laya router is still loading —
    # it does not need the router, only the key.
    if request.model == "jev":
        try:
            result = jev.predict(_stringify_state(request.state), questions)
        except jev.JevError as exc:
            raise HTTPException(
                status_code=502 if exc.status else 500,
                detail=f"jev failed: {exc}" + (f" — {exc.body[:400]}" if exc.body else ""),
            ) from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "result": result,
            "latency_ms": latency_ms,
            "routing": {"model": "jev", "repo": jev.JEV_MODEL, "reason": "explicit model=jev"},
        }

    # Ciel is also a network call to a separate service (the local decision_lab
    # dynamic head), so it likewise skips predict_lock and never 503s on a cold
    # laya router — it does not need the router, only the service up.
    if request.model == "ciel_decision_model_v1":
        try:
            result = ciel.predict(_stringify_state(request.state), questions)
        except ciel.CielError as exc:
            raise HTTPException(
                status_code=502 if exc.status else 500,
                detail=f"ciel failed: {exc}" + (f" — {exc.body[:400]}" if exc.body else ""),
            ) from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "result": result,
            "latency_ms": latency_ms,
            "routing": {
                "model": "ciel_decision_model_v1",
                "repo": ciel.SERVICE_NAME,
                "reason": "explicit model=ciel_decision_model_v1",
            },
        }

    router = _RouterState.router
    if router is None:
        raise HTTPException(status_code=503, detail="router is still loading")
    try:
        with _RouterState.predict_lock:
            result = router.predict(
                _stringify_state(request.state),
                questions,
                **({"model": request.model} if request.model != "auto" else {}),
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return {"result": result, "latency_ms": latency_ms}


# A coarse ceiling on the request body, so a runaway client cannot make the server
# buffer something enormous. Deliberately well ABOVE what asr accepts (a 30s clip is
# ~960 KB): this guard is about memory, and the sample-accurate length check belongs in
# asr._wav_to_float32 where the real limits live. Tightening this to match would make
# the two limits fight and produce a less specific error for the same input.
MAX_AUDIO_BYTES = 4 * 1024 * 1024


def _content_length(request: Request) -> int | None:
    """The declared Content-Length, or None when absent or unusable.

    None rather than an error for a malformed header: the caller re-checks the real body
    length anyway, so a bad header is not worth failing the request over — it just means
    the cheap pre-check cannot run.
    """
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@app.post("/api/transcribe")
async def transcribe_audio(
    request: Request,
    lang: str = Query(default=asr.DEFAULT_LANG),
) -> dict[str, Any]:
    """Transcribe a raw PCM WAV body into text.

    The body is the WAV itself rather than multipart or base64: the browser already has
    the bytes, and every encoding step is somewhere for a wrapper to go wrong.

    Like the jev and ciel branches, this touches no laya router — so it takes no
    predict_lock and does not 503 while the router is still loading. It is the one
    endpoint that also does not need the network, unless the checkpoint is not cached yet.
    """
    # Normalized rather than compared literally: asr itself folds case and surrounding
    # whitespace (see its `model_id`), and rejecting "EN" here would make the endpoint
    # stricter than the module behind it for no benefit.
    if (lang or "").strip().lower() not in asr.MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported lang {lang!r}; expected one of {asr.available_languages()}",
        )

    # Check the declared length BEFORE reading. `await request.body()` buffers the whole
    # thing, so a size test after it does not bound memory at all — it just rejects after
    # the damage. This pre-check is what makes the limit real.
    declared = _content_length(request)
    if declared is not None and declared > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio too large: {declared} bytes (max {MAX_AUDIO_BYTES})",
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=422, detail="empty audio body")
    # A chunked upload declares no length, so the guard runs again on what actually
    # arrived. Belt and braces: the pre-check stops the large case cheaply, this one stops
    # the case that lied about it.
    if len(body) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio too large: {len(body)} bytes (max {MAX_AUDIO_BYTES})",
        )

    start = time.perf_counter()
    try:
        # asr.transcribe is synchronous and CPU-bound, so it is run in the threadpool the
        # same way the sync predict endpoint is — via a def call, not by blocking the loop.
        result = await run_in_threadpool(asr.transcribe, body, lang)
    except asr.AsrError as exc:
        # 502 for a load/upstream failure (status set), 400 for a bad-body failure. The
        # distinction is what lets the UI say "check your connection" vs "that recording
        # was unusable" instead of showing one unhelpful error for both.
        raise HTTPException(
            status_code=exc.status or 400,
            detail=f"transcription failed: {exc}",
        ) from exc

    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ready": _RouterState.router is not None}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")