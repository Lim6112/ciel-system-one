"""Local speech recognition for the penguin demo, with no new dependencies.

WHY MOONSHINE AND NOT WHISPER. The repo already depends on `transformers` and `torch`,
and transformers ships Moonshine — so this adds a model but not a package. On the same
speech Moonshine tiny is both faster and more accurate than Whisper tiny (34ms vs 277ms
on an M-series Mac, 12.00% vs 12.81% WER), which for a command demo answered out loud is
the whole ballgame. Both models are ~27M parameters; Moonshine loads in well under a
second once cached.

Two languages, two checkpoints, because Moonshine's Korean model is a separate release:

    en -> UsefulSensors/moonshine-tiny     (MIT)
    ko -> UsefulSensors/moonshine-tiny-ko  (NOT MIT — see the README limitations note)

WHY THERE IS NO AUDIO DECODER HERE. The browser resamples to 16 kHz mono and hands us a
WAV it assembled itself, so the only format that ever arrives is plain PCM WAV — which
the stdlib `wave` module reads. No ffmpeg, no librosa, no soundfile, no torchaudio, no
`av`. Those are all heavy dependencies for one decode, and every one of them is a thing
that can fail to build.

The one real hazard is the offline flag: `laya_startup.py` sets HF_HUB_OFFLINE=1 so laya
never reaches for the network, and that flag would also stop the first Moonshine download
dead. `_hub_online()` lifts it around the load and restores it after; see the comment
there for why that is not simply "unset it globally".
"""

from __future__ import annotations

import io
import os
import threading
import wave
from typing import Any, Dict, Tuple

import numpy as np

# Model per language. `tiny` throughout: this is a four-way command classifier, and the
# larger checkpoints would buy accuracy the keyword fallback in penguin_commands.js
# already covers, at a download cost the demo cannot justify.
MODELS: Dict[str, str] = {
    "en": "UsefulSensors/moonshine-tiny",
    "ko": "UsefulSensors/moonshine-tiny-ko",
}
DEFAULT_LANG = "en"

# The rate the browser resamples to and the models expect. Asserted on decode rather than
# resampled here — a wrong rate means the client sent the wrong thing, and silently
# fixing it would hide that.
SAMPLE_RATE = 16000

# Below this there is no speech to find; transcribing a click returns hallucinated text.
MIN_SAMPLES = SAMPLE_RATE // 10          # 100ms
# Above this the model is being asked to do something it was not trained for, and the
# cost is unbounded. The UI caps recordings well under this.
MAX_SAMPLES = SAMPLE_RATE * 30           # 30s


class AsrError(RuntimeError):
    """A recognition failure the caller should surface rather than swallow.

    Shaped like `CielError` so the server can branch on it the same way: `status` is set
    for an upstream/loading failure and None for a bad request.
    """

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


# ---- the offline flag ----------------------------------------------------------------

_HUB_ENV = "HF_HUB_OFFLINE"


class _hub_online:
    """Temporarily clear HF_HUB_OFFLINE so a first-time model download can happen.

    WHY NOT JUST DELETE IT AT IMPORT. `laya_startup` sets it deliberately: laya's weights
    are expected to be cached, and reaching for the network mid-prediction is a latency
    bug it wants to fail loudly instead. Clearing it process-wide would re-arm that bug in
    the module that is not ours to change. Scoping it to the load is the narrow fix.

    The restore is in a finally block and puts back the ORIGINAL value, not merely
    "unset" — if the flag was set to something else by a parent process, this does not
    quietly downgrade it. Note this mutates the process environment, so it is only safe
    around a load; `_LOAD_LOCK` keeps concurrent loads from interleaving the window.
    """

    def __init__(self) -> None:
        self._saved: str | None = None
        self._was_set = False

    def __enter__(self) -> "_hub_online":
        self._saved = os.environ.get(_HUB_ENV)
        self._was_set = _HUB_ENV in os.environ
        os.environ.pop(_HUB_ENV, None)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._was_set:
            os.environ[_HUB_ENV] = self._saved if self._saved is not None else "1"
        else:
            os.environ.pop(_HUB_ENV, None)


# ---- WAV decoding --------------------------------------------------------------------

def _wav_to_float32(wav_bytes: bytes) -> np.ndarray:
    """Decode a PCM WAV into mono float32 samples in [-1, 1].

    A pure function, so the format handling is testable without loading a model.

    Stereo is averaged rather than rejected: the browser asks for one channel but some
    devices hand back two regardless, and that is not a reason to fail a working mic.
    Anything else — a compressed codec, a nonsense header, the wrong sample rate — is
    raised, because each of those means the client is not sending what this expects.
    """
    if not wav_bytes:
        raise AsrError("empty audio body")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        # EOFError is NOT a subclass of wave.Error, which is easy to miss and was a real
        # bug: a body truncated inside the 44-byte header (or a bare b"RIFF") makes the
        # stdlib reader raise EOFError, which sailed past a `except wave.Error` and
        # surfaced as an unhandled 500 on an endpoint documented to answer 400.
        #
        # ValueError is not caught and should not be: it would mean a bug in this
        # function's own arguments, not a bad upload.
        raise AsrError(f"not a readable WAV: {exc or 'truncated header'}") from exc

    if width != 2:
        raise AsrError(f"expected 16-bit PCM, got {width * 8}-bit")
    if rate != SAMPLE_RATE:
        raise AsrError(f"expected {SAMPLE_RATE} Hz, got {rate} Hz")
    if channels < 1:
        raise AsrError("WAV declares no channels")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        # Trim to a whole number of frames first: a truncated final frame would otherwise
        # misalign the reshape and mix channels into each other.
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    if samples.size < MIN_SAMPLES:
        raise AsrError(f"audio too short: {samples.size} samples")
    if samples.size > MAX_SAMPLES:
        raise AsrError(f"audio too long: {samples.size} samples")
    return samples


# ---- model loading and inference ------------------------------------------------------

_LOAD_LOCK = threading.Lock()
# lang -> (processor, model). Loaded once per language and kept for the process lifetime.
_CACHE: Dict[str, Tuple[Any, Any]] = {}


def available_languages() -> list[str]:
    return sorted(MODELS)


def model_id(lang: str) -> str:
    """The checkpoint for a language, or a clear error naming what is supported."""
    key = (lang or DEFAULT_LANG).strip().lower()
    if key not in MODELS:
        raise AsrError(f"unsupported language {lang!r}; expected one of {available_languages()}")
    return MODELS[key]


def is_loaded(lang: str) -> bool:
    return (lang or DEFAULT_LANG).strip().lower() in _CACHE


def _load(lang: str) -> Tuple[Any, Any]:
    """Load (and cache) the processor and model for a language.

    Double-checked locking: the fast path is a dict read, and only a genuine miss takes
    the lock. Without this two concurrent requests on a cold cache would both load, which
    on a 27M model is wasteful but — more importantly — would interleave two
    `_hub_online` windows over one shared environment variable.
    """
    key = (lang or DEFAULT_LANG).strip().lower()
    model_id(key)                                   # validates before taking the lock
    if key in _CACHE:
        return _CACHE[key]

    with _LOAD_LOCK:
        if key in _CACHE:                           # another thread won the race
            return _CACHE[key]
        # Imported here, not at module scope: transformers pulls in torch, and the WAV
        # path above is worth keeping importable without that cost.
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        repo = MODELS[key]
        try:
            with _hub_online():
                processor = AutoProcessor.from_pretrained(repo)
                model = MoonshineForConditionalGeneration.from_pretrained(repo)
        except Exception as exc:                    # network, cache, or a bad checkpoint
            raise AsrError(
                f"could not load {repo}: {exc}. The first run downloads the model; "
                f"check connectivity, or pre-cache it with huggingface-cli download {repo}"
            ) from exc

        model.eval()
        _CACHE[key] = (processor, model)
        return _CACHE[key]


def transcribe(wav_bytes: bytes, lang: str = DEFAULT_LANG) -> Dict[str, Any]:
    """Transcribe a PCM WAV. Returns the text plus what produced it.

    Returns a dict rather than a bare string so the server can report the model and the
    audio duration without re-deriving them, and so a caller can log which language arm
    answered.
    """
    samples = _wav_to_float32(wav_bytes)
    processor, model = _load(lang)

    # torch is imported lazily for the same reason as in `_load`.
    import torch

    inputs = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        # No sampling and a bounded decode: this is a four-way classification disguised as
        # transcription, and a hallucination loop is the known failure mode of a small
        # encoder-decoder on near-silence. The cap converts that from a hang into a
        # truncated answer.
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=64)

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return {
        "transcript": text.strip(),
        "model": model_id(lang),
        "lang": (lang or DEFAULT_LANG).strip().lower(),
        "duration_s": round(samples.size / SAMPLE_RATE, 2),
    }
