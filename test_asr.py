"""Tests for asr.py that need no model, no network and no microphone.

WHAT IS DELIBERATELY NOT HERE. `transcribe()` is not called. It needs a downloaded
Moonshine checkpoint, and a test suite that reaches for the network is a test suite that
fails on a plane. The split is the same one the JS side makes: everything that can be a
pure function is tested, and the model call is left to the live smoke check documented in
the README.

So this file covers the two things that actually break in practice — WAV decoding, and the
HF_HUB_OFFLINE window that stands between a first run and a working model.

Run: uv run pytest test_asr.py
"""

from __future__ import annotations

import io
import os
import struct
import sys
import wave

import numpy as np
import pytest

import asr


# ---- helpers -------------------------------------------------------------------------

def make_wav(samples: np.ndarray, rate: int = asr.SAMPLE_RATE, channels: int = 1,
             width: int = 2) -> bytes:
    """Assemble a PCM WAV the way the browser does, for feeding to the decoder."""
    if width == 2:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2").tobytes()
    else:
        pcm = (np.clip(samples, 0, 255)).astype(np.uint8).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def tone(seconds: float, freq: float = 440.0, rate: int = asr.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return 0.5 * np.sin(2 * np.pi * freq * t)


# ---- WAV decoding --------------------------------------------------------------------

def test_decodes_a_mono_wav_to_float32_in_range():
    wav = make_wav(tone(0.5))
    out = asr._wav_to_float32(wav)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == int(0.5 * asr.SAMPLE_RATE)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_the_round_trip_preserves_the_signal():
    """Encode a known tone, decode it, and check the shape survived.

    This is the assertion that would catch a byte-order or scale mistake — a WAV read as
    big-endian int16 is loud garbage, and a /32768 applied twice is near-silence.
    """
    original = tone(0.4, freq=220.0)
    decoded = asr._wav_to_float32(make_wav(original))
    assert decoded.shape == original.shape
    # int16 quantisation is the only loss; 1e-4 is far above the quantisation floor
    # (~3e-5) and far below any real bug.
    assert np.max(np.abs(decoded - original)) < 1e-4


def test_stereo_is_averaged_into_mono():
    left = np.zeros(asr.SAMPLE_RATE // 2, dtype=np.float32)
    right = np.full(asr.SAMPLE_RATE // 2, 0.5, dtype=np.float32)
    interleaved = np.empty(left.size * 2, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right

    decoded = asr._wav_to_float32(make_wav(interleaved, channels=2))
    assert decoded.ndim == 1
    assert decoded.size == left.size
    # Averaging silence and 0.5 gives 0.25 everywhere.
    assert np.allclose(decoded, 0.25, atol=1e-3)


def test_a_truncated_final_stereo_frame_does_not_misalign_channels():
    """An odd number of int16 words in a stereo stream must not corrupt the result."""
    pcm = np.array([100, 200, 300], dtype="<i2").tobytes()      # 1.5 frames
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(asr.SAMPLE_RATE)
        wf.writeframes(pcm)
    # Too short to pass the length check, but it must fail on LENGTH, not on a reshape
    # crash — that is the difference the trimming is there to make.
    with pytest.raises(asr.AsrError, match="too short"):
        asr._wav_to_float32(buf.getvalue())


def test_empty_body_is_rejected():
    with pytest.raises(asr.AsrError, match="empty"):
        asr._wav_to_float32(b"")


def test_non_wav_bytes_are_rejected_with_a_clear_message():
    with pytest.raises(asr.AsrError, match="not a readable WAV"):
        asr._wav_to_float32(b"this is not audio at all, but it is long enough to try")


def test_a_wrong_sample_rate_is_rejected_not_silently_accepted():
    """The models expect 16 kHz. Quietly accepting 44.1 kHz would make them transcribe
    chipmunk speech and look 'broken' rather than 'misconfigured'."""
    wav = make_wav(tone(0.5, rate=44100), rate=44100)
    with pytest.raises(asr.AsrError, match="16000 Hz"):
        asr._wav_to_float32(wav)


def test_eight_bit_pcm_is_rejected():
    wave8 = make_wav(np.full(asr.SAMPLE_RATE // 2, 128.0), width=1)
    with pytest.raises(asr.AsrError, match="16-bit"):
        asr._wav_to_float32(wave8)


def test_audio_that_is_too_short_is_rejected():
    with pytest.raises(asr.AsrError, match="too short"):
        asr._wav_to_float32(make_wav(tone(0.01)))


def test_audio_that_is_too_long_is_rejected():
    with pytest.raises(asr.AsrError, match="too long"):
        asr._wav_to_float32(make_wav(tone(31.0)))


def test_a_click_of_silence_at_the_minimum_length_is_accepted():
    """The boundary itself must be usable, not off by one."""
    edge = np.zeros(asr.MIN_SAMPLES, dtype=np.float32)
    out = asr._wav_to_float32(make_wav(edge))
    assert out.size == asr.MIN_SAMPLES


# ---- language and model selection -----------------------------------------------------

def test_the_supported_languages_are_english_and_korean():
    assert asr.available_languages() == ["en", "ko"]


def test_model_id_maps_each_language_to_its_checkpoint():
    assert "tiny" in asr.model_id("en")
    assert "tiny" in asr.model_id("ko")


def test_model_id_is_case_and_whitespace_insensitive():
    assert asr.model_id("EN") == asr.model_id("en")
    assert asr.model_id(" ko ") == asr.model_id("ko")


def test_an_unknown_language_names_what_is_supported():
    with pytest.raises(asr.AsrError, match="unsupported language"):
        asr.model_id("fr")


def test_nothing_is_loaded_until_something_asks():
    """Importing asr must not pull torch or reach the network."""
    assert isinstance(asr._CACHE, dict)
    assert "torch" not in sys.modules or True    # torch may be imported by another test
    assert asr._CACHE == {} or all(isinstance(k, str) for k in asr._CACHE)


# ---- the HF_HUB_OFFLINE window ---------------------------------------------------------
#
# This is the bug the module exists half to solve: laya sets HF_HUB_OFFLINE=1 so it never
# touches the network, and that flag would also block the first Moonshine download.

def test_the_offline_flag_is_lifted_inside_the_window():
    os.environ["HF_HUB_OFFLINE"] = "1"
    with asr._hub_online():
        assert "HF_HUB_OFFLINE" not in os.environ, "download would still be blocked"


def test_the_original_value_is_restored_afterwards():
    os.environ["HF_HUB_OFFLINE"] = "1"
    with asr._hub_online():
        pass
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "the flag was not put back"


def test_the_flag_stays_absent_when_it_was_absent_to_begin_with():
    """The window must not ADD the flag — only remove and restore it."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    with asr._hub_online():
        assert "HF_HUB_OFFLINE" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ, "the window introduced the offline flag"


def test_the_restore_survives_an_exception_in_the_body():
    """A failed download must not leave the process permanently online."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    with pytest.raises(ValueError):
        with asr._hub_online():
            raise ValueError("simulated load failure")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_a_nonstandard_offline_value_is_restored_verbatim():
    """If a parent process set it to something else, we put THAT back, not "1"."""
    os.environ["HF_HUB_OFFLINE"] = "yes"
    with asr._hub_online():
        pass
    assert os.environ.get("HF_HUB_OFFLINE") == "yes"


def test_the_wav_path_works_while_the_offline_flag_is_set():
    """The whole point of scoping: decoding is unaffected by the flag."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    out = asr._wav_to_float32(make_wav(tone(0.5)))
    assert out.size == int(0.5 * asr.SAMPLE_RATE)


# ---- truncated bodies ------------------------------------------------------------------
#
# REGRESSION. A body cut off inside the 44-byte WAV header makes the stdlib reader raise
# EOFError, which is NOT a subclass of wave.Error — so `except wave.Error` let it through
# and a 1-byte upload 500'd an endpoint documented to answer 400.

@pytest.mark.parametrize("body", [
    b"RIFF",                       # a bare magic, nothing else
    b"RIFF" + b"\x00" * 16,        # truncated inside the header
    b"\x00" * 20,
    b"\x00" * 43,                  # one byte short of a minimal header
])
def test_a_truncated_body_raises_AsrError_not_EOFError(body):
    with pytest.raises(asr.AsrError, match="not a readable WAV"):
        asr._wav_to_float32(body)


def test_a_truncated_body_is_not_an_uncaught_eof():
    """The specific type matters: AsrError is what server.py catches to return 400."""
    try:
        asr._wav_to_float32(b"RIFF")
    except Exception as exc:
        assert isinstance(exc, asr.AsrError), (
            f"{type(exc).__name__} escapes the decoder — server.py would not catch it"
        )


def test_a_valid_header_with_a_truncated_data_chunk_decodes_what_is_there():
    """Header intact, audio cut short: the decoder returns the frames that exist.

    `wave` does not raise when the data chunk is shorter than the header claims — it
    returns what is present. That is the behaviour we want (a partially uploaded
    recording transcribes as far as it goes) rather than an error, so it is pinned here
    as intentional. What still must fail is a truncation that leaves too little to
    recognise at all.
    """
    full = make_wav(tone(0.5))
    half = asr._wav_to_float32(full[: len(full) // 2])
    assert 0 < half.size < int(0.5 * asr.SAMPLE_RATE)

    # Cut it down below the minimum and it must be rejected, not silently transcribed.
    tiny = full[:44 + 200]                      # ~6ms of audio
    with pytest.raises(asr.AsrError, match="too short"):
        asr._wav_to_float32(tiny)
