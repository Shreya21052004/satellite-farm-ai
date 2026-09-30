import os
import io
import base64
from typing import Any, Optional, Union

import requests
from sarvamai import SarvamAI


class SarvamSpeechError(RuntimeError):
    pass


def _client() -> SarvamAI:
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise SarvamSpeechError("Missing SARVAM_API_KEY environment variable.")
    return SarvamAI(api_subscription_key=key)


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return {"type": str(type(obj)), "repr": repr(obj)}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return obj
    # last resort
    return {"type": str(type(obj)), "repr": repr(obj)}


def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _maybe_b64_decode(s: str) -> Optional[bytes]:
    """
    Try base64 decode. Returns bytes if valid base64 else None.
    """
    try:
        cleaned = "".join(s.split())
        # heuristic: too short is probably not audio
        if len(cleaned) < 100:
            return None
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _find_audio_anywhere(obj: Any) -> Optional[bytes]:
    """
    Recursive best-effort:
    - if bytes found -> return
    - if URL string found -> download and return
    - if base64 string found -> decode and return
    """
    if obj is None:
        return None

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    # string: url or base64
    if isinstance(obj, str):
        if _looks_like_url(obj):
            r = requests.get(obj, timeout=60)
            r.raise_for_status()
            return r.content
        decoded = _maybe_b64_decode(obj)
        if decoded:
            return decoded
        return None

    # dict: walk values
    if isinstance(obj, dict):
        # prioritize likely keys first
        priority_keys = [
            "audio", "audio_bytes", "audio_base64", "audio_b64", "audioUrl", "audio_url",
            "url", "file_url", "data", "output", "result", "response", "payload", "audios"
        ]
        for k in priority_keys:
            if k in obj:
                got = _find_audio_anywhere(obj.get(k))
                if got:
                    return got

        for v in obj.values():
            got = _find_audio_anywhere(v)
            if got:
                return got
        return None

    # list/tuple: walk items
    if isinstance(obj, (list, tuple)):
        for item in obj:
            got = _find_audio_anywhere(item)
            if got:
                return got
        return None

    # pydantic model: dump and recurse
    if hasattr(obj, "model_dump"):
        return _find_audio_anywhere(_to_dict(obj))

    # unknown object: try its dict
    if hasattr(obj, "__dict__"):
        return _find_audio_anywhere(vars(obj))

    return None


# ---------- STT ----------
def _extract_text_any(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return _extract_text_any(_to_dict(obj))

    if isinstance(obj, dict):
        for k in ("text", "transcript", "transcription"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # nested
        for k in ("output", "data", "result", "results", "response", "payload", "segments"):
            v = obj.get(k)
            if isinstance(v, dict):
                t = _extract_text_any(v)
                if t:
                    return t
            elif isinstance(v, list):
                parts = []
                for it in v:
                    t = _extract_text_any(it)
                    if t:
                        parts.append(t)
                if parts:
                    return " ".join(parts).strip()

        return None

    for attr in ("text", "transcript", "transcription"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for attr in ("results", "segments", "outputs"):
        v = getattr(obj, attr, None)
        if isinstance(v, list) and v:
            parts = []
            for it in v:
                t = _extract_text_any(it)
                if t:
                    parts.append(t)
            if parts:
                return " ".join(parts).strip()

    return None


def speech_to_text_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    model: str = "saaras:v3",
    mode: str = "transcribe",
    debug: bool = False,
) -> str:
    if not audio_bytes:
        raise SarvamSpeechError("Empty audio bytes provided to STT.")

    client = _client()
    bio = io.BytesIO(audio_bytes)
    bio.name = filename  # type: ignore[attr-defined]

    resp = client.speech_to_text.transcribe(file=bio, model=model, mode=mode)
    text = _extract_text_any(resp)

    if not text:
        if debug:
            raise SarvamSpeechError(f"STT debug response: {_to_dict(resp)}")
        raise SarvamSpeechError("Could not extract transcript text from STT response.")

    return text


# ---------- TTS ----------
def text_to_speech_bytes(text: str, target_language_code: str, debug: bool = False) -> bytes:
    if not text or not text.strip():
        raise SarvamSpeechError("Empty text provided to TTS.")

    client = _client()
    resp = client.text_to_speech.convert(text=text, target_language_code=target_language_code)

    dumped = _to_dict(resp)
    audio_bytes = _find_audio_anywhere(dumped)

    if audio_bytes:
        return audio_bytes

    # At this point, no audio was found anywhere in the response.
    # Raise a debug payload that you can copy-paste.
    if debug:
        raise SarvamSpeechError(f"TTS debug response (no audio found): {dumped}")

    raise SarvamSpeechError("Could not extract audio bytes from TTS response.")