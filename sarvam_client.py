import os
from sarvamai import SarvamAI

class SarvamError(RuntimeError):
    pass

# Keep one client instance (faster + avoids re-init on every call)
_client: SarvamAI | None = None

def _get_client() -> SarvamAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamError("Missing SARVAM_API_KEY env var.")

    # Sarvam SDK expects a subscription key
    _client = SarvamAI(api_subscription_key=api_key)
    return _client


def sarvam_translate(
    text: str,
    target_lang: str,
    source_lang: str = "en-IN",
    speaker_gender: str | None = None,
    model: str = "sarvam-translate:v1",
) -> str:
    """
    Translate text using Sarvam official SDK.

    Args:
      text: input text (<=2000 chars for sarvam-translate:v1)
      target_lang: e.g. "kn-IN", "hi-IN", "mr-IN"
      source_lang: e.g. "en-IN" or "auto"
      speaker_gender: optional "Male" / "Female" (improves translation sometimes)
      model: "sarvam-translate:v1" (recommended) or "mayura:v1"

    Returns:
      translated string
    """
    if not text:
        return text

    client = _get_client()

    kwargs = {"model": model}
    if speaker_gender:
        kwargs["speaker_gender"] = speaker_gender

    try:
        resp = client.text.translate(
            input=text,
            source_language_code=source_lang,
            target_language_code=target_lang,
            **kwargs,
        )
    except Exception as e:
        raise SarvamError(str(e))

    # SDK response usually has translated_text attribute
    translated = getattr(resp, "translated_text", None)

    # fallback if SDK returns dict-like
    if not translated and isinstance(resp, dict):
        translated = resp.get("translated_text")

    if not translated:
        raise SarvamError(f"Unexpected Sarvam response: {resp}")

    return translated