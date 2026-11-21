"""Check which Google endpoint your GEMINI_API_KEY works with, and which Gemma models it can call.

Reads the key from .env / the environment and never prints it.

    uv run python scripts/check_gemini_key.py
"""

import _bootstrap  # noqa: F401
from google import genai

from sentineldesk import config

key = config.GEMINI_API_KEY
if not key:
    raise SystemExit("No key found: put GEMINI_API_KEY=... in .env first.")
print(f"key: {key[:4]}… ({len(key)} chars)")

ok_modes = []
for vertex, label in ((False, "Gemini API (AI Studio)"), (True, "Vertex AI express mode")):
    try:
        client = genai.Client(api_key=key, vertexai=vertex)
        resp = client.models.generate_content(model=config.GEMINI_GEMMA_MODEL,
                                              contents="Reply with the single word: ok")
        print(f"[OK]   {label}: {config.GEMINI_GEMMA_MODEL} answered {resp.text.strip()[:20]!r}")
        ok_modes.append(vertex)
        try:
            names = [m.name.split("/")[-1] for m in client.models.list() if "gemma" in m.name.lower()]
            if names:
                print("       Gemma models available:", ", ".join(sorted(names)))
        except Exception:
            pass
    except Exception as e:
        print(f"[FAIL] {label}: {str(e).splitlines()[0][:160]}")

if ok_modes == [True]:
    print("\n-> Add GOOGLE_GENAI_USE_VERTEXAI=true to .env so the app uses Vertex AI mode.")
elif ok_modes:
    print("\n-> Works with the default settings. Restart the app to use hosted Gemma.")
else:
    print("\n-> Key didn't work with either endpoint for this model. Check that the key's project "
          "has the API enabled, or try SD_GEMINI_MODEL=<one of the listed models>.")
