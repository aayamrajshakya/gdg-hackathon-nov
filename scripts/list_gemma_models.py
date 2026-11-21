"""List the Gemma models your GEMINI_API_KEY can call (pick one for SD_GEMINI_MODEL)."""

import _bootstrap  # noqa: F401
from google import genai

from sentineldesk import config

if not config.GEMINI_API_KEY:
    raise SystemExit("Set GEMINI_API_KEY (in .env or the environment) first.")
client = genai.Client(api_key=config.GEMINI_API_KEY)
for m in client.models.list():
    if "gemma" in m.name.lower():
        print(m.name.removeprefix("models/"), "-", m.display_name)
