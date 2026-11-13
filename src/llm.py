"""Tiny LLM client shared by extraction (S2) and the direct-LLM baseline (B3).

Uses the OpenAI-compatible "chat completions" API, so the same code works with:
  * Groq   (default, free tier, no card):  https://api.groq.com/openai/v1
  * Gemini (backup):  https://generativelanguage.googleapis.com/v1beta/openai
  * OpenAI or any other compatible provider.

Settings come from environment variables or a .env file in the repo root:
  LLM_API_KEY   your key (GROQ_API_KEY is also accepted)
  LLM_BASE_URL  default https://api.groq.com/openai/v1
  LLM_MODEL     default openai/gpt-oss-120b

Standard library only: no SDK to install.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMError(RuntimeError):
    pass


def load_dotenv(path=".env"):
    """Read KEY=VALUE lines from .env into os.environ (without overriding)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_json(text):
    """Pull a JSON object out of a model reply, tolerating ```json fences or extra prose."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=60, max_retries=5):
        load_dotenv()
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise LLMError("No API key. Put LLM_API_KEY=... in a .env file (see .env.example).")

    def _post(self, payload):
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "cs-f407-framing/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, system, user, json_mode=True, temperature=0.0, max_tokens=1500):
        """Send one prompt; return the reply text. Retries on rate limits (HTTP 429)
        and server errors with back-off."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"     # fewer hidden tokens -> fits free-tier limits

        delay = 5
        for attempt in range(self.max_retries):
            try:
                data = self._post(payload)
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code == 429 or e.code >= 500:
                    wait = float(e.headers.get("retry-after") or delay)
                    print(f"  [llm] HTTP {e.code}, waiting {wait:.0f}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    delay = min(delay * 2, 60)
                    continue
                if e.code == 400 and "response_format" in payload:
                    payload.pop("response_format")       # model without JSON mode: retry plain
                    payload.pop("reasoning_effort", None)
                    continue
                raise LLMError(f"HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  [llm] network error {e}, retrying in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise LLMError("Gave up after repeated rate-limit/network errors.")

    def chat_json(self, system, user, **kw):
        text = self.chat(system, user, json_mode=True, **kw)
        try:
            return parse_json(text), text
        except (json.JSONDecodeError, ValueError):
            raise LLMError(f"Model did not return valid JSON: {text[:200]!r}") from None
