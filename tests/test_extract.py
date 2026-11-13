"""Offline tests for Stage S2 (no API key or internet needed).
Run:  python -m tests.test_extract
"""

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src import llm
from src.extract import Extractor, find_span, select_stories, validate
from src.llm import LLMClient, LLMError, parse_json
from src.load import load_stories

H = "Police fire tear gas on peaceful protesters as city faces curfew"


# ---------------------------------------------------------------- validation

def test_validate():
    raw = {
        "actors": [{"name": "police", "type": "LawEnforcement", "span": "Police"},
                   {"name": "protesters", "type": "Martian", "span": "protesters"},
                   {"name": "mayor", "type": "Politician", "span": "Mayor"}],          # not in headline
        "items": [{"text": "fire tear gas", "span": "fire tear gas"}],
        "agents": [{"actor": "police", "action": "fire tear gas", "span": "Police fire tear gas"},
                   {"actor": "protesters", "action": "march", "span": "protesters"},  # action not listed
                   {"actor": "army", "action": "fire tear gas", "span": "fire"}],     # unknown actor
        "evals": [{"target": "protesters", "polarity": "positive", "word": "peaceful"},
                  {"target": "police", "polarity": "negative", "word": "brutal"},     # invented word
                  {"target": "police", "polarity": "mixed", "word": "fire"}],         # bad polarity
        "outcomes": [{"text": "curfew", "type": "Safety", "span": "faces curfew"}],
    }
    f, warn = validate(H, raw)
    assert [a["name"] for a in f["actors"]] == ["police", "protesters"]
    assert f["actors"][1]["type"] == "Other"                       # unknown type mapped
    assert {i["text"] for i in f["items"]} == {"fire tear gas", "march"}
    assert [(a["actor"], a["action"]) for a in f["agents"]] == [("police", "fire tear gas"),
                                                               ("protesters", "march")]
    assert [e["word"] for e in f["evals"]] == ["peaceful"]
    assert f["outcomes"][0]["type"] == "Safety"
    assert H[f["evals"][0]["start"]:f["evals"][0]["end"]] == "peaceful"
    assert len(warn) == 4, warn


def test_spans_and_quotes():
    h = "Michael Steele’s Afghanistan Comments Stir Up Furious Reaction"
    s, e = find_span(h, "Steele's afghanistan comments")        # straight quote, other case
    assert h[s:e] == "Steele’s Afghanistan Comments"
    assert find_span(h, "calm reaction") is None
    assert validate(h, "not a dict")[0]["items"] == []


def test_parse_json():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}


# ---------------------------------------------------------------- extractor with a fake model

class FakeClient:
    model = "fake-model"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def chat_json(self, system, user):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply, json.dumps(self.reply)


def test_extractor_cache_and_errors():
    good = {"actors": [{"name": "police", "type": "LawEnforcement", "span": "Police"}],
            "items": [{"text": "fire tear gas", "span": "fire tear gas"}],
            "agents": [], "evals": [], "outcomes": []}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        fake = FakeClient(good)
        ex = Extractor(client=fake, cache_path=path)
        r1 = ex.extract(H)
        r2 = ex.extract(H)
        assert r1["ok"] and r2["ok"] and fake.calls == 1             # second call served by cache
        ex.save()
        ex2 = Extractor(client=FakeClient(error=AssertionError("must not be called")), cache_path=path)
        assert ex2.extract(H)["ok"]                                   # cache persisted to disk

    # model returns nothing usable -> clear message, no crash
    empty = Extractor(client=FakeClient({"items": [{"text": "x", "span": "not here"}]}),
                      cache_path=os.devnull + "x")
    r = empty.extract(H)
    assert not r["ok"] and "Could not extract" in r["error"]

    # API failure -> clear message, no crash, not cached as success
    broken = Extractor(client=FakeClient(error=LLMError("HTTP 401: bad key")), cache_path="/nonexistent")
    r = broken.extract(H)
    assert not r["ok"] and "bad key" in r["error"]
    assert not Extractor(client=FakeClient(good), cache_path="/nonexistent").extract("   ")["ok"]


def test_select_stories():
    stories = load_stories()
    assert len(select_stories(stories, "annotated")) == 20
    assert len(select_stories(stories, "dev")) == 6
    assert len(select_stories(stories, "all")) == 100


# ---------------------------------------------------------------- HTTP client vs a fake server

class FakeAPI(BaseHTTPRequestHandler):
    script = []           # list of (status, body, headers) returned in order
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.seen.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
        status, payload, headers = FakeAPI.script.pop(0)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *a):
        pass


def _reply(content):
    return (200, {"choices": [{"message": {"content": content}}]}, {})


def test_http_client():
    srv = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    llm.time.sleep = lambda s: None                     # do not really wait in tests
    try:
        FakeAPI.seen, FakeAPI.script = [], [
            (429, {"error": "rate limit"}, {"retry-after": "1"}),       # rate limited once
            _reply('{"items": []}'),
        ]
        c = LLMClient(api_key="k", base_url=base, model="openai/gpt-oss-120b")
        data, _ = c.chat_json("sys", "user")
        assert data == {"items": []}
        req = FakeAPI.seen[-1]
        assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer k"
        assert req["body"]["response_format"] == {"type": "json_object"}
        assert req["body"]["reasoning_effort"] == "low"
        assert req["body"]["messages"][0]["role"] == "system"

        FakeAPI.seen, FakeAPI.script = [], [
            (400, {"error": "response_format not supported"}, {}),      # no JSON mode
            _reply('```json\n{"ok": true}\n```'),
        ]
        c = LLMClient(api_key="k", base_url=base, model="some-model")
        assert c.chat_json("s", "u")[0] == {"ok": True}
        assert "response_format" not in FakeAPI.seen[-1]["body"]

        FakeAPI.script = [(401, {"error": "invalid api key"}, {})]
        try:
            c.chat_json("s", "u")
            raise AssertionError("expected LLMError")
        except LLMError as e:
            assert "401" in str(e)
    finally:
        srv.shutdown()


def test_missing_key():
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "GROQ_API_KEY")}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)                                     # no .env here
        try:
            LLMClient()
            raise AssertionError("expected LLMError")
        except LLMError as e:
            assert ".env" in str(e)
        finally:
            os.chdir(cwd)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All S2 checks passed.")
