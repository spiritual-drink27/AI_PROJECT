"""Stage S2: LLM-based structured extraction (headline text -> structured fields).

For each headline the LLM returns five lists that later become logical facts (S4):
    actors    who appears              -> used by Agent/Eval atoms and the ontology (IsA)
    items     what is reported         -> Reports(h, i)
    agents    who is shown doing what  -> Agent(h, x, i)
    evals     loaded/evaluative words  -> Eval(h, x, pol, w)
    outcomes  consequences stressed    -> Outcome(h, k, t)

Every field must quote a "span" that literally occurs in the headline. validate()
drops anything the model invents, and records character offsets so the demo can
highlight the evidence. This is what keeps the explanations traceable (goal G2)
and stops extraction errors from turning into wrong explanations (goal G3).

Results are cached in data/extractions/cache.json, so each headline is sent to the
LLM only once and the demo works even if the API is down.

Run:
    python -m src.extract --split annotated   # 20 dev+test stories = 60 headlines (do first)
    python -m src.extract --split all         # all 300 headlines
    python -m src.extract --headline "Police fire tear gas on peaceful protesters"
"""

import argparse
import hashlib
import json
import os
import re
import time

from src.load import SOURCES, load_stories

PROMPT_VERSION = "v1"
CACHE_PATH = os.path.join("data", "extractions", "cache.json")
OUT_PATH = os.path.join("data", "extractions", "extractions.json")

# Fixed category lists. The ontology (S4, src/facts.py) must use the same names.
ACTOR_TYPES = ["Politician", "GovernmentOfficial", "GovernmentBody", "PoliticalParty",
               "Court", "LawEnforcement", "Military", "Country", "Organization",
               "Company", "Media", "Activist", "Citizen", "Person", "Other"]
OUTCOME_TYPES = ["Economic", "Safety", "Legal", "Political", "Health", "Social",
                 "Reputational", "Other"]
POLARITIES = ["positive", "negative"]

SYSTEM_PROMPT = f"""You extract structured facts from ONE news headline for a framing-analysis system.
Return ONLY a JSON object with exactly these keys:

"actors":   people, groups or institutions named or clearly referred to.
            [{{"name": short canonical name, "type": one of {ACTOR_TYPES}, "span": exact words}}]
"items":    the pieces of information the headline reports (events, actions, claims, facts),
            each as a short normalised phrase in present tense.
            [{{"text": e.g. "fire tear gas", "span": exact words}}]
"agents":   for each item where the headline SHOWS who performs it, link actor to item.
            [{{"actor": an actor name, "action": an item text, "span": exact words}}]
            If the actor is hidden (passive voice, "tear gas used"), add NO agent entry.
"evals":    loaded or evaluative words that judge an actor or item (e.g. "slams", "furious",
            "peaceful", "radical", "inaccurate"). Neutral reporting verbs ("says", "announces") are NOT evals.
            [{{"target": an actor name or item text, "polarity": "positive" or "negative", "word": exact words}}]
"outcomes": consequences or stakes the headline emphasises.
            [{{"text": short phrase, "type": one of {OUTCOME_TYPES}, "span": exact words}}]

Rules:
- Every "span" and "word" must be copied EXACTLY from the headline (a contiguous substring).
- Do not use outside knowledge; extract only what this headline says.
- Use an empty list when nothing fits. Keep lists short (usually 0-4 entries).

Example headline: "Police fire tear gas on peaceful protesters as city faces curfew"
{{"actors": [{{"name": "police", "type": "LawEnforcement", "span": "Police"}},
             {{"name": "protesters", "type": "Activist", "span": "protesters"}}],
 "items": [{{"text": "fire tear gas", "span": "fire tear gas"}},
           {{"text": "city faces curfew", "span": "city faces curfew"}}],
 "agents": [{{"actor": "police", "action": "fire tear gas", "span": "Police fire tear gas"}}],
 "evals": [{{"target": "protesters", "polarity": "positive", "word": "peaceful"}}],
 "outcomes": [{{"text": "curfew", "type": "Safety", "span": "faces curfew"}}]}}"""


# ---------------------------------------------------------------- validation

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-"})


def _norm(s):
    """Lower-case and straighten quotes/dashes. Keeps string length unchanged so
    offsets found in the normalised headline are valid in the original."""
    return (s or "").translate(_QUOTES).lower()


def find_span(headline, span):
    """Return (start, end) of span inside headline, or None if it does not occur."""
    span = re.sub(r"\s+", " ", (span or "")).strip()
    if not span:
        return None
    pos = _norm(headline).find(_norm(span))
    return (pos, pos + len(span)) if pos >= 0 else None


def _same(a, b):
    return _norm(a).strip() == _norm(b).strip()


def validate(headline, raw):
    """Clean the model's JSON. Returns (fields, warnings).
    Anything not grounded in the headline text is dropped and reported."""
    warnings = []
    raw = raw if isinstance(raw, dict) else {}

    def grounded(entry, key, kind):
        off = find_span(headline, entry.get(key))
        if off is None:
            warnings.append(f"dropped {kind}: {entry.get(key)!r} not found in headline")
            return None
        entry[key] = headline[off[0]:off[1]]           # use the headline's own spelling
        entry["start"], entry["end"] = off
        return entry

    def as_list(key):
        v = raw.get(key, [])
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    actors = []
    for a in as_list("actors"):
        a = {"name": str(a.get("name", "")).strip(), "type": a.get("type"), "span": a.get("span")}
        if not a["name"]:
            continue
        if a["type"] not in ACTOR_TYPES:
            a["type"] = "Other"
        if grounded(a, "span", "actor") and not any(_same(a["name"], x["name"]) for x in actors):
            actors.append(a)

    items = []
    for it in as_list("items"):
        it = {"text": str(it.get("text", "")).strip(), "span": it.get("span")}
        if it["text"] and grounded(it, "span", "item") and not any(_same(it["text"], x["text"]) for x in items):
            items.append(it)

    agents = []
    for ag in as_list("agents"):
        ag = {"actor": str(ag.get("actor", "")).strip(), "action": str(ag.get("action", "")).strip(),
              "span": ag.get("span")}
        if not grounded(ag, "span", "agent"):
            continue
        actor = next((x for x in actors if _same(x["name"], ag["actor"])), None)
        if actor is None:
            warnings.append(f"dropped agent: actor {ag['actor']!r} is not in actors")
            continue
        ag["actor"] = actor["name"]
        item = next((x for x in items if _same(x["text"], ag["action"])), None)
        if item is None:                      # model forgot to list the action as an item
            item = {"text": ag["action"] or ag["span"], "span": ag["span"],
                    "start": ag["start"], "end": ag["end"]}
            items.append(item)
        ag["action"] = item["text"]
        agents.append(ag)

    evals = []
    for ev in as_list("evals"):
        ev = {"target": str(ev.get("target", "")).strip(), "polarity": ev.get("polarity"),
              "word": ev.get("word")}
        if ev["polarity"] not in POLARITIES:
            warnings.append(f"dropped eval: bad polarity {ev['polarity']!r}")
            continue
        if not grounded(ev, "word", "eval"):
            continue
        known = [x["name"] for x in actors] + [x["text"] for x in items]
        match = next((k for k in known if _same(k, ev["target"])), None)
        ev["target"] = match or ev["target"]
        ev["target_known"] = match is not None
        evals.append(ev)

    outcomes = []
    for oc in as_list("outcomes"):
        oc = {"text": str(oc.get("text", "")).strip(), "type": oc.get("type"), "span": oc.get("span")}
        if oc["type"] not in OUTCOME_TYPES:
            oc["type"] = "Other"
        if grounded(oc, "span", "outcome"):
            outcomes.append(oc)

    fields = {"actors": actors, "items": items, "agents": agents,
              "evals": evals, "outcomes": outcomes}
    return fields, warnings


# ---------------------------------------------------------------- extraction + cache

def _key(model, headline):
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{headline}".encode("utf-8")).hexdigest()


class Extractor:
    def __init__(self, client=None, cache_path=CACHE_PATH):
        self._client = client
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    @property
    def client(self):
        if self._client is None:                      # create lazily: cached runs need no key
            from src.llm import LLMClient
            self._client = LLMClient()
        return self._client

    @property
    def model(self):
        if self._client is not None:
            return self._client.model
        from src.llm import DEFAULT_MODEL, load_dotenv
        load_dotenv()
        return os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.cache_path)

    def extract(self, headline, use_cache=True):
        """Return a result dict:
           {"ok": bool, "headline", "fields": {...}, "warnings": [...], "error": str|None}
        Never raises for model/API problems: the demo shows result["error"] instead."""
        headline = re.sub(r"\s+", " ", headline or "").strip()
        if not headline:
            return {"ok": False, "headline": headline, "fields": None, "warnings": [],
                    "error": "Empty headline."}
        key = _key(self.model, headline)
        if use_cache and key in self.cache and self.cache[key]["ok"]:
            return self.cache[key]

        from src.llm import LLMError
        result = {"ok": False, "headline": headline, "model": self.model,
                  "prompt_version": PROMPT_VERSION, "fields": None, "warnings": [],
                  "error": None, "raw": None}
        try:
            data, text = self.client.chat_json(SYSTEM_PROMPT, f'Headline: "{headline}"')
            result["raw"] = text
            fields, warnings = validate(headline, data)
            result["fields"], result["warnings"] = fields, warnings
            if not (fields["items"] or fields["actors"]):
                result["error"] = "Could not extract any facts from this headline."
            else:
                result["ok"] = True
        except LLMError as e:
            result["error"] = f"Extraction failed: {e}"
        except Exception as e:                        # never crash the demo
            result["error"] = f"Extraction failed: {type(e).__name__}: {e}"

        self.cache[key] = result
        return result


# ---------------------------------------------------------------- batch run over BASIL

def select_stories(stories, split):
    if split == "all":
        return stories
    with open(os.path.join("data", "splits.json")) as f:
        sp = json.load(f)
    ids = set(sp["dev"] + sp["test"]) if split == "annotated" else set(sp[split])
    return [s for s in stories if s["story_id"] in ids]


def run_batch(split, extractor, pause=0.0):
    stories = select_stories(load_stories(), split)
    out = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            out = json.load(f)

    todo = [(s, src) for s in stories for src in SOURCES]
    failed = 0
    for n, (s, src) in enumerate(todo, 1):
        headline = s["articles"][src]["headline"]
        was_cached = _key(extractor.model, headline) in extractor.cache
        res = extractor.extract(headline)
        out.setdefault(s["story_id"], {})[src] = {k: v for k, v in res.items() if k != "raw"}
        status = "ok " if res["ok"] else "ERR"
        print(f"[{n:3d}/{len(todo)}] {status} {src} {headline[:70]}"
              + ("" if res["ok"] else f"  -> {res['error']}"))
        failed += not res["ok"]
        if not was_cached:
            extractor.save()                      # save after every call: safe to Ctrl+C and resume
            if pause:
                time.sleep(pause)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nDone: {len(todo) - failed} ok, {failed} failed -> {OUT_PATH}")
    if failed:
        print("Re-run the same command to retry only the failed headlines.")


def load_extractions(path=OUT_PATH):
    """{story_id: {source: result}} for use by alignment (S3) and facts (S4)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Stage S2: structured extraction with an LLM")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["annotated", "dev", "test", "all"])
    g.add_argument("--headline", help="extract one headline and print the result")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds to wait between API calls (free-tier friendly)")
    args = ap.parse_args()

    ex = Extractor()
    if args.headline:
        res = ex.extract(args.headline)
        ex.save()
        print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2, ensure_ascii=False))
    else:
        run_batch(args.split, ex, pause=args.pause)


if __name__ == "__main__":
    main()
