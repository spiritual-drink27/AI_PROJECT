"""Stage S1 (data part): load BASIL and group headlines by same-event story.

BASIL layout (https://github.com/launchnlp/BASIL):
    articles/<year>/<triplet-uuid>_<n>.json          n in {1,2,3}
    annotations/<year>/<triplet-uuid>_<n>_ann.json   matching gold annotations

One "triplet" = one news event reported by Fox, NYT and HuffPost.
The _1/_2/_3 suffix does NOT tell you the outlet; the "source" field does.

Run:   python -m src.load            (reads data/raw/BASIL, writes data/processed/)
Use:   from src.load import load_stories, headline_pairs
"""

import argparse
import csv
import glob
import json
import os
import re
from itertools import combinations

SOURCES = ("fox", "nyt", "hpo")          # fixed order used everywhere downstream
SOURCE_NAMES = {"fox": "Fox News", "nyt": "New York Times", "hpo": "HuffPost"}

RAW_DIR = os.path.join("data", "raw", "BASIL")
OUT_DIR = os.path.join("data", "processed")

# Outlet names sometimes appended to titles, e.g. "... | Fox News"
_SUFFIX = re.compile(r"\s*[|\-–—]\s*(Fox News|The New York Times|HuffPost|Huffington Post)\s*$", re.I)
# Datelines at the start of a lead sentence, e.g. "WASHINGTON — "
_DATELINE = re.compile(r"^[A-Z][A-Z .,'\-]{2,40}\s*[—–-]\s+")


def clean_text(text):
    """Collapse whitespace; keep the original wording (it is our evidence)."""
    return re.sub(r"\s+", " ", text or "").strip()


def make_headline(article):
    """Return (headline, from_title). Falls back to the first sentence if no title."""
    title = _SUFFIX.sub("", clean_text(article.get("title")))
    if title:
        return title, True
    for para in article.get("body-paragraphs", []):
        for sent in para:
            sent = clean_text(sent)
            if sent:
                return _DATELINE.sub("", sent), False
    return "", False


def _locate_span(sentences, ann):
    """BASIL offsets are relative to ONE sentence, but the sentence is not named.
    Find the sentence where text[start:end] == txt; fall back to a plain search."""
    txt, start, end = ann.get("txt", ""), ann.get("start"), ann.get("end")
    for s in sentences:
        if isinstance(start, int) and s["text"][start:end] == txt:
            return s["para"], s["sent"], start, end
    for s in sentences:
        pos = s["text"].find(txt) if txt else -1
        if pos >= 0:
            return s["para"], s["sent"], pos, pos + len(txt)
    return None, None, None, None


def load_article(path, ann_path=None):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    sentences = []
    for p_i, para in enumerate(raw.get("body-paragraphs", [])):
        for s_i, sent in enumerate(para):
            # keep raw text (no cleaning) so BASIL character offsets stay valid
            sentences.append({"para": p_i, "sent": s_i, "text": sent})

    headline, from_title = make_headline(raw)
    article = {
        "uuid": raw["uuid"],
        "source": raw["source"].lower(),
        "outlet": SOURCE_NAMES.get(raw["source"].lower(), raw["source"]),
        "headline": headline,
        "headline_from_title": from_title,
        "date": raw.get("date"),
        "url": raw.get("url"),
        "sentences": sentences,
        "bias_spans": [],
        "stance": None,
    }

    if ann_path and os.path.exists(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)
        article["stance"] = ann.get("article-level-annotations", {}).get("relative_stance")
        for a in ann.get("phrase-level-annotations", []):
            para, sent, start, end = _locate_span(sentences, a)
            article["bias_spans"].append({
                "bias": a.get("bias"),            # "lex" (wording) or "inf" (information)
                "polarity": a.get("polarity"),    # "pos" / "neg"
                "target": a.get("target"),
                "text": a.get("txt"),
                "para": para, "sent": sent, "start": start, "end": end,
                "quote": a.get("quote"),
            })
    return article, raw


def build_stories(raw_dir=RAW_DIR):
    """Return a list of stories, each with exactly one article per outlet."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "articles", "*", "*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No BASIL articles under {raw_dir}. Run scripts/get_data.sh first.")

    stories = {}
    for path in paths:
        year = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]          # <uuid>_<n>
        ann_path = os.path.join(raw_dir, "annotations", year, stem + "_ann.json")
        article, raw = load_article(path, ann_path)

        sid = raw["triplet-uuid"]
        story = stories.setdefault(sid, {
            "story_id": sid,
            "year": int(year),
            "main_event": clean_text(raw.get("main-event")),
            "main_entities": raw.get("main-entities", []),
            "articles": {},
        })
        if article["source"] in story["articles"]:
            raise ValueError(f"Story {sid} has two {article['source']} articles")
        story["articles"][article["source"]] = article

    out = sorted(stories.values(), key=lambda s: (s["year"], s["story_id"]))
    bad = [s["story_id"] for s in out if set(s["articles"]) != set(SOURCES)]
    if bad:
        raise ValueError(f"Stories missing an outlet: {bad}")
    return out


def headline_pairs(story):
    """The 3 headline pairs of a story in a fixed order:
    (fox, nyt), (fox, hpo), (nyt, hpo). Each rule/baseline compares these."""
    return [(story["articles"][a], story["articles"][b]) for a, b in combinations(SOURCES, 2)]


def load_stories(path=os.path.join(OUT_DIR, "stories.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stories_by_id(stories):
    return {s["story_id"]: s for s in stories}


def write_outputs(stories, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stories.json"), "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False, indent=1)

    # Flat table: handy for retrieval (TF-IDF), baselines and annotation sheets.
    with open(os.path.join(out_dir, "headlines.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_id", "year", "source", "outlet", "headline",
                    "headline_from_title", "main_event", "date", "url"])
        for s in stories:
            for src in SOURCES:
                a = s["articles"][src]
                w.writerow([s["story_id"], s["year"], src, a["outlet"], a["headline"],
                            a["headline_from_title"], s["main_event"], a["date"], a["url"]])


def summary(stories):
    arts = [a for s in stories for a in s["articles"].values()]
    spans = [sp for a in arts for sp in a["bias_spans"]]
    located = sum(sp["sent"] is not None for sp in spans)
    return {
        "stories": len(stories),
        "articles": len(arts),
        "headlines_from_first_sentence": sum(not a["headline_from_title"] for a in arts),
        "bias_spans": len(spans),
        "lexical_spans": sum(sp["bias"] == "lex" for sp in spans),
        "informational_spans": sum(sp["bias"] == "inf" for sp in spans),
        "spans_located_in_text": located,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=RAW_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    stories = build_stories(args.raw)
    write_outputs(stories, args.out)
    for k, v in summary(stories).items():
        print(f"{k:32s} {v}")
    print(f"\nWrote {args.out}/stories.json and {args.out}/headlines.csv")


if __name__ == "__main__":
    main()
