"""Offline tests for Stage S3. They use a simple word-overlap "embedder" in place of
Sentence-BERT, so they need no model download.  Run:  python -m tests.test_align
"""

import csv
import json
import os
import tempfile

import numpy as np

import src.align as A


class WordEmbedder:
    """Bag-of-words vectors: cosine = word overlap. Stand-in for Sentence-BERT."""

    def __init__(self):
        self.vocab = {}

    def encode(self, texts):
        rows = []
        for t in texts:
            v = {}
            for w in t.lower().split():
                v[self.vocab.setdefault(w, len(self.vocab))] = 1.0
            rows.append(v)
        m = np.zeros((len(texts), max(len(self.vocab), 1)))
        for i, v in enumerate(rows):
            for k, x in v.items():
                m[i, k] = x
        return m


def ext(actors=(), items=(), outcomes=(), ok=True):
    return {"ok": ok, "fields": {
        "actors": [{"name": n, "type": "Person", "span": n} for n in actors],
        "items": [{"text": t, "span": t} for t in items],
        "agents": [], "evals": [],
        "outcomes": [{"text": t, "type": "Safety", "span": t} for t in outcomes]}}


TH = {"items": 0.6, "actors": 0.75, "outcomes": 0.6}


def test_basic_alignment():
    story = {
        "fox": ext(["Police", "protesters"], ["fire tear gas", "city faces curfew"]),
        "nyt": ext(["protesters"], ["use tear gas", "protest turns violent"]),
        "hpo": ext(["police officers"], ["tear gas fired"]),
    }
    al = A.align_story(story, WordEmbedder(), TH)
    assert al["ok"] and al["missing"] == []
    ids = {(m["src"], m["label"]): m["id"] for m in al["mentions"]["items"]}
    # all three tear-gas phrasings share one ID; the others stay separate
    assert ids[("fox", "fire tear gas")] == ids[("nyt", "use tear gas")] == ids[("hpo", "tear gas fired")]
    assert len({ids[("fox", "city faces curfew")], ids[("nyt", "protest turns violent")],
                ids[("fox", "fire tear gas")]}) == 3
    assert A.id_of(al, "items", "fox", "fire tear gas") == "i1"           # reading order
    # actors: exact name, and name containment ("police" in "police officers")
    act = {(m["src"], m["label"]): m["id"] for m in al["mentions"]["actors"]}
    assert act[("fox", "protesters")] == act[("nyt", "protesters")]
    assert act[("fox", "Police")] == act[("hpo", "police officers")]
    assert any(m["by"] == "name-match" for m in al["matches"]["actors"])


def test_one_to_one_and_no_same_headline_merge():
    # fox has two different tear-gas items; nyt has one. Only ONE may match it.
    story = {"fox": ext(items=["fire tear gas", "tear gas canisters"]),
             "nyt": ext(items=["tear gas fired"]),
             "hpo": ext(items=["tear gas canisters found"])}
    al = A.align_story(story, WordEmbedder(), {"items": 0.5})
    by_src = {}
    for m in al["mentions"]["items"]:
        by_src.setdefault(m["id"], []).append(m["src"])
    for cid, srcs in by_src.items():
        assert len(srcs) == len(set(srcs)), f"cluster {cid} merged one headline's items: {srcs}"


def test_threshold_and_missing_headline():
    story = {"fox": ext(items=["fire tear gas"]), "nyt": ext(items=["use tear gas"]),
             "hpo": ext(ok=False)}
    loose = A.align_story(story, WordEmbedder(), {"items": 0.6})
    strict = A.align_story(story, WordEmbedder(), {"items": 0.9})
    assert loose["ok"] and loose["missing"] == ["hpo"]
    assert len(loose["clusters"]["items"]) == 1
    assert len(strict["clusters"]["items"]) == 2                          # 0.67 < 0.9: not merged
    one = A.align_story({"fox": story["fox"], "nyt": ext(ok=False), "hpo": ext(ok=False)},
                        WordEmbedder(), TH)
    assert not one["ok"]


def test_names():
    assert A.name_contained("Steele", "Michael Steele")
    assert A.name_contained("Michael Steele’s", "Steele")
    assert A.name_contained("Rep. Grimm", "Michael Grimm")
    assert not A.name_contained("Obama", "Romney")
    assert not A.name_contained("the", "President")                        # only stop words


def test_tune_and_params():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pairs.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["story_id", "kind", "src_a", "a", "src_b", "b", "sim", "same"])
            for sim, same in [(0.9, "y"), (0.8, "y"), (0.72, "y"), (0.66, "n"), (0.5, "n"), (0.3, "")]:
                w.writerow(["s", "items", "fox", "x", "nyt", "y", sim, same])
        old = A.OUT_DIR, A.PARAMS_PATH
        A.OUT_DIR, A.PARAMS_PATH = d, os.path.join(d, "params.json")
        try:
            best = A.tune(path)
            assert 0.66 < best["items"] <= 0.72, best
            assert A.load_thresholds()["items"] == best["items"]
            assert A.load_thresholds()["actors"] == A.DEFAULT_THRESHOLDS["actors"]
        finally:
            A.OUT_DIR, A.PARAMS_PATH = old


def test_run_all_on_file():
    with tempfile.TemporaryDirectory() as d:
        ep = os.path.join(d, "ext.json")
        with open(ep, "w") as f:
            json.dump({"s1": {"fox": ext(items=["fire tear gas"]), "nyt": ext(items=["use tear gas"]),
                              "hpo": ext(items=["curfew imposed"])}}, f)
        old = A.EXTRACTIONS_PATH
        A.EXTRACTIONS_PATH = ep
        try:
            out = os.path.join(d, "alignment.json")
            A.run_all(WordEmbedder(), TH, out_path=out)
            al = A.load_alignment(out)["s1"]
            assert len(al["clusters"]["items"]) == 2
        finally:
            A.EXTRACTIONS_PATH = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All S3 checks passed.")
