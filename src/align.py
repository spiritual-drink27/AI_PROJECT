"""Stage S3: cross-headline alignment with Sentence-BERT.

Problem: the LLM writes the same information in different words in different
headlines ("fire tear gas" vs "use tear gas", "Steele" vs "Michael Steele").
The logic rules (S5) can only compare facts if equivalent things share ONE identifier.

Method, for each story and each kind of thing (items, actors, outcomes):
  1. Embed every mention with Sentence-BERT (all-MiniLM-L6-v2).
  2. For each pair of headlines, compute cosine similarity between their mentions.
  3. Greedy one-to-one matching: take the most similar pair first; accept it if
     similarity >= threshold and neither mention is already matched in that pair.
  4. Merge matched mentions into clusters (union-find). Two mentions from the SAME
     headline are never merged, so a headline keeps its distinct items distinct.
  5. Each cluster gets an ID: i1, i2... (items), a1... (actors), o1... (outcomes).

Actors also match when one name's words are contained in the other's
("Steele" ~ "Michael Steele"), because embeddings of bare names are unreliable.

The threshold is tuned on the 6 dev stories:
    python -m src.align --dump-pairs        # writes data/alignment/dev_pairs.csv
    (M3 fills the "same" column with y/n by hand)
    python -m src.align --tune              # prints precision/recall/F1 per threshold
Then align everything with the chosen threshold:
    python -m src.align --threshold 0.70

Known limitation (proposal, Section 7): embeddings ignore roles, so
"police shoot man" and "man shoots police" can look similar.
"""

import argparse
import csv
import json
import os
import re
from itertools import combinations

import numpy as np

from src.load import SOURCES

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLDS = {"items": 0.70, "actors": 0.75, "outcomes": 0.70}
KINDS = {"items": ("text", "i"), "actors": ("name", "a"), "outcomes": ("text", "o")}

EXTRACTIONS_PATH = os.path.join("data", "extractions", "extractions.json")
OUT_DIR = os.path.join("data", "alignment")
PARAMS_PATH = os.path.join(OUT_DIR, "params.json")

_STOP = {"the", "a", "an", "of", "rep", "sen", "gov", "mr", "mrs", "ms", "dr", "president", "u.s."}


# ---------------------------------------------------------------- embeddings

class SBERTEmbedder:
    """Sentence-BERT wrapper. The model (~90 MB) downloads on first use, then is cached."""

    def __init__(self, model_name=MODEL_NAME):
        from sentence_transformers import SentenceTransformer   # imported only when needed
        self.model = SentenceTransformer(model_name)
        self._cache = {}

    def encode(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            vecs = self.model.encode(new, normalize_embeddings=True, show_progress_bar=False)
            self._cache.update(zip(new, np.asarray(vecs, dtype=float)))
        return np.array([self._cache[t] for t in texts]) if texts else np.zeros((0, 1))


def cosine_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def name_tokens(name):
    """'Rep. Michael Grimm's' -> {'michael', 'grimm'}"""
    name = name.lower().replace("’", "'")
    name = re.sub(r"'s\b", "", name)                   # drop possessive 's
    toks = (t.strip(".") for t in re.findall(r"[a-z0-9.]+", name))
    return {t for t in toks if t and t not in _STOP}


def name_contained(a, b):
    """'Steele' vs 'Michael Steele' -> True; 'Trump' vs 'Trump administration' -> True."""
    ta, tb = name_tokens(a), name_tokens(b)
    return bool(ta and tb) and (ta <= tb or tb <= ta)


# ---------------------------------------------------------------- core alignment

class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def similarity_table(mentions, kind, embedder):
    """All cross-headline mention pairs with their similarity.
    mentions: list of dicts with 'src' and 'label' (text that gets embedded)."""
    vecs = embedder.encode([m["label"] for m in mentions])
    rows = []
    for sa, sb in combinations(SOURCES, 2):
        ia = [i for i, m in enumerate(mentions) if m["src"] == sa]
        ib = [i for i, m in enumerate(mentions) if m["src"] == sb]
        if not ia or not ib:
            continue
        sim = cosine_matrix(vecs[ia], vecs[ib])
        for x, i in enumerate(ia):
            for y, j in enumerate(ib):
                s = float(sim[x, y])
                rule = "embedding"
                if kind == "actors" and name_contained(mentions[i]["label"], mentions[j]["label"]):
                    s, rule = max(s, 1.0), "name-match"
                rows.append({"i": i, "j": j, "sim": round(s, 4), "by": rule})
    return rows


def align_mentions(mentions, kind, embedder, threshold):
    """Return (cluster_ids, matches). cluster_ids[k] is the ID string of mention k."""
    rows = similarity_table(mentions, kind, embedder)
    uf = _UnionFind(len(mentions))
    members = {k: {mentions[k]["src"]} for k in range(len(mentions))}   # sources per cluster root
    matched_in_pair = set()                                            # (mention, other source)
    matches = []

    for r in sorted(rows, key=lambda r: -r["sim"]):
        if r["sim"] < threshold:
            break
        i, j = r["i"], r["j"]
        si, sj = mentions[i]["src"], mentions[j]["src"]
        if (i, sj) in matched_in_pair or (j, si) in matched_in_pair:
            continue                                   # one-to-one per headline pair
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if members[ri] & members[rj]:
            continue                                   # would merge two mentions of one headline
        uf.union(ri, rj)                               # rj becomes the root
        members[rj] = members.pop(ri) | members[rj]
        matched_in_pair |= {(i, sj), (j, si)}
        matches.append(r)

    prefix = KINDS[kind][1]
    ids, order = {}, []
    for k in range(len(mentions)):                     # number clusters in reading order
        root = uf.find(k)
        if root not in ids:
            ids[root] = f"{prefix}{len(ids) + 1}"
        order.append(ids[root])
    return order, matches


def align_story(story_extractions, embedder, thresholds=None):
    """story_extractions: {src: extraction result from S2}.
    Returns {"ok", "missing", "mentions": {kind: [...]}, "clusters": {kind: {id: [...]}},
             "matches": {kind: [...]}} where every mention carries its cluster "id"."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    usable = {s: r for s, r in story_extractions.items() if r and r.get("ok")}
    out = {"ok": len(usable) >= 2, "missing": sorted(set(SOURCES) - set(usable)),
           "mentions": {}, "clusters": {}, "matches": {}}

    for kind, (label_key, _) in KINDS.items():
        mentions = []
        for src in SOURCES:
            if src not in usable:
                continue
            for k, m in enumerate(usable[src]["fields"][kind]):
                mentions.append({"src": src, "index": k, "label": m[label_key],
                                 "span": m.get("span"), "start": m.get("start"), "end": m.get("end"),
                                 **({"type": m["type"]} if "type" in m else {})})
        ids, matches = align_mentions(mentions, kind, embedder, thresholds[kind])
        clusters = {}
        for m, cid in zip(mentions, ids):
            m["id"] = cid
            clusters.setdefault(cid, []).append(
                {"src": m["src"], "label": m["label"], "span": m["span"]})
        out["mentions"][kind] = mentions
        out["clusters"][kind] = clusters
        out["matches"][kind] = [{"a": mentions[r["i"]]["label"], "a_src": mentions[r["i"]]["src"],
                                 "b": mentions[r["j"]]["label"], "b_src": mentions[r["j"]]["src"],
                                 "sim": r["sim"], "by": r["by"]} for r in matches]
    return out


def id_of(alignment, kind, src, label):
    """Look up the shared ID of a mention, e.g. id_of(al, "items", "fox", "fire tear gas")."""
    for m in alignment["mentions"][kind]:
        if m["src"] == src and m["label"] == label:
            return m["id"]
    return None


# ---------------------------------------------------------------- threshold tuning on dev

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dev_ids():
    return _load_json(os.path.join("data", "splits.json"))["dev"]


def dump_pairs(embedder, path=os.path.join(OUT_DIR, "dev_pairs.csv")):
    """Write every cross-headline pair in the dev stories for hand labelling."""
    ext = _load_json(EXTRACTIONS_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_id", "kind", "src_a", "a", "src_b", "b", "sim", "same"])
        for sid in _dev_ids():
            usable = {s: r for s, r in ext.get(sid, {}).items() if r.get("ok")}
            for kind, (label_key, _) in KINDS.items():
                mentions = [{"src": s, "label": m[label_key]}
                            for s in SOURCES if s in usable for m in usable[s]["fields"][kind]]
                for r in sorted(similarity_table(mentions, kind, embedder), key=lambda r: -r["sim"]):
                    a, b = mentions[r["i"]], mentions[r["j"]]
                    w.writerow([sid, kind, a["src"], a["label"], b["src"], b["label"], r["sim"], ""])
                    n += 1
    print(f"Wrote {n} pairs to {path}.\nFill the 'same' column with y or n, save as CSV, "
          f"then run: python -m src.align --tune")


def tune(path=os.path.join(OUT_DIR, "dev_pairs.csv")):
    """Precision/recall/F1 of 'sim >= t' against hand labels, for each kind and threshold."""
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["same"].strip().lower() in ("y", "n")]
    if not rows:
        print("No labelled rows yet: fill the 'same' column with y/n first.")
        return None
    best = {}
    for kind in KINDS:
        rk = [(float(r["sim"]), r["same"].strip().lower() == "y") for r in rows if r["kind"] == kind]
        if not rk:
            continue
        n_same = sum(g for _, g in rk)
        print(f"\n{kind}: {len(rk)} labelled pairs, {n_same} truly the same")
        if n_same < 3:
            print(f"  -> too few 'same' pairs to tune; keeping default "
                  f"{DEFAULT_THRESHOLDS[kind]} for {kind}")
            continue
        print("  threshold  precision  recall   F1")
        scores = []
        for t in np.arange(0.40, 0.96, 0.05):
            tp = sum(s >= t and g for s, g in rk)
            fp = sum(s >= t and not g for s, g in rk)
            fn = sum(s < t and g for s, g in rk)
            p = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
            scores.append((f1, p, round(float(t), 2)))
            print(f"  {t:9.2f}  {p:9.2f}  {r_:6.2f}  {f1:5.2f}")
        f1, p, t = max(scores, key=lambda x: (x[0], x[1]))      # best F1, ties -> higher precision
        best[kind] = t
        print(f"  -> best threshold for {kind}: {t} (F1 {f1:.2f})")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump({"thresholds": best, "tuned_on": "dev", "labels": path}, f, indent=1)
    print(f"\nSaved to {PARAMS_PATH}; 'python -m src.align' will use these thresholds.")
    return best


def load_thresholds():
    if os.path.exists(PARAMS_PATH):
        return {**DEFAULT_THRESHOLDS, **_load_json(PARAMS_PATH)["thresholds"]}
    return dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------- batch

def run_all(embedder, thresholds, out_path=os.path.join(OUT_DIR, "alignment.json")):
    ext = _load_json(EXTRACTIONS_PATH)
    result = {}
    for sid, story_ext in ext.items():
        al = align_story(story_ext, embedder, thresholds)
        result[sid] = al
        n = {k: len(al["clusters"][k]) for k in KINDS}
        m = {k: len(al["matches"][k]) for k in KINDS}
        flag = "" if al["ok"] else "  (skipped: fewer than 2 usable headlines)"
        print(f"{sid[:8]}  items {n['items']:2d} ({m['items']} matched)  "
              f"actors {n['actors']:2d} ({m['actors']} matched)  "
              f"outcomes {n['outcomes']:2d}{flag}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds, "model": MODEL_NAME, "stories": result},
                  f, ensure_ascii=False, indent=1)
    print(f"\nAligned {len(result)} stories with thresholds {thresholds} -> {out_path}")


def load_alignment(path=os.path.join(OUT_DIR, "alignment.json")):
    return _load_json(path)["stories"]


def show(story_id, embedder, thresholds):
    from src.load import load_stories, stories_by_id
    story = stories_by_id(load_stories())[story_id]
    for src in SOURCES:
        print(f"{src}: {story['articles'][src]['headline']}")
    al = align_story(_load_json(EXTRACTIONS_PATH)[story_id], embedder, thresholds)
    for kind in KINDS:
        print(f"\n{kind}:")
        for cid, ms in al["clusters"][kind].items():
            print(f"  {cid}: " + "  |  ".join(f"{m['src']}: {m['label']}" for m in ms))
        for mt in al["matches"][kind]:
            print(f"     matched {mt['a_src']}:{mt['a']!r} ~ {mt['b_src']}:{mt['b']!r} "
                  f"sim={mt['sim']} ({mt['by']})")


def main():
    ap = argparse.ArgumentParser(description="Stage S3: Sentence-BERT alignment")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dump-pairs", action="store_true", help="write dev pairs for hand labelling")
    g.add_argument("--tune", action="store_true", help="pick thresholds from labelled dev pairs")
    g.add_argument("--show", metavar="STORY_ID", help="print the alignment of one story")
    ap.add_argument("--threshold", type=float, help="use one threshold for every kind")
    args = ap.parse_args()

    if args.tune:
        tune()
        return
    thresholds = load_thresholds()
    if args.threshold is not None:
        thresholds = {k: args.threshold for k in KINDS}
    emb = SBERTEmbedder()
    if args.dump_pairs:
        dump_pairs(emb)
    elif args.show:
        show(args.show, emb, thresholds)
    else:
        run_all(emb, thresholds)


if __name__ == "__main__":
    main()
