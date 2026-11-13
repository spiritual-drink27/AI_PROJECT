# Explainable Same-Event News Framing Analysis

CS F407 Artificial Intelligence, Team BATCH, Demo track
(Knowledge Representation & Reasoning + Applications of Deep Learning)

Given a headline, the system finds other outlets' headlines about the same event and explains
how their framing differs along four dimensions (information selection, actor/action framing,
evaluative wording, consequence emphasis), using logical facts, Horn rules and forward chaining.
Every difference is linked to the exact supporting words. No left/right labels are assigned.

## Setup

```bash
pip install -r requirements.txt
bash scripts/get_data.sh          # downloads BASIL into data/raw/BASIL (not committed)
python -m src.load                # S1: builds data/processed/stories.json + headlines.csv
python -m src.split               # fixed 6 dev / 14 test stories -> data/splits.json
python -m tests.test_load         # sanity checks
```

## Data (Stage S1)

BASIL (Fan et al., 2019): 100 events, each reported by Fox News, NYT and HuffPost
(2010–2019, 300 articles), with gold lexical/informational bias spans.

`data/processed/stories.json` holds a list of stories:

```
story_id, year, main_event, main_entities,
articles: { fox | nyt | hpo: {
    headline, headline_from_title, outlet, date, url, stance,
    sentences:  [{para, sent, text}],
    bias_spans: [{bias: lex|inf, polarity, target, text, para, sent, start, end}] } }
```

- Headlines come from the article title. Outlet suffixes such as "| Fox News" are removed.
  One NYT article has no title, so its first sentence (dateline removed) is used, as stated in the proposal.
- BASIL span offsets are relative to a single sentence, so the loader finds each span's sentence
  (1723 of the 1726 spans are located).
- `src.load.headline_pairs(story)` gives the 3 comparison pairs in a fixed order:
  (fox, nyt), (fox, hpo), (nyt, hpo). All later stages and baselines use this order.

## Structured extraction (Stage S2)

An LLM reads each headline and returns five lists, which later become logical facts:

| Field | Example (from "Police fire tear gas on peaceful protesters") | Becomes atom |
|---|---|---|
| actors | police (LawEnforcement), protesters (Activist) | used by Agent / Eval / IsA |
| items | "fire tear gas" | `Reports(h, i)` |
| agents | police → "fire tear gas" | `Agent(h, x, i)` |
| evals | "peaceful" about protesters, positive | `Eval(h, x, pol, w)` |
| outcomes | "curfew", type Safety | `Outcome(h, k, t)` |

Every extracted entry must quote words that literally occur in the headline. `validate()` drops
anything the model invents and stores character offsets, so the demo can highlight the evidence.
If nothing usable is extracted, or the API fails, the result has `ok: false` and a readable
`error` message instead of crashing (goal G3).

**Setup:** copy `.env.example` to `.env` and paste a Groq API key (free, from console.groq.com).

```bash
python -m src.extract --headline "Police fire tear gas on peaceful protesters"   # try one
python -m src.extract --split annotated    # the 20 dev+test stories (60 headlines)
python -m src.extract --split all          # all 300 headlines
python -m tests.test_extract               # offline tests, no key needed
```

- Results are cached in `data/extractions/cache.json`, keyed by prompt version, model and headline.
  Re-running only calls the API for new or failed headlines, so you can stop with Ctrl+C and resume.
- `data/extractions/extractions.json` maps `{story_id: {fox|nyt|hpo: result}}` for S3/S4.
  Both files are committed, so teammates and the demo don't need to call the API again.
- `src/llm.py` is a small client for any OpenAI-compatible API. Baseline B3 reuses it, so
  both use the same LLM, as the proposal requires.

## Cross-headline alignment (Stage S3)

Different headlines describe the same thing in different words ("fire tear gas" vs "use tear gas",
"Steele" vs "Michael Steele"). Alignment gives equivalent mentions one shared ID
(`i1`, `a1`, `o1`...), so the rules in S5 can tell what one headline has and another lacks.

For each story and each kind (items, actors, outcomes):
1. Embed every mention with Sentence-BERT (`all-MiniLM-L6-v2`).
2. For each headline pair, compute cosine similarity between their mentions.
3. Greedy one-to-one matching, most similar pair first, kept only if similarity ≥ threshold.
4. Merge matches into clusters (union-find). Two mentions from the same headline are never merged.

Actors also match when one name is contained in the other ("Steele" ~ "Michael Steele").
Each accepted match is stored with its similarity score, so the demo can show why two phrases were treated as the same.

**Threshold tuning on the 6 dev stories** (test stories are never used for tuning):
```bash
python -m src.align --dump-pairs   # data/alignment/dev_pairs.csv: fill "same" with y/n
python -m src.align --tune         # precision/recall/F1 per threshold -> data/alignment/params.json
python -m src.align                # align all extracted stories with the tuned thresholds
python -m src.align --show <story_id>   # inspect one story
python -m tests.test_align         # offline tests (stand-in embedder, no download)
```
Defaults before tuning: items 0.70, actors 0.75, outcomes 0.70.
Known limitation: embeddings ignore roles, so "police shoot man" and "man shoots police" can look similar.

## Repository layout

| Path | Stage | Owner |
|---|---|---|
| `src/load.py`, `src/split.py` | S1 data loading, grouping, dev/test split | M1, M2 |
| `src/extract.py` | S2 LLM structured extraction | M2 |
| `src/align.py` | S3 Sentence-BERT alignment | M3 |
| `src/facts.py` | S4 ontology and logical facts | M4 |
| `src/engine.py` | S5 forward chaining and rules R1–R4 | M5 |
| `src/explain.py`, `app.py` | S6 explanations and web demo | M6 |
| `baselines/` | B1 TF-IDF, B2 Sentence-BERT, B3 direct LLM | M7 |
| `eval/` | annotation, metrics, ablation | M8 |
