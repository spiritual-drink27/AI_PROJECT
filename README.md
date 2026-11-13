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
