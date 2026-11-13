"""Sanity checks for Stage S1.  Run:  python -m tests.test_load"""

from src.load import SOURCES, headline_pairs, load_stories, stories_by_id
from src.split import make_split


def run():
    stories = load_stories()
    assert len(stories) == 100, len(stories)

    for s in stories:
        assert set(s["articles"]) == set(SOURCES), s["story_id"]
        for a in s["articles"].values():
            assert a["headline"].strip(), (s["story_id"], a["source"])
            assert "| Fox News" not in a["headline"]
            assert a["sentences"], (s["story_id"], a["source"])
        assert len(headline_pairs(s)) == 3

    # every located span really points at its text
    for s in stories:
        for a in s["articles"].values():
            sents = {(x["para"], x["sent"]): x["text"] for x in a["sentences"]}
            for sp in a["bias_spans"]:
                if sp["sent"] is not None:
                    assert sents[(sp["para"], sp["sent"])][sp["start"]:sp["end"]] == sp["text"]

    split = make_split(stories)
    assert len(split["dev"]) == 6 and len(split["test"]) == 14
    assert not set(split["dev"]) & set(split["test"])
    assert make_split(stories) == split, "split must be deterministic"
    assert set(split["dev"] + split["test"]) <= set(stories_by_id(stories))
    print("All S1 checks passed.")


if __name__ == "__main__":
    run()
