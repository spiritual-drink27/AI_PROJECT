"""Pick the 20 annotated stories (6 dev + 14 test) from the proposal, Section 5.1.

Deterministic (fixed seed) and balanced: 2 stories per year, 2010-2019.
Run once:  python -m src.split
Output:    data/splits.json  ->  {"dev": [...], "test": [...], "unused": [...]}

The test story IDs are the only thing in this file the pipeline builders see.
The test ANNOTATIONS live in annotations/test/, which M2-M5 must not open
until the single test run (proposal, Section 5.1).
"""

import json
import os
import random
from collections import defaultdict

from src.load import load_stories

SEED = 407          # course number; fixed so the split never changes
PER_YEAR = 2
N_DEV = 6


def make_split(stories, seed=SEED):
    rng = random.Random(seed)
    by_year = defaultdict(list)
    for s in stories:
        by_year[s["year"]].append(s["story_id"])

    chosen = []
    for year in sorted(by_year):
        chosen += rng.sample(sorted(by_year[year]), PER_YEAR)

    rng.shuffle(chosen)
    dev, test = sorted(chosen[:N_DEV]), sorted(chosen[N_DEV:])
    unused = sorted(s["story_id"] for s in stories if s["story_id"] not in set(chosen))
    return {"seed": seed, "dev": dev, "test": test, "unused": unused}


def main(out="data/splits.json"):
    split = make_split(load_stories())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f, indent=1)
    print(f"dev={len(split['dev'])} test={len(split['test'])} unused={len(split['unused'])} -> {out}")


if __name__ == "__main__":
    main()
