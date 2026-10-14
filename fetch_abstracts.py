"""
fetch_abstracts.py
==================
Downloads the most recent openly licensed bioRxiv cell biology preprints and
saves them to abstracts_v2.csv, the input for the benchmark (run_batch.py).

All licenses are included, even bioRxiv's "cc_no" (all rights reserved):
abstracts are treated as metadata, which bioRxiv itself distributes freely
through this public API. Each paper's license is still recorded in the CSV.

Only version 1 of each preprint is used: the abstract as first posted. That
gives the newest papers, which also lowers the chance that the model saw
them in training.

How to run
----------
    python fetch_abstracts.py

No API key needed: the bioRxiv API is public and free.
"""

import json
import ssl
import urllib.request
from datetime import date, timedelta

import certifi   # an up-to-date list of trusted HTTPS certificates
import pandas as pd

# Python from python.org on macOS can't find the system's certificates, so
# HTTPS downloads fail with CERTIFICATE_VERIFY_FAILED. Pointing it at
# certifi's list fixes that without changing anything system-wide.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
N_ABSTRACTS = 100
CATEGORY = "cell_biology"   # bioRxiv's name for the subject area

# How far back to search. bioRxiv posts ~70 new cell biology preprints a
# week, so a month is plenty for 100; extra is harmless.
DAYS_BACK = 45

# Only keep preprints posted AFTER the model's training data cutoff, so the
# model can't have seen them (or their real authors) in training.
# Claude Haiku/Sonnet/Opus 5.5 all list "Jun 2026" (Anthropic models overview
# page, checked 2026-10-14). We use the last day of that month to be safe.
# If you test a newer model, update this to its cutoff.
TRAINING_CUTOFF = "2026-06-30"

# Very short abstracts give the reviewer little to judge; skip them.
MIN_ABSTRACT_CHARS = 500

OUTPUT_FILE = "abstracts_v2.csv"


def fetch_page(start, end, cursor):
    """Fetch one page (up to 30 records) from the bioRxiv 'details' API."""
    url = (f"https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}"
           f"?category={CATEGORY}")
    with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def main():
    end = date.today()
    start = end - timedelta(days=DAYS_BACK)
    print(f"Searching bioRxiv {CATEGORY} from {start} to {end}...")

    # The API returns 30 records per page. 'cursor' is the record number to
    # start from, so we step it by 30 until we've seen the total.
    records, cursor = [], 0
    while True:
        data = fetch_page(start, end, cursor)
        page = data.get("collection", [])
        records.extend(page)
        total = int(data["messages"][0].get("total", 0))
        cursor += len(page)
        if not page or cursor >= total:
            break
        if cursor % 300 == 0:
            print(f"  ...{cursor}/{total} records")
    print(f"Fetched {len(records)} records (all versions)")

    df = pd.DataFrame(records)

    # Keep: version 1 only, posted after the training cutoff, long enough abstract.
    # (Dates are "YYYY-MM-DD" strings, which sort correctly as text.)
    df = df[df["version"].astype(str) == "1"]
    df = df[df["date"] > TRAINING_CUTOFF]
    df["abstract"] = df["abstract"].str.strip()
    df = df[df["abstract"].str.len() >= MIN_ABSTRACT_CHARS]
    df = df.drop_duplicates(subset="doi")

    # Newest first; DOI as a tie-breaker so the order is reproducible.
    df = df.sort_values(["date", "doi"], ascending=[False, True]).head(N_ABSTRACTS)

    if len(df) < N_ABSTRACTS:
        print(f"WARNING: only found {len(df)} eligible preprints. Increase DAYS_BACK.")

    # Same columns as the pilot's abstracts.csv, so the files look alike.
    out = pd.DataFrame({
        "date": df["date"],
        "doi": df["doi"],
        "version": df["version"],
        "url": "https://www.biorxiv.org/content/" + df["doi"] + "v" + df["version"].astype(str),
        "title": df["title"].str.strip(),
        "authors": df["authors"],
        "corresponding_author": df["author_corresponding"],
        "institution": df["author_corresponding_institution"],
        "license": df["license"],
        "abstract": df["abstract"],
    })
    # Final safety checks: stop with an error rather than save bad data.
    assert (out["version"].astype(str) == "1").all(), "non-v1 preprint slipped through"
    assert (out["date"] > TRAINING_CUTOFF).all(), "preprint from before the training cutoff"

    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(out)} preprints to {OUTPUT_FILE}")
    print(f"Posted between {out['date'].min()} and {out['date'].max()}")
    print("Licenses:", out["license"].value_counts().to_dict())


if __name__ == "__main__":
    main()
