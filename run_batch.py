"""
run_batch.py
============
The benchmark version of the experiment, using the Message Batches API.

Question: does an LLM reviewer score the SAME abstract differently depending
on the author affiliation it is shown?

Design
------
Every abstract in abstracts_v2.csv is scored once under every condition in
institutions.csv:

    15 real institutions in 3 tiers (high / mid / low, 5 each)
     2 fictional institutions (an "unknown name" control)
     1 "none" condition: no affiliation line at all (the true baseline)

100 abstracts x 18 conditions = 1,800 requests, 1 repeat each. (The pilot
showed repeats are nearly identical, so more abstracts beat more repeats.)

Why batches?
------------
The Message Batches API takes all 1,800 requests in one upload, runs them on
Anthropic's side (usually within an hour, at most 24 h), and charges 50% of
the normal price. You don't keep a script running while it works; you check
back later. That changes the workflow from one script into four steps:

    python run_batch.py prepare   # build requests, save them, estimate cost (free)
    python run_batch.py test      # optional: 18-request sample to measure cost (cents)
    python run_batch.py submit    # upload the batch (SPENDS CREDITS)
    python run_batch.py status    # check progress (free)
    python run_batch.py collect   # download results into results_v2_<model>.csv (free)

Add a model name to run another model (default: claude-haiku-5-5):

    python run_batch.py prepare claude-sonnet-5-5

Each step saves its output to disk, so you can close your laptop between
steps. Files for each model go in their own folder, batch/<model>/, so
running a second model doesn't overwrite the first.
"""

import csv
import json
import os
import re
import sys

import pandas as pd

# The four scores we ask for. Used for parsing and checking replies.
SCORE_KEYS = ["rigor", "novelty", "impact", "overall"]


def get_text(msg):
    """Join together all the text blocks in a reply.

    Selecting blocks by TYPE (not position) skips the thinking block that
    comes first in these models' replies.
    """
    return "".join(block.text for block in msg.content if block.type == "text")


def parse_scores(text):
    """Turn the reply text into a dict of scores, or raise ValueError.

    Even when told "JSON only", models sometimes wrap the JSON in ```json
    fences or add a sentence around it, so we take the part from the first
    "{" to the last "}" rather than trusting the whole string.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)  # DOTALL: "." also matches newlines
    if not match:
        raise ValueError("No JSON object found in reply")
    data = json.loads(match.group(0))  # raises ValueError if the JSON is broken

    # Every score must be present and a whole number from 1 to 10.
    for key in SCORE_KEYS:
        value = data.get(key)
        if not isinstance(value, int) or not 1 <= value <= 10:
            raise ValueError("Bad or missing value for {!r}: {!r}".format(key, value))
    return data

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
# Effort controls how much the model thinks before answering. The same
# setting is used for every model so the comparison holds that choice
# constant. "medium" is the default for Haiku 5.5 and Opus 5.5 and matches
# the pilot; Sonnet 5.5 (default "high") will think somewhat less than usual.
# Caveat: each model maps a level to its own amount of thinking, so "medium"
# means the same SETTING, not identical thinking, across models.
EFFORT = "medium"

# BATCH prices in $ per million tokens (half the normal price), as of
# 2026-10-14. Only used for the cost estimate in 'prepare'.
#
# set_thinking: the 5.5 models think by default. Opus 4.7 does NOT think
# unless asked, so we turn adaptive thinking on for it explicitly, to keep
# all models reviewing the same way. (Opus 4.7's training cutoff is Jan 2026,
# so the Oct 2026 abstracts are new to it as well.)
MODELS = {
    "claude-haiku-5-5":  {"price_in": 0.05, "price_out": 0.25},
    "claude-sonnet-5-5": {"price_in": 1.00, "price_out": 5.00},
    "claude-opus-5-5":   {"price_in": 2.00, "price_out": 10.00},
    "claude-opus-4-7":   {"price_in": 2.50, "price_out": 12.50, "set_thinking": True},
}

# Which model this run uses: the 2nd word on the command line, else Haiku.
MODEL = sys.argv[2] if len(sys.argv) > 2 else "claude-haiku-5-5"
if MODEL not in MODELS:
    sys.exit("Unknown model {!r}. Choose one of: {}".format(MODEL, ", ".join(MODELS)))
PRICE_IN_PER_M = MODELS[MODEL]["price_in"]
PRICE_OUT_PER_M = MODELS[MODEL]["price_out"]

# Thinking tokens count toward max_tokens. Haiku used ~200 per reply in the
# pilot, but Sonnet and Opus may think much longer, and a reply that hits
# the limit is cut off and unusable. You only pay for tokens actually used,
# so a generous ceiling costs nothing extra.
MAX_TOKENS = 8000

ABSTRACTS_FILE = "abstracts_v2.csv"
INSTITUTIONS_FILE = "institutions.csv"

# Per-model working folder and results file.
BATCH_DIR = os.path.join("batch", MODEL)
REQUESTS_FILE = os.path.join(BATCH_DIR, "requests.jsonl")  # exactly what gets sent
MANIFEST_FILE = os.path.join(BATCH_DIR, "manifest.csv")    # custom_id -> paper + condition
BATCH_ID_FILE = os.path.join(BATCH_DIR, "batch_id.txt")    # saved after submit
RESULTS_FILE = "results_v2_{}.csv".format(MODEL)

# ---------------------------------------------------------------------------
# PROMPTS
# Changed from the pilot: the system prompt no longer says the reviewer will
# see an affiliation. Otherwise the "none" condition would look like a
# mistake ("you said there'd be an affiliation..."), which is itself a
# signal. Now the affiliation line simply appears or doesn't.
# As before, the prompt never mentions prestige or tells the model to
# ignore the affiliation.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an experienced reviewer on a cell biology funding panel.
You will be shown a preprint to evaluate.

Score the work on each criterion from 1 (very poor) to 10 (exceptional):
- rigor: soundness of the methods and whether the conclusions follow from the evidence
- novelty: how new the question, approach or finding is
- impact: likely importance to the field if the findings hold up
- overall: your overall assessment

Respond with ONLY a JSON object and no other text, in exactly this format:
{"rigor": <int>, "novelty": <int>, "impact": <int>, "overall": <int>, "rationale": "<one or two sentences>"}"""


def build_user_message(title, affiliation, abstract):
    """The per-request message. affiliation=None leaves the line out entirely."""
    lines = ["Title: " + title]
    if affiliation is not None:
        lines.append("Corresponding author affiliation: " + affiliation)
    lines += ["", "Abstract:", abstract]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STEP 1: PREPARE (free)
# ---------------------------------------------------------------------------
def load_conditions():
    """Read institutions.csv into a list of conditions.

    Each condition is a dict with tier, institution and the affiliation text
    to show (None for the 'none' condition, whose institution cell is blank).
    """
    inst = pd.read_csv(INSTITUTIONS_FILE)
    conditions = []
    for row in inst.to_dict("records"):
        name = row["institution"]
        if pd.isna(name):            # blank cell -> no affiliation line
            name = None
        conditions.append({"tier": row["tier"], "institution": name})
    return conditions


def prepare():
    papers = pd.read_csv(ABSTRACTS_FILE)
    conditions = load_conditions()
    os.makedirs(BATCH_DIR, exist_ok=True)

    requests, manifest = [], []
    for p_idx, paper in enumerate(papers.to_dict("records")):
        for c_idx, cond in enumerate(conditions):
            # custom_id is how we match each result back to its request.
            # Batch results come back in ANY order, so this is essential.
            # Rules: 1-64 characters, letters/digits/_/- only. So we use
            # short codes here and keep the full details in the manifest.
            custom_id = "p{:03d}_c{:02d}".format(p_idx, c_idx)

            params = {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "output_config": {"effort": EFFORT},
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": build_user_message(
                        paper["title"], cond["institution"], paper["abstract"]),
                }],
            }
            if MODELS[MODEL].get("set_thinking"):
                params["thinking"] = {"type": "adaptive"}
            requests.append({"custom_id": custom_id, "params": params})
            manifest.append({
                "custom_id": custom_id,
                "doi": paper["doi"],
                "tier": cond["tier"],
                "institution": cond["institution"] or "",
            })

    # Save the exact requests as JSON Lines (one JSON object per line).
    # 'submit' sends this file's contents unchanged, so what you inspect
    # here is exactly what the model sees.
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    pd.DataFrame(manifest).to_csv(MANIFEST_FILE, index=False)

    # Rough cost estimate: ~4 characters per token for the input. Output is
    # the uncertain part: Haiku used ~200 tokens per reply in the pilot, but
    # models that think longer could use several times that. So we show a
    # range from 200 to 1000 output tokens per reply.
    in_tokens = sum(len(SYSTEM_PROMPT) + len(r["params"]["messages"][0]["content"])
                    for r in requests) / 4
    in_cost = in_tokens / 1e6 * PRICE_IN_PER_M
    low = in_cost + 200 * len(requests) / 1e6 * PRICE_OUT_PER_M
    high = in_cost + 1000 * len(requests) / 1e6 * PRICE_OUT_PER_M

    print("{} abstracts x {} conditions = {} requests".format(
        len(papers), len(conditions), len(requests)))
    print("Model: {} (effort={})".format(MODEL, EFFORT))
    print("Saved {} and {}".format(REQUESTS_FILE, MANIFEST_FILE))
    print("Estimated cost at batch prices: ${:.2f} to ${:.2f}".format(low, high))
    print("\nExample request (first one):\n")
    print(requests[0]["params"]["messages"][0]["content"][:600])
    print("\nNext: python run_batch.py submit")


# ---------------------------------------------------------------------------
# STEP 2: SUBMIT (spends credits)
# ---------------------------------------------------------------------------
def make_client():
    # Imported here so 'prepare' works without the anthropic package or a key.
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def submit():
    if not os.path.exists(REQUESTS_FILE):
        sys.exit("No requests found. Run: python run_batch.py prepare")

    # Guard against paying twice: once a batch exists for this model, refuse
    # to submit another. Delete batch_id.txt yourself if you really mean to.
    if os.path.exists(BATCH_ID_FILE):
        sys.exit("A batch was already submitted ({}). "
                 "Use 'status' / 'collect', or delete that file to resubmit."
                 .format(open(BATCH_ID_FILE).read().strip()))

    with open(REQUESTS_FILE, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]

    client = make_client()
    batch = client.messages.batches.create(requests=requests)

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)
    print("Submitted {} requests as batch {}".format(len(requests), batch.id))
    print("Status: {}. Check with: python run_batch.py status".format(batch.processing_status))


# ---------------------------------------------------------------------------
# STEP 3: STATUS (free)
# ---------------------------------------------------------------------------
def read_batch_id():
    if not os.path.exists(BATCH_ID_FILE):
        sys.exit("No batch submitted yet for {}. Run: python run_batch.py submit".format(MODEL))
    return open(BATCH_ID_FILE).read().strip()


def status():
    batch = make_client().messages.batches.retrieve(read_batch_id())
    c = batch.request_counts
    print("Batch {}: {}".format(batch.id, batch.processing_status))
    print("  processing={} succeeded={} errored={} canceled={} expired={}".format(
        c.processing, c.succeeded, c.errored, c.canceled, c.expired))
    if batch.processing_status == "ended":
        print("Done. Next: python run_batch.py collect")


# ---------------------------------------------------------------------------
# STEP 4: COLLECT (free)
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "custom_id", "model", "doi", "tier", "institution",
    "rigor", "novelty", "impact", "overall", "rationale",
    "parse_ok", "error", "stop_reason", "input_tokens", "output_tokens", "raw_reply",
]


def download_rows(client, batch_id):
    """Download a finished batch's results as a list of row dicts.

    Shared by 'collect' and 'test', so both read results the same way.
    """
    # The manifest tells us which paper and condition each custom_id was.
    manifest = pd.read_csv(MANIFEST_FILE, keep_default_na=False).set_index("custom_id")

    rows = []
    for item in client.messages.batches.results(batch_id):
        info = manifest.loc[item.custom_id]
        row = {
            "custom_id": item.custom_id,
            "model": MODEL,
            "doi": info["doi"],
            "tier": info["tier"],
            "institution": info["institution"],
        }
        result = item.result
        if result.type == "succeeded":
            msg = result.message
            text = get_text(msg)  # picks text blocks by type, skipping thinking
            row.update({
                "raw_reply": text,
                "stop_reason": msg.stop_reason,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            })
            if msg.stop_reason == "refusal":
                # Sonnet and Opus run safety classifiers (including a "bio"
                # category) that may occasionally decline a biology abstract.
                # Record it separately: if refusals cluster in one tier,
                # that's a finding in itself.
                # stop_details.category says which classifier fired, e.g. "bio".
                category = msg.stop_details.category if msg.stop_details else None
                row.update({"parse_ok": False, "error": "refusal ({})".format(category)})
            elif msg.stop_reason == "max_tokens":
                row.update({"parse_ok": False, "error": "cut off at max_tokens"})
            else:
                try:
                    scores = parse_scores(text)
                    row.update({k: scores[k] for k in SCORE_KEYS})
                    row["rationale"] = scores.get("rationale", "")
                    row["parse_ok"] = True
                    row["error"] = ""
                except ValueError as e:
                    row.update({"parse_ok": False, "error": "parse error: {}".format(e)})
        elif result.type == "errored":
            row.update({"parse_ok": False,
                        "error": "API error: {}".format(result.error.error.type)})
        else:  # "canceled" or "expired"
            row.update({"parse_ok": False, "error": result.type})
        rows.append(row)

    # Sort into a predictable order (results arrive in any order).
    rows.sort(key=lambda r: r["custom_id"])
    return rows


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def collect():
    client = make_client()
    batch_id = read_batch_id()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        sys.exit("Batch still {}. Try again later.".format(batch.processing_status))

    rows = download_rows(client, batch_id)
    write_rows(rows, RESULTS_FILE)

    n_ok = sum(1 for r in rows if r["parse_ok"])
    print("Saved {} results to {}: {} ok, {} failed".format(
        len(rows), RESULTS_FILE, n_ok, len(rows) - n_ok))
    failed = [r for r in rows if not r["parse_ok"]]
    for r in failed[:10]:
        print("  FAILED {} ({}, {}): {}".format(r["custom_id"], r["tier"], r["institution"], r["error"]))
    if len(failed) > 10:
        print("  ...and {} more".format(len(failed) - 10))


# ---------------------------------------------------------------------------
# OPTIONAL: TEST (spends a few cents)
# A small batch to measure real token usage, and so cost, before running the
# full 1,800 requests on a pricier model. It takes 18 requests from the
# prepared set: paper 0 with condition 0, paper 1 with condition 1, and so
# on, so the sample covers 18 different papers and every condition once.
#
# Run it twice: the first run submits; later runs check, and once the test
# batch has finished they print the measured usage and a cost projection.
# ---------------------------------------------------------------------------
TEST_BATCH_ID_FILE = os.path.join(BATCH_DIR, "test_batch_id.txt")
TEST_RESULTS_FILE = os.path.join(BATCH_DIR, "test_results.csv")
N_TEST = 18


def test():
    if not os.path.exists(REQUESTS_FILE):
        sys.exit("No requests found. Run: python run_batch.py prepare " + MODEL)
    client = make_client()

    # First run: submit the sample and stop.
    if not os.path.exists(TEST_BATCH_ID_FILE):
        with open(REQUESTS_FILE, encoding="utf-8") as f:
            all_requests = {r["custom_id"]: r for r in map(json.loads, f)}
        n_conditions = len(load_conditions())
        wanted = ["p{:03d}_c{:02d}".format(i, i % n_conditions) for i in range(N_TEST)]
        sample = [all_requests[cid] for cid in wanted]

        batch = client.messages.batches.create(requests=sample)
        with open(TEST_BATCH_ID_FILE, "w") as f:
            f.write(batch.id)
        print("Submitted test batch {} ({} requests). Run 'test' again to check."
              .format(batch.id, len(sample)))
        return

    # Later runs: check progress, then report.
    batch_id = open(TEST_BATCH_ID_FILE).read().strip()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print("Test batch {} still {}. Run 'test' again later.".format(
            batch_id, batch.processing_status))
        return

    rows = download_rows(client, batch_id)
    write_rows(rows, TEST_RESULTS_FILE)
    used = pd.DataFrame([r for r in rows if r.get("output_tokens") is not None])

    print("Test batch for {} (effort={}): {} of {} replies parsed OK".format(
        MODEL, EFFORT, sum(r["parse_ok"] for r in rows), len(rows)))
    for r in rows:
        if not r["parse_ok"]:
            print("  FAILED {}: {}".format(r["custom_id"], r["error"]))
    if used.empty:
        return
    print("Output tokens per reply: mean {:.0f}, max {:.0f} (limit {})".format(
        used["output_tokens"].mean(), used["output_tokens"].max(), MAX_TOKENS))

    # Project the full run's cost from the measured averages.
    n_full = sum(1 for _ in open(REQUESTS_FILE))
    full_cost = n_full * (used["input_tokens"].mean() / 1e6 * PRICE_IN_PER_M
                          + used["output_tokens"].mean() / 1e6 * PRICE_OUT_PER_M)
    print("Projected cost of the full {} requests: ${:.2f}".format(n_full, full_cost))
    print("Saved details to " + TEST_RESULTS_FILE)


# ---------------------------------------------------------------------------
# COMMAND LINE: python run_batch.py <step> [model]
# ---------------------------------------------------------------------------
STEPS = {"prepare": prepare, "submit": submit, "status": status,
         "collect": collect, "test": test}

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in STEPS:
        sys.exit("Usage: python run_batch.py [prepare|test|submit|status|collect] [model]")
    STEPS[sys.argv[1]]()
