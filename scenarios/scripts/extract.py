#!/usr/bin/env python3
"""
Run the rubric extraction over corpus/*.txt with the Anthropic API.

Usage (from the scenarios/ directory):
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 scripts/extract.py [--model claude-sonnet-5] [--only id1,id2] [--overwrite]

Each corpus file produces extractions/<id>.json following rubric.json's
extraction_schema. Documents longer than ~150k characters are truncated from
the middle (intro + conclusion preserved), which is where positions live.
"""

import argparse
import json
import os
import sys
from datetime import date

try:
    import anthropic
except ImportError:
    sys.exit("pip install anthropic")

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.dirname(HERE)
CORPUS = os.path.join(SCEN, "corpus")
EXTRACT = os.path.join(SCEN, "extractions")
MAX_CHARS = 150_000

PROMPT = """You are an expert research assistant for an economics project that maps
expert views about future AI onto a fixed rubric. Read the document and fill the
extraction schema in rubric.json EXACTLY.

Rules:
- Score each dimension ONLY if the document takes a position (explicitly or through
  its central scenario). Use null for silence. Do not impute your own views.
- Every non-null score needs at least one verbatim quote (<= 40 words) with an
  approximate location.
- central_scenario_summary must be faithful to the AUTHOR's view, not a critique.
- Return ONLY the JSON object, no markdown fences.

RUBRIC:
{rubric}

DOCUMENT METADATA:
{meta}

DOCUMENT TEXT:
{text}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--only", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip())

    os.makedirs(EXTRACT, exist_ok=True)
    with open(os.path.join(SCEN, "rubric.json"), encoding="utf-8") as f:
        rubric = f.read()

    client = anthropic.Anthropic()
    files = sorted(f for f in os.listdir(CORPUS)
                   if f.endswith(".txt") and not f.startswith("_"))
    for fname in files:
        sid = fname[:-4]
        if only and sid not in only:
            continue
        out = os.path.join(EXTRACT, f"{sid}.json")
        if os.path.exists(out) and not args.overwrite:
            print(f"  skip (exists): {sid}")
            continue
        with open(os.path.join(CORPUS, fname), encoding="utf-8") as f:
            raw = f.read()
        meta, _, text = raw.partition("=" * 70)
        if len(text) > MAX_CHARS:
            half = MAX_CHARS // 2
            text = text[:half] + "\n\n[... middle truncated ...]\n\n" + text[-half:]
        print(f"  extracting {sid} ({len(text):,} chars) ...", flush=True)
        try:
            msg = client.messages.create(
                model=args.model, max_tokens=4000,
                messages=[{"role": "user", "content": PROMPT.format(
                    rubric=rubric, meta=meta.strip(), text=text)}])
            payload = msg.content[0].text.strip()
            if payload.startswith("```"):
                payload = payload.strip("`").lstrip("json").strip()
            data = json.loads(payload)
            data["extractor"] = args.model
            data["extraction_date"] = str(date.today())
            with open(out, "w", encoding="utf-8") as g:
                json.dump(data, g, indent=2, ensure_ascii=False)
            print("    ok")
        except Exception as e:
            print(f"    FAILED: {e}")
    print("done.")


if __name__ == "__main__":
    main()
