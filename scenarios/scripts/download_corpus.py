#!/usr/bin/env python3
"""
Download the scenario corpus listed in sources.csv into corpus/ as clean text.

Usage (from the scenarios/ directory):
    pip install requests beautifulsoup4 pypdf
    python3 scripts/download_corpus.py [--priority 2] [--only id1,id2] [--overwrite]

Notes:
- HTML pages are reduced to readable text (scripts/nav stripped).
- PDFs are extracted with pypdf.
- arXiv abstract pages are automatically upgraded to full-text PDF.
- Failures are recorded in corpus/_failures.csv so you can fetch those by hand
  (paywalled books/reports usually need manual excerpts).
"""

import argparse
import csv
import io
import os
import re
import sys
import time

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("pip install requests beautifulsoup4 pypdf")

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.dirname(HERE)
CORPUS = os.path.join(SCEN, "corpus")
HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus builder; academic use)"}


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "form", "noscript", "svg", "button"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def pdf_to_text(content):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def fetch(url, timeout=60):
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or url.endswith(".pdf"):
        return pdf_to_text(r.content)
    return clean_html(r.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", type=int, default=3,
                    help="fetch sources with priority <= this (default 3 = all)")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated source ids")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip())

    os.makedirs(CORPUS, exist_ok=True)
    failures = []
    with open(os.path.join(SCEN, "sources.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        sid = row["id"]
        if only and sid not in only:
            continue
        if not only and int(row["priority"]) > args.priority:
            continue
        out = os.path.join(CORPUS, f"{sid}.txt")
        if os.path.exists(out) and not args.overwrite:
            print(f"  skip (exists): {sid}")
            continue
        try:
            print(f"  fetching {sid} ...", flush=True)
            text = fetch(row["url"])
            if len(text) < 500:
                raise ValueError(f"suspiciously short ({len(text)} chars)")
            header = (f"SOURCE_ID: {sid}\nAUTHOR: {row['author']}\n"
                      f"TITLE: {row['title']}\nYEAR: {row['year']}\n"
                      f"TYPE: {row['type']}\nURL: {row['url']}\n"
                      + "=" * 70 + "\n\n")
            with open(out, "w", encoding="utf-8") as g:
                g.write(header + text)
            print(f"    ok: {len(text):,} chars")
        except Exception as e:
            print(f"    FAILED: {e}")
            failures.append({"id": sid, "url": row["url"], "error": str(e)})
        time.sleep(1.0)

    if failures:
        with open(os.path.join(CORPUS, "_failures.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "url", "error"])
            w.writeheader()
            w.writerows(failures)
        print(f"\n{len(failures)} failures written to corpus/_failures.csv "
              "(fetch those manually).")
    print("done.")


if __name__ == "__main__":
    main()
