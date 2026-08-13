#!/usr/bin/env python3
"""Capture Mother of Learning chapters from Royal Road to a text file."""

import re
import time
import urllib.request
from bs4 import BeautifulSoup

DELAY = 2  # seconds between requests

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8")

def extract(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else "Unknown Chapter"

    content_div = soup.find(class_="chapter-content")
    if not content_div:
        return title, "", None

    lines = []
    for tag in content_div.find_all(["p", "div"]):
        text = tag.get_text(" ", strip=True)
        if not text or re.fullmatch(r"-\s*break\s*-|[\*\s]+", text):
            continue
        if "ADVERTISEMENT" in text:
            break
        lines.append(text)

    next_url = None
    for a in soup.find_all("a"):
        if re.search(r"Next\s*Chapter", a.get_text(strip=True), re.I):
            href = a.get("href", "")
            if href:
                next_url = href if href.startswith("http") else "https://www.royalroad.com" + href
                break

    return title, "\n\n".join(lines), next_url

def capture_chapters(start_url, output_file):
    url = start_url
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        while url:
            print(f"[{count + 1}] Fetching: {url}", flush=True)
            try:
                html = fetch(url)
            except Exception as e:
                print(f"  ERROR: {e} — stopping.", flush=True)
                break
            title, body, next_url = extract(html)
            f.write(f"{title}\n\n{body}\n\n{'=' * 60}\n\n")
            f.flush()
            count += 1
            print(f"  Done: {title}", flush=True)
            url = next_url
            if url:
                time.sleep(DELAY)
    print(f"\nFinished — {count} chapters written to {output_file}", flush=True)

def main():
    capture_chapters(
        start_url="https://www.royalroad.com/fiction/21220/mother-of-learning/chapter/301778/1-good-morning-brother",
        output_file="/home/dqromney/books/mother-of-learning.txt",
    )


if __name__ == "__main__":
    main()
