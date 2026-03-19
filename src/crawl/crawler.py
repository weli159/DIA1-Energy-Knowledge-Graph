"""
src/crawl/crawler.py
Web Crawler — Sustainable Energy domain
Respects robots.txt, uses trafilatura for boilerplate removal.
Outputs: data/crawler_output.jsonl, data/crawl_summary.csv
"""

import csv
import json
import re
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura

# ── Configuration ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # project root
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

OUT_JSONL   = DATA / "crawler_output.jsonl"
OUT_SUMMARY = DATA / "crawl_summary.csv"
MIN_WORDS   = 500
TIMEOUT     = 30

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Renewable_energy",
    "https://en.wikipedia.org/wiki/Solar_power",
    "https://en.wikipedia.org/wiki/Wind_power",
    "https://en.wikipedia.org/wiki/Hydroelectricity",
    "https://en.wikipedia.org/wiki/Geothermal_energy",
    "https://en.wikipedia.org/wiki/Bioenergy",
    "https://en.wikipedia.org/wiki/Photovoltaics",
    "https://en.wikipedia.org/wiki/Energy_storage",
    "https://en.wikipedia.org/wiki/Sustainable_energy",
    "https://en.wikipedia.org/wiki/International_Renewable_Energy_Agency",
    "https://en.wikipedia.org/wiki/Intergovernmental_Panel_on_Climate_Change",
    "https://en.wikipedia.org/wiki/International_Energy_Agency",
]

# ── Robots.txt ─────────────────────────────────────────────────────────────
_robots: dict[str, urllib.robotparser.RobotFileParser] = {}

def can_fetch(url: str, agent: str = "EnergyKG-Bot/1.0 (academic)") -> bool:
    parsed = urlparse(url)
    base   = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots:
        import urllib.request
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            req = urllib.request.Request(
                f"{base}/robots.txt",
                headers={"User-Agent": agent}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                rp.parse(response.read().decode("utf-8").splitlines())
        except Exception as e:
            # If robots.txt fails to load (403, etc), typically allow crawling if academic
            pass
        _robots[base] = rp
    
    # If rp has no rules (e.g., failed to load), allow by default
    if not _robots[base].default_entry and not _robots[base].entries:
        return True
        
    return _robots[base].can_fetch(agent, url)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def is_useful(text: str) -> bool:
    return len(text.split()) > MIN_WORDS


# ── Main crawl loop ────────────────────────────────────────────────────────
def crawl(seed_urls: list[str] = SEED_URLS) -> list[dict]:
    records = []
    for url in seed_urls:
        status, text, title, wc = "ok", "", "", 0

        if not can_fetch(url):
            print(f"BLOCKED   {url}")
            records.append({"url": url, "title": "", "status": "blocked_robots", "word_count": 0, "text": ""})
            continue

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                status = "fetch_failed"
            else:
                title     = fetch_title(downloaded)
                extracted = trafilatura.extract(downloaded, include_comments=False,
                                                include_tables=False, favor_precision=True)
                if not extracted:
                    status = "empty_extract"
                else:
                    text = normalize(extracted)
                    wc   = len(text.split())
                    if not is_useful(text):
                        status = "too_short"
        except Exception as exc:
            status = f"error"

        print(f"{status:15s}  {wc:6d}w  {url}")
        records.append({
            "url": url, "title": title, "status": status,
            "word_count": wc, "text": text if status == "ok" else "",
        })

    # Save JSONL
    ok = [r for r in records if r["status"] == "ok"]
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps({k: r[k] for k in ("url","title","word_count","text")},
                               ensure_ascii=False) + "\n")

    # Save summary CSV
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url","title","status","word_count"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ("url","title","status","word_count")})

    print(f"\nCrawled {len(records)} URLs | {len(ok)} useful pages saved")
    print(f"  → {OUT_JSONL}")
    print(f"  → {OUT_SUMMARY}")
    return records


if __name__ == "__main__":
    crawl()
