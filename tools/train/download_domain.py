"""
Download domain image sets via Wikimedia Commons API (no LFW / gated HF).

Presets:
  faces   — portrait photos
  anime   — anime / manga related images
  nature  — landscapes / nature
  people  — people photographs (broader than portraits)

Usage:
  python download_domain.py --preset faces --out ./raw/faces --limit 300
  python download_domain.py --preset anime --out ./raw/anime --limit 300

Also reliable in this repo:
  python download_samples.py --out ./raw/samples --count 200
  python download_div2k.py --out ./raw/div2k
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Category → Wikimedia Commons title (without "Category:" prefix in some APIs we include it)
PRESETS: dict[str, list[str]] = {
    "faces": [
        "Category:Portrait_photographs",
        "Category:Self-portraits",
        "Category:Portrait_photographs_of_women",
        "Category:Portrait_photographs_of_men",
    ],
    "anime": [
        "Category:Anime_and_manga",
        "Category:Anime_screenshots",
        "Category:Anime_illustrations",
        "Category:Manga_covers",
    ],
    "nature": [
        "Category:Landscapes",
        "Category:Nature_photographs",
        "Category:Forests",
    ],
    "people": [
        "Category:People_of_Europe",
        "Category:Street_photography",
        "Category:Crowds",
    ],
}

API = "https://commons.wikimedia.org/w/api.php"
UA = "VKProjDatasetBot/1.0 (educational; local training only)"


def http_json(url: str, retries: int = 4) -> dict:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"API request failed: {url}") from last


def http_download(url: str, dest: Path, retries: int = 3) -> bool:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                data = resp.read()
            if len(data) < 1000:
                return False
            dest.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(attempt)
    return False


def iter_category_images(category: str, max_pages: int = 30):
    """Yield dicts with thumb/original url from a Commons category."""
    cont: dict[str, str] | None = None
    pages = 0
    while pages < max_pages:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": "640",
            "format": "json",
        }
        if cont:
            params.update(cont)
        url = API + "?" + urllib.parse.urlencode(params)
        data = http_json(url)
        pages_data = (data.get("query") or {}).get("pages") or {}
        for page in pages_data.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = (info.get("mime") or "").lower()
            if not mime.startswith("image/"):
                continue
            if mime in ("image/svg+xml", "image/gif"):
                continue
            # Prefer scaled thumb, else original
            img_url = info.get("thumburl") or info.get("url")
            if not img_url:
                continue
            yield {
                "title": page.get("title", "file"),
                "url": img_url,
                "mime": mime,
            }

        cont_raw = data.get("continue")
        if not cont_raw:
            break
        cont = {k: str(v) for k, v in cont_raw.items()}
        pages += 1
        time.sleep(0.2)  # be polite to Commons


def download_preset(preset: str, out: Path, limit: int, seed: int) -> int:
    cats = PRESETS[preset]
    out.mkdir(parents=True, exist_ok=True)

    # Collect candidates from all categories, then shuffle
    candidates: list[dict] = []
    seen_urls: set[str] = set()
    print(f"Collecting from Wikimedia Commons ({preset}) …")
    for cat in cats:
        print(f"  category: {cat}")
        try:
            for item in iter_category_images(cat):
                if item["url"] in seen_urls:
                    continue
                seen_urls.add(item["url"])
                candidates.append(item)
                if len(candidates) >= limit * 4:
                    break
        except Exception as e:
            print(f"  skip category ({e})")
        if len(candidates) >= limit * 4:
            break

    if not candidates:
        raise SystemExit(
            f"Не удалось получить список файлов для preset={preset}.\n"
            "Проверьте сеть / доступ к commons.wikimedia.org.\n"
            "Альтернативы: download_samples.py, download_div2k.py или свои фото в raw/."
        )

    random.Random(seed).shuffle(candidates)
    saved = 0
    for item in candidates:
        if saved >= limit:
            break
        ext = ".jpg"
        mime = item["mime"]
        if "png" in mime:
            ext = ".png"
        elif "webp" in mime:
            ext = ".webp"
        dest = out / f"{saved:05d}{ext}"
        ok = http_download(item["url"], dest)
        if not ok:
            continue
        saved += 1
        if saved % 25 == 0 or saved == limit:
            print(f"  saved {saved}/{limit}")
        time.sleep(0.15)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Download faces/anime/nature via Wikimedia Commons")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    n = download_preset(args.preset, args.out, args.limit, args.seed)
    if n == 0:
        raise SystemExit("Downloaded 0 images.")
    print(f"Done: {n} images → {args.out.resolve()}")
    print(f"Next: python augment.py --input {args.out} --output ./dataset --count 10000")


if __name__ == "__main__":
    main()
