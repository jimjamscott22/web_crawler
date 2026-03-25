# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup (preferred — uses uv for reproducible installs):**
```bash
uv sync
```

**Setup (alternative):**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run (CLI mode):**
```bash
python crawler.py --url https://example.com --query "keyword"
```

**Run (GUI mode):**
```bash
python crawler.py --gui
```

**Run with options:**
```bash
# Async with custom concurrency and delay
python crawler.py --url https://example.com --query "contact" --concurrency 10 --delay 0.5

# Sync mode (slower, simpler)
python crawler.py --url https://example.com --query "privacy" --sync

# Export results
python crawler.py --url https://example.com --query "about" --export-json --export-csv
```

There is no test suite; there are no lint/format commands configured.

## Architecture

Everything lives in a single file: `crawler.py`. The entry point is `main()` at the bottom, which dispatches to `run_gui()` (tkinter GUI) or CLI logic depending on `--gui` flag.

**Key classes:**
- `URLFrontier` — BFS queue backed by `deque`. Prioritizes same-domain URLs (pushed to front), deduplicates via a "seen" set, and shuffles external URLs.
- `FormHandler` — Uses `mechanicalsoup` to detect, categorize, and submit HTML forms; drives authentication login flows.
- `DomainRateLimiter` — Per-domain asyncio locks + timestamps to enforce `DEFAULT_POLITENESS_DELAY` between requests to the same host.

**Two crawl paths:**
- `crawl_async()` — primary path using `aiohttp` with concurrent workers (configurable via `--concurrency`), exponential-backoff retries, and `DomainRateLimiter`.
- `crawl()` — synchronous fallback using `requests` + `mechanicalsoup` browser; used when `--sync` is passed or `aiohttp` is unavailable.
- `run_async_crawl()` wraps `crawl_async()` to run it from synchronous contexts (CLI, GUI thread).

**Authentication flow:** `FormHandler` logs in via `mechanicalsoup`, session cookies are saved to `crawler_session.pkl` and restored on future runs.

**Output files** (written to cwd):
- `crawler.log` — always written
- `results.json` / `results.csv` — written when `--export-json` / `--export-csv` flags are passed

**Robots.txt** is checked via `urllib.robotparser` with a per-domain cache (`robots_cache` dict passed through the call stack). Assumes allowed if robots.txt is unreachable.
