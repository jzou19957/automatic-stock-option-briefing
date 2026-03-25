"""
fetch.py — Fetches ALL data for each ticker in one step:
  1. /async/options_ticker_info?ticker=X  → IV, HV, IVR, IVP, OI, volume
  2. /async/option_chain?ticker=X&...     → strikes at ~45 DTE
  3. yfinance                             → price, MA21, earnings date

Folder structure:
    data/
    └── 2026-03-25/
        ├── fetch_log.json
        └── MSFT/
            ├── optioncharts_overview.html  ← IV/HV/IVR/OI/volume
            ├── optioncharts_chain.html     ← strike chain at 45 DTE
            ├── yfinance.json               ← price, MA21, earnings
            └── meta.json                   ← fetch status + timestamps

Usage:
    python fetch.py MSFT
    python fetch.py MSFT AAPL NVDA
    python fetch.py                  (reads universe.csv)

Requirements:
    pip install requests yfinance pandas
"""

import sys
import csv
import json
import time
import random
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("./data")
TODAY           = str(date.today())
TODAY_DIR       = DATA_DIR / TODAY

RETRY_COUNT     = 3
RETRY_DELAYS    = [5, 15, 30]
TICKER_DELAY    = (2, 4)
REQUEST_TIMEOUT = 30

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

HEADERS_HTMX = {
    **HEADERS_BROWSER,
    "Referer":    "https://optioncharts.io/",
    "HX-Request": "true",
    "HX-Target":  "option-charts-main-content-container",
}


# ── Date helpers ──────────────────────────────────────────────────────────────
from expiry_utils import get_third_friday, find_best_monthly_expiry, build_chain_urls


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def fetch_with_retry(session, url, headers, label):
    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            print(f"      [{attempt}/{RETRY_COUNT}] GET {url[:75]}...", end=" ")
            resp = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                print(f"✅ {len(resp.text):,} chars")
                return resp.text, "OK"
            elif resp.status_code == 404:
                print(f"❌ 404 — skipping")
                return None, "HTTP_404"
            elif resp.status_code == 403:
                print(f"🚫 403 — blocked")
                last_error = "BLOCKED_403"
                wait = RETRY_DELAYS[attempt-1] * 2
            elif resp.status_code == 429:
                print(f"⏳ 429 — rate limited")
                last_error = "RATE_LIMITED_429"
                wait = RETRY_DELAYS[attempt-1] * 3
            else:
                print(f"❌ HTTP {resp.status_code}")
                last_error = f"HTTP_{resp.status_code}"
                wait = RETRY_DELAYS[attempt-1]

            if attempt < RETRY_COUNT:
                print(f"      ⏳ waiting {wait}s...")
                time.sleep(wait)

        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout")
            last_error = "TIMEOUT"
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAYS[attempt-1])
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error")
            last_error = "CONNECTION_ERROR"
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAYS[attempt-1])
        except Exception as e:
            print(f"❌ {str(e)[:60]}")
            last_error = "UNKNOWN_ERROR"
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAYS[attempt-1])

    print(f"      ❌ All {RETRY_COUNT} attempts failed — {last_error}")
    return None, last_error or "RETRY_EXHAUSTED"


# ── Source 1: Overview (IV + OI + volume) ────────────────────────────────────

def fetch_overview(ticker, session, ticker_dir):
    url = f"https://optioncharts.io/async/options_ticker_info?ticker={ticker}"
    print(f"\n    [1/3] Overview (IV/HV/IVR/OI)")
    html, status = fetch_with_retry(session, url, HEADERS_HTMX, "overview")

    result = {
        "source": "optioncharts_overview", "url": url,
        "status": status,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": None, "size_bytes": 0,
    }
    if html:
        out = ticker_dir / "optioncharts_overview.html"
        out.write_text(html, encoding="utf-8")
        result["file"]       = str(out)
        result["size_bytes"] = len(html.encode("utf-8"))
        print(f"      💾 {out.name} ({result['size_bytes']:,} bytes)")
    return result


# ── Source 2: Chain at ~45 DTE ────────────────────────────────────────────────

def fetch_chain(ticker, session, ticker_dir):
    """Fetch monthly chain at ~45 DTE. Always uses 3rd Friday + :m suffix."""
    urls = build_chain_urls(ticker)
    date_str     = urls["date_str"]
    dte          = urls["dte"]
    full_page_url = urls["full_url"]
    async_url    = urls["async_url"]

    print(f"\n    [2/3] Chain (monthly {date_str}, {dte} DTE)")
    print(f"      {async_url}")

    # Prime session — visit full page first so HTMX knows which expiry we want
    print(f"      Priming session...")
    try:
        prime_headers = {**HEADERS_BROWSER,
                         "Referer": f"https://optioncharts.io/options/{ticker}"}
        session.get(full_page_url, headers=prime_headers, timeout=20)
        time.sleep(1.5)
    except Exception as e:
        print(f"      ⚠️  Prime failed: {e} — continuing")

    htmx_headers = {
        **HEADERS_BROWSER,
        "Referer":        full_page_url,
        "HX-Request":     "true",
        "HX-Current-URL": full_page_url,
        "HX-Target":      "option-charts-main-content-container",
    }
    html, status = fetch_with_retry(session, async_url, htmx_headers, "chain")

    result = {
        "source": "optioncharts_chain", "url": async_url,
        "expiry": date_str, "dte": dte, "expiry_type": "monthly",
        "status": status,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": None, "size_bytes": 0,
    }
    if html:
        out = ticker_dir / "optioncharts_chain.html"
        out.write_text(html, encoding="utf-8")
        result["file"]       = str(out)
        result["size_bytes"] = len(html.encode("utf-8"))
        print(f"      💾 {out.name} ({result['size_bytes']:,} bytes)")
    return result


# ── Source 3: yfinance (price, MA21, earnings) ───────────────────────────────

def fetch_yfinance(ticker, ticker_dir):
    """
    Fetch price history + earnings date from yfinance.
    Saves to yfinance.json — no HTML, just clean JSON.
    """
    print(f"\n    [3/3] yfinance (price, MA21, earnings)")

    result = {
        "source":              "yfinance",
        "status":              "FAILED",
        "fetched_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":       None,
        "ma_21":               None,
        "price_vs_ma_pct":     None,
        "next_earnings_date":  None,
        "next_earnings_days":  None,
        "file":                None,
    }

    try:
        stock = yf.Ticker(ticker)

        # ── Price + MA21 ──────────────────────────────────────────────────────
        hist = stock.history(period="60d")
        if hist.empty:
            print(f"      ❌ No price history returned")
            return result

        current_price    = round(float(hist["Close"].iloc[-1]), 2)
        ma_21            = round(float(hist["Close"].tail(21).mean()), 2)
        price_vs_ma_pct  = round((current_price - ma_21) / ma_21 * 100, 2)

        result["current_price"]   = current_price
        result["ma_21"]           = ma_21
        result["price_vs_ma_pct"] = price_vs_ma_pct

        print(f"      📊 Price: ${current_price} | MA21: ${ma_21} | "
              f"vs MA: {price_vs_ma_pct:+.2f}%", end="")

        # ── Earnings date ─────────────────────────────────────────────────────
        try:
            cal = stock.calendar
            earn_date = None

            if cal is not None:
                # calendar can be a dict or DataFrame depending on yfinance version
                if isinstance(cal, dict) and "Earnings Date" in cal:
                    raw = cal["Earnings Date"]
                    earn_date = raw[0] if isinstance(raw, list) else raw
                elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
                    earn_date = cal.loc["Earnings Date"].iloc[0]
                elif hasattr(cal, "columns") and len(cal.columns) > 0:
                    earn_date = cal.iloc[0, 0]

            if earn_date is not None:
                # Normalize to date object
                if hasattr(earn_date, "date"):
                    earn_date = earn_date.date()
                elif isinstance(earn_date, str):
                    earn_date = datetime.strptime(earn_date[:10], "%Y-%m-%d").date()

                days_to_earnings = (earn_date - date.today()).days
                result["next_earnings_date"] = str(earn_date)
                result["next_earnings_days"] = days_to_earnings
                print(f" | Earnings: {earn_date} ({days_to_earnings}d)", end="")

        except Exception as e:
            print(f" | Earnings: unavailable ({str(e)[:40]})", end="")

        print()  # newline after the inline prints
        result["status"] = "OK"

    except Exception as e:
        print(f"      ❌ yfinance error: {e}")
        result["status"] = f"ERROR: {str(e)[:80]}"
        return result

    # Save JSON
    out = ticker_dir / "yfinance.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["file"] = str(out)
    print(f"      💾 yfinance.json ({out.stat().st_size:,} bytes)")

    return result


# ── Per-ticker orchestrator ───────────────────────────────────────────────────

def fetch_ticker(ticker, session, log):
    ticker     = ticker.upper()
    ticker_dir = TODAY_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'━'*52}")
    print(f"  📌 {ticker}")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Warm up optioncharts session
    try:
        session.get("https://optioncharts.io/", headers=HEADERS_BROWSER, timeout=15)
        time.sleep(random.uniform(0.8, 1.5))
    except Exception:
        pass

    # 1. Overview
    ov = fetch_overview(ticker, session, ticker_dir)
    time.sleep(random.uniform(1.0, 2.0))

    # 2. Chain
    ch = fetch_chain(ticker, session, ticker_dir)
    time.sleep(random.uniform(0.5, 1.0))

    # 3. yfinance (no rate limit needed — direct API)
    yf_result = fetch_yfinance(ticker, ticker_dir)

    # Overall status
    ov_ok = ov["status"] == "OK"
    ch_ok = ch["status"] == "OK"
    yf_ok = yf_result["status"] == "OK"

    if ov_ok and ch_ok and yf_ok:
        overall = "COMPLETE"
    elif ch_ok and yf_ok:
        overall = "PARTIAL_NO_IV"       # can still calculate with estimated IV
    elif ov_ok and yf_ok:
        overall = "PARTIAL_NO_CHAIN"    # no options chain — skip
    elif ch_ok or ov_ok:
        overall = "PARTIAL_NO_PRICE"    # no price data — skip
    else:
        overall = "FAILED"

    icon = {
        "COMPLETE":          "✅",
        "PARTIAL_NO_IV":     "⚠️ ",
        "PARTIAL_NO_CHAIN":  "⚠️ ",
        "PARTIAL_NO_PRICE":  "⚠️ ",
        "FAILED":            "❌",
    }.get(overall, "❓")

    print(f"\n  {icon} {ticker}: {overall}")

    ticker_result = {
        "ticker":         ticker,
        "date":           TODAY,
        "expiry_type":    "monthly",
        "note":           "monthly 3rd Friday expiry",
        "started_at":     started,
        "finished_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall,
        "sources": {
            "overview": ov,
            "chain":    ch,
            "yfinance": yf_result,
        }
    }

    (ticker_dir / "meta.json").write_text(
        json.dumps(ticker_result, indent=2), encoding="utf-8"
    )
    log[ticker] = ticker_result
    return ticker_result


# ── Main ──────────────────────────────────────────────────────────────────────

def load_universe(csv_file="universe.csv"):
    tickers = []
    try:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get("symbol", "").strip().upper()
                if s:
                    tickers.append(s)
        print(f"📋 Loaded {len(tickers)} tickers from {csv_file}")
    except FileNotFoundError:
        print(f"⚠️  {csv_file} not found — using defaults")
        tickers = ["MSFT", "AAPL"]
    return tickers


def main():
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
        print(f"🎯 Running: {', '.join(tickers)}")
    else:
        tickers = load_universe()

    expiry_info = build_chain_urls("SPY")  # preview expiry for display
    TODAY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Fetch Session — {TODAY}")
    print(f"   Sources : optioncharts.io (2 endpoints) + yfinance")
    print(f"   Expiry  : {expiry_info['date_str']} ({expiry_info['dte']} DTE, monthly :m)")
    print(f"   Tickers : {len(tickers)}")
    print(f"   Output  : {TODAY_DIR}")
    print(f"{'='*60}")

    session = requests.Session()
    log     = {}

    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}]", end="")
        fetch_ticker(ticker, session, log)
        if i < len(tickers) - 1:
            pause = random.uniform(*TICKER_DELAY)
            print(f"\n  ⏸  Pausing {pause:.1f}s...")
            time.sleep(pause)

    log_file = TODAY_DIR / "fetch_log.json"
    log_file.write_text(json.dumps(log, indent=2), encoding="utf-8")

    complete = sum(1 for r in log.values() if r["overall_status"] == "COMPLETE")
    partial  = sum(1 for r in log.values() if "PARTIAL" in r["overall_status"])
    failed   = sum(1 for r in log.values() if r["overall_status"] == "FAILED")

    print(f"\n{'='*60}")
    print(f"📊 Done — {TODAY}")
    print(f"   ✅ Complete : {complete}/{len(tickers)}")
    print(f"   ⚠️  Partial  : {partial}/{len(tickers)}")
    print(f"   ❌ Failed   : {failed}/{len(tickers)}")
    print(f"   📁 Data     : {TODAY_DIR}")
    print(f"{'='*60}")

    if partial or failed:
        print(f"\n⚠️  Issues:")
        for ticker, r in log.items():
            if r["overall_status"] not in ("COMPLETE",):
                for src, sr in r["sources"].items():
                    if sr.get("status") != "OK":
                        print(f"   {ticker}/{src}: {sr.get('status')}")


if __name__ == "__main__":
    main()