"""
parser.py — Converts raw optioncharts HTML files into clean JSON.

Two parsers:
  1. parse_overview(html)  → extracts IV, HV, IVR, IVP, OI, volume, per-expiry stats
  2. parse_chain(html)     → extracts full strike table with bid/ask/vol/OI

Each saves a clean .json file alongside the raw .html.
JSON is compact, typed, and ready for calculation — no further cleaning needed.

Usage:
    python parser.py AAPL
    python parser.py AAPL MSFT NVDA
    python parser.py              (all tickers in today's folder)
    python parser.py --date 2026-03-24 AAPL

Requirements:
    pip install beautifulsoup4 lxml
"""

import sys
import re
import json
from datetime import date, datetime
from pathlib import Path
from bs4 import BeautifulSoup

DATA_DIR = Path("./data")
TODAY    = str(date.today())

STRIP_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "head", "img", "button", "form",
]

def clean_soup(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in STRIP_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    return soup

def to_float(val):
    """Convert string to float, return None if not possible."""
    if val is None:
        return None
    s = str(val).replace(",", "").replace("%", "").replace("$", "").strip()
    if s in ["-", "", "N/A", "n/a", "--"]:
        return None
    try:
        return float(s)
    except:
        return None

def to_int(val):
    """Convert string to int, return None if not possible."""
    f = to_float(val)
    return int(f) if f is not None else None


# ── Overview parser ───────────────────────────────────────────────────────────

def parse_overview(html, ticker):
    """
    Parse optioncharts async overview response.
    Expected endpoint: /async/options_ticker_info?ticker=AAPL
    
    Returns structured dict with all IV/OI/volume metrics.
    """
    soup = clean_soup(html)
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    result = {
        "ticker":      ticker.upper(),
        "parsed_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source":      "optioncharts_overview",

        # IV metrics
        "iv_30":       None,   # Implied Volatility 30d %
        "iv_rank":     None,   # IV Rank %
        "iv_pct":      None,   # IV Percentile %
        "hv_30":       None,   # Historical Volatility %
        "iv_low":      None,   # IV 52wk low %
        "iv_high":     None,   # IV 52wk high %
        "iv_low_date": None,
        "iv_high_date":None,

        # Open Interest
        "oi_total":    None,
        "oi_put":      None,
        "oi_call":     None,
        "oi_pc_ratio": None,
        "oi_avg_30d":  None,

        # Volume
        "vol_total":   None,
        "vol_put":     None,
        "vol_call":    None,
        "vol_pc_ratio":None,
        "vol_avg_30d": None,

        # Per-expiry stats table
        "expiry_stats": [],

        # Parse quality
        "parse_notes": []
    }

    # ── Strategy: find label→value pairs ─────────────────────────────────────
    # Labels and values appear on consecutive lines in the cleaned text
    # We scan for known labels and grab the next numeric line

    LABEL_MAP = {
        # label text (lowercase)       : result key, is_pct
        "implied volatility (30d)":     ("iv_30",       True),
        "iv (30d)":                     ("iv_30",       True),
        "iv rank":                      ("iv_rank",     True),
        "iv percentile":                ("iv_pct",      True),
        "historical volatility":        ("hv_30",       True),
        "iv low":                       ("iv_low",      True),
        "iv high":                      ("iv_high",     True),
        "today's open interest":        ("oi_total",    False),
        "put open interest":            ("oi_put",      False),
        "call open interest":           ("oi_call",     False),
        "open interest avg (30-day)":   ("oi_avg_30d",  False),
        "today's volume":               ("vol_total",   False),
        "put volume":                   ("vol_put",     False),
        "call volume":                  ("vol_call",    False),
        "volume avg (30-day)":          ("vol_avg_30d", False),
    }

    # Special: Put-Call Ratio appears for both OI and Volume sections
    pc_ratio_count = 0

    for i, line in enumerate(lines):
        ll = line.lower().strip()

        # Exact label match
        if ll in LABEL_MAP:
            key, is_pct = LABEL_MAP[ll]
            # Find next meaningful value in next few lines
            for j in range(i+1, min(i+6, len(lines))):
                val_line = lines[j]
                # Value must contain digits
                if not re.search(r'\d', val_line):
                    continue
                # Skip lines that look like another label (too many words)
                if len(val_line.split()) > 4 and not re.search(r'[\d,]+', val_line):
                    continue

                # Extract numeric value
                # Handle formats: "25.91%", "25.91", "3,992,452", "$3.69T"
                num = re.search(r'([\d,]+\.?\d*)', val_line.replace(",", ""))
                if num:
                    val = to_float(num.group(1))
                    if val is not None and result[key] is None:
                        result[key] = val
                        break

        # Put-Call Ratio (appears twice — first for OI, second for Volume)
        elif ll == "put-call ratio":
            for j in range(i+1, min(i+4, len(lines))):
                val_line = lines[j]
                num = re.search(r'([\d.]+)', val_line)
                if num:
                    val = to_float(num.group(1))
                    if val is not None:
                        if pc_ratio_count == 0 and result["oi_pc_ratio"] is None:
                            result["oi_pc_ratio"] = val
                            pc_ratio_count += 1
                        elif pc_ratio_count == 1 and result["vol_pc_ratio"] is None:
                            result["vol_pc_ratio"] = val
                            pc_ratio_count += 1
                        break

    # ── IV Low/High dates ─────────────────────────────────────────────────────
    iv_low_match = re.search(
        r'IV Low\s*([\d.]+)%\s*on\s*([\d/]+)', text, re.IGNORECASE
    )
    if iv_low_match:
        result["iv_low"]      = to_float(iv_low_match.group(1))
        result["iv_low_date"] = iv_low_match.group(2)

    iv_high_match = re.search(
        r'IV High\s*([\d.]+)%\s*on\s*([\d/]+)', text, re.IGNORECASE
    )
    if iv_high_match:
        result["iv_high"]      = to_float(iv_high_match.group(1))
        result["iv_high_date"] = iv_high_match.group(2)

    # ── Per-expiry stats table ────────────────────────────────────────────────
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        headers = []
        for row in rows[:2]:
            cells = [c.get_text(strip=True) for c in row.find_all(["th","td"])]
            cells = [c for c in cells if c]
            if cells and len(cells) > 3:
                headers = cells
                break

        for row in rows[1:]:
            cells = [re.sub(r'\s+', ' ', c.get_text()).strip()
                     for c in row.find_all(["td","th"])]
            cells = [c for c in cells if c]
            if len(cells) < 4:
                continue

            # First cell contains expiration like "Mar 25, 2026 (1 days) (w)"
            exp_raw = cells[0]
            exp_match = re.search(
                r'(\w+ \d+,?\s*\d{4})\s*\((\d+)\s*days?\)\s*\((\w)\)',
                exp_raw
            )
            if not exp_match:
                continue

            # Parse remaining numeric cells
            nums = []
            for c in cells[1:]:
                nums.append(to_float(c.replace(",","")))

            # Expected column order from the page:
            # call_vol | put_vol | pc_vol_ratio |
            # call_oi  | put_oi  | pc_oi_ratio  |
            # iv | expected_move | max_pain | max_pain_vs_price
            entry = {
                "expiry":         exp_match.group(1),
                "dte":            to_int(exp_match.group(2)),
                "type":           "monthly" if exp_match.group(3) == "m" else "weekly",
                "call_vol":       nums[0]  if len(nums) > 0 else None,
                "put_vol":        nums[1]  if len(nums) > 1 else None,
                "vol_pc_ratio":   nums[2]  if len(nums) > 2 else None,
                "call_oi":        nums[3]  if len(nums) > 3 else None,
                "put_oi":         nums[4]  if len(nums) > 4 else None,
                "oi_pc_ratio":    nums[5]  if len(nums) > 5 else None,
                "iv":             nums[6]  if len(nums) > 6 else None,
                "raw":            " | ".join(cells)  # keep raw for debugging
            }
            result["expiry_stats"].append(entry)

    # ── Parse quality notes ───────────────────────────────────────────────────
    missing = [k for k in ["iv_30","iv_rank","iv_pct","hv_30"] if result[k] is None]
    if missing:
        result["parse_notes"].append(
            f"Missing fields: {missing} — overview page may need async endpoint"
        )
    else:
        result["parse_notes"].append("All key IV fields parsed successfully")

    return result


# ── Chain parser ──────────────────────────────────────────────────────────────

def parse_chain(html, ticker):
    """
    Parse optioncharts straddle chain HTML.
    Expected endpoint: /async/option_chain?ticker=AAPL&expiration_dates=...
    
    Returns structured dict with full strike table.
    """
    soup = BeautifulSoup(html, "lxml")  # don't strip scripts yet — need price

    # ── Underlying price from JS ──────────────────────────────────────────────
    price_match = re.search(r'underlyingPrice[^\d]*([0-9.]+)', html)
    stock_price = to_float(price_match.group(1)) if price_match else None

    # ── Expiration info ───────────────────────────────────────────────────────
    exp_match = re.search(
        r'<b>([^<]+?),\s*(\w+ \d+,?\s*\d{4})\s*\((\d+)\s*days?\)\s*\((\w)\)</b>',
        html
    )
    expiry     = exp_match.group(2).strip() if exp_match else None
    dte        = to_int(exp_match.group(3))  if exp_match else None
    exp_type   = ("monthly" if exp_match and exp_match.group(4) == "m" else "weekly") if exp_match else None

    # Now strip scripts for cleaner parsing
    for tag in ["script","style","nav","head","footer","img"]:
        for el in soup.find_all(tag):
            el.decompose()

    # ── Find options table ────────────────────────────────────────────────────
    table = (
        soup.find("table", id="option-chain-straddle-table") or
        soup.find("table")
    )

    result = {
        "ticker":      ticker.upper(),
        "parsed_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source":      "optioncharts_chain",
        "expiry":      expiry,
        "dte":         dte,
        "expiry_type": exp_type,
        "stock_price": stock_price,
        "total_strikes": 0,
        "chain":       [],
        "parse_notes": []
    }

    if not table:
        result["parse_notes"].append("ERROR: No options table found")
        return result

    rows = table.find_all("tr")

    # ── Parse each strike row ─────────────────────────────────────────────────
    # Straddle format:
    # [Call Last | Call Bid | Call Ask | Call Vol | Call OI | Strike | Put Last | Put Bid | Put Ask | Put Vol | Put OI]

    strikes = []
    for row in rows[2:]:  # skip 2 header rows
        cells = [re.sub(r'\s+','', td.get_text()).strip()
                 for td in row.find_all(["td","th"])]
        cells = [c for c in cells if c]

        if len(cells) < 7:
            continue

        # Strike is cell index 5
        strike = to_float(cells[5])
        if strike is None or strike == 0:
            continue

        # ITM flag from row class
        row_classes = " ".join(row.get("class", []))
        is_itm = any(kw in row_classes.lower()
                     for kw in ["itm","highlight","in-the-money","tw-bg-blue"])

        def opt_side(last, bid, ask, vol, oi):
            bid_f  = to_float(bid)
            ask_f  = to_float(ask)
            mid    = round((bid_f + ask_f) / 2, 2) if bid_f is not None and ask_f is not None else None
            return {
                "last":   to_float(last),
                "bid":    bid_f,
                "ask":    ask_f,
                "mid":    mid,
                "volume": to_int(vol),
                "oi":     to_int(oi),
            }

        call = opt_side(cells[0], cells[1], cells[2], cells[3], cells[4])
        put  = opt_side(cells[6], cells[7], cells[8], cells[9], cells[10]) if len(cells) >= 11 else opt_side(None,None,None,None,None)

        # Moneyness % vs stock price
        moneyness = round((strike - stock_price) / stock_price * 100, 2) if stock_price else None

        strikes.append({
            "strike":      strike,
            "moneyness":   moneyness,
            "itm":         is_itm,
            "call":        call,
            "put":         put,
        })

    result["chain"]         = strikes
    result["total_strikes"] = len(strikes)

    if not strikes:
        result["parse_notes"].append("WARNING: No strikes parsed — check table structure")
    else:
        result["parse_notes"].append(f"Parsed {len(strikes)} strikes OK")
        if stock_price:
            result["parse_notes"].append(f"Stock price from JS: ${stock_price}")

    # ── DTE validation ────────────────────────────────────────────────────────
    # Warn if we got the wrong expiry (site ignored our date param)
    if dte is not None and dte < 10:
        result["parse_notes"].append(
            f"⚠️  DTE WARNING: Got {dte} DTE chain — expected ~45 DTE. "
            f"Site may have ignored expiration_dates param. "
            f"Re-run fetch.py with fresh session to get correct chain."
        )
        result["dte_warning"] = True
    elif dte is not None and abs(dte - 45) > 15:
        result["parse_notes"].append(
            f"⚠️  DTE MISMATCH: Got {dte} DTE, expected ~45 DTE. "
            f"Closest available expiry was used."
        )
    else:
        result["dte_warning"] = False

    return result


# ── Per-ticker orchestrator ───────────────────────────────────────────────────

def parse_ticker(ticker, target_date=None):
    ticker      = ticker.upper()
    target_date = target_date or TODAY
    ticker_dir  = DATA_DIR / target_date / ticker

    if not ticker_dir.exists():
        print(f"  ❌ No folder: {ticker_dir}")
        return {"ticker": ticker, "status": "NO_DATA"}

    print(f"  📂 {ticker_dir}")
    results = {"ticker": ticker, "date": target_date, "files": {}}

    SOURCES = [
        ("overview", "optioncharts_overview.html", "optioncharts_overview.json", parse_overview),
        ("chain",    "optioncharts_chain.html",    "optioncharts_chain.json",    parse_chain),
    ]

    for src_name, in_file, out_file, parse_fn in SOURCES:
        in_path  = ticker_dir / in_file
        out_path = ticker_dir / out_file

        if not in_path.exists():
            print(f"    ⚠️  {in_file} missing — skipping")
            results["files"][src_name] = {"status": "MISSING"}
            continue

        in_size = in_path.stat().st_size
        print(f"    🔍 {in_file} ({in_size:,} bytes) → ", end="")

        try:
            html   = in_path.read_text(encoding="utf-8")
            parsed = parse_fn(html, ticker)

            # Save JSON
            out_path.write_text(
                json.dumps(parsed, indent=2), encoding="utf-8"
            )
            out_size = out_path.stat().st_size
            pct      = round(out_size / in_size * 100)

            # Summary
            if src_name == "overview":
                iv_found = parsed.get("iv_30") is not None
                ivr_found = parsed.get("iv_rank") is not None
                n_expiry = len(parsed.get("expiry_stats", []))
                print(f"{out_file} ({out_size:,}b, {pct}% of original)")
                print(f"       IV:{parsed.get('iv_30')}% | IVR:{parsed.get('iv_rank')}% | "
                      f"HV:{parsed.get('hv_30')}% | IVP:{parsed.get('iv_pct')}% | "
                      f"Expiry rows:{n_expiry}")
                for note in parsed.get("parse_notes", []):
                    print(f"       ℹ️  {note}")
            else:
                n = parsed.get("total_strikes", 0)
                price = parsed.get("stock_price")
                print(f"{out_file} ({out_size:,}b, {pct}% of original)")
                print(f"       Strikes:{n} | Price:${price} | "
                      f"Expiry:{parsed.get('expiry')} ({parsed.get('dte')} DTE)")
                for note in parsed.get("parse_notes", []):
                    print(f"       ℹ️  {note}")

            results["files"][src_name] = {
                "status":   "OK",
                "in_bytes": in_size,
                "out_bytes":out_size,
                "reduction":f"{100-pct}%",
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            results["files"][src_name] = {"status": "ERROR", "error": str(e)}

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = sys.argv[1:]
    target_date = TODAY

    if "--date" in args:
        idx         = args.index("--date")
        target_date = args[idx + 1]
        args        = [a for i,a in enumerate(args) if i not in (idx, idx+1)]

    if args:
        tickers = [t.upper() for t in args]
    else:
        day_dir = DATA_DIR / target_date
        if not day_dir.exists():
            print(f"❌ No data for {target_date} — run fetch.py first")
            sys.exit(1)
        tickers = sorted([
            d.name for d in day_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    print(f"\n{'='*60}")
    print(f"🔬 Parser — {target_date} | {len(tickers)} ticker(s)")
    print(f"   Output: clean JSON per ticker")
    print(f"{'='*60}")

    all_results = []
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] {ticker}")
        all_results.append(parse_ticker(ticker, target_date))

    # Summary
    ok = sum(1 for r in all_results
             if all(v.get("status") == "OK" for v in r.get("files",{}).values()))
    print(f"\n{'='*60}")
    print(f"✅ {ok}/{len(all_results)} tickers fully parsed")
    print(f"📁 JSON files saved alongside HTML in {DATA_DIR / target_date}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()