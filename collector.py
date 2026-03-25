"""
collector.py — Collects ALL data for one ticker:
  1. Options chain from optioncharts.io (45 DTE)
  2. IV / HV / IVR / IVP from barchart.com
  3. Price / 21-day MA / earnings date from yfinance

Usage:
    python collector.py MSFT
    
Requirements:
    pip install requests beautifulsoup4 yfinance pandas
"""

import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from bs4 import BeautifulSoup

# ── Headers ───────────────────────────────────────────────────────────────────

HEADERS_BASIC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HEADERS_HTMX = {
    **HEADERS_BASIC,
    "Referer": "https://optioncharts.io/",
    "HX-Request": "true",
    "HX-Target": "option-charts-main-content-container",
}

OUTPUT_DIR = Path("./output")


# ── Date helpers ──────────────────────────────────────────────────────────────

def get_third_friday(year, month):
    d = datetime(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_until_friday)
    return (first_friday + timedelta(weeks=2)).date()


def find_closest_to_45dte():
    today = date.today()
    target = today + timedelta(days=45)
    fridays = []
    d = today + timedelta(days=1)
    while (d - today).days <= 100:
        if d.weekday() == 4:
            fridays.append(d)
        d += timedelta(days=1)
    closest = min(fridays, key=lambda x: abs((x - target).days))
    return closest, (closest - today).days


# ── 1. OPTIONS CHAIN — optioncharts.io ───────────────────────────────────────

def fetch_options_chain(ticker, expiry, dte, session):
    """Fetch and parse the options chain from optioncharts.io async endpoint."""
    date_str = expiry.strftime('%Y-%m-%d')
    third_fri = get_third_friday(expiry.year, expiry.month)
    suffix = "%3Am" if expiry == third_fri else ""

    url = (
        f"https://optioncharts.io/async/option_chain"
        f"?expiration_dates={date_str}{suffix}"
        f"&option_type=all&strike_range=all"
        f"&ticker={ticker.upper()}&view=straddle"
    )

    print(f"  📡 Options chain: {url}")

    try:
        resp = session.get(url, headers=HEADERS_HTMX, timeout=30)
        resp.raise_for_status()
        html = resp.text
        print(f"     ✅ Got {len(html):,} chars")
    except requests.exceptions.HTTPError as e:
        print(f"     ❌ HTTP error: {e.response.status_code} — {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"     ❌ Connection error (blocked or offline): {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"     ❌ Timeout fetching options chain")
        return None
    except Exception as e:
        print(f"     ❌ Unexpected error: {e}")
        return None

    # Parse HTML table
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="option-chain-straddle-table") or soup.find("table")

    if not table:
        print(f"     ⚠️  No table found — site may have changed or blocked request")
        # Save raw for inspection
        (OUTPUT_DIR / f"{ticker}_raw_response.html").write_text(html, encoding="utf-8")
        print(f"     💾 Raw HTML saved for inspection")
        return None

    rows = table.find_all("tr")
    chains = []

    for row in rows[2:]:
        cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cols) < 11:
            continue

        def clean(val):
            val = str(val).replace(",", "").strip()
            if val in ["-", "", "N/A"]:
                return 0.0
            try:
                return float(val)
            except:
                return 0.0

        try:
            strike = clean(cols[5])
            if strike == 0:
                continue
            chains.append({
                "strike": strike,
                "call": {
                    "bid": clean(cols[1]), "ask": clean(cols[2]),
                    "volume": int(clean(cols[3])), "oi": int(clean(cols[4])),
                    "mid": round((clean(cols[1]) + clean(cols[2])) / 2, 2),
                },
                "put": {
                    "bid": clean(cols[7]), "ask": clean(cols[8]),
                    "volume": int(clean(cols[9])), "oi": int(clean(cols[10])),
                    "mid": round((clean(cols[7]) + clean(cols[8])) / 2, 2),
                }
            })
        except Exception:
            continue

    if not chains:
        print(f"     ⚠️  Parsed 0 rows — check raw HTML")
        return None

    # Estimate ATM price from highest volume strike
    atm = max(chains, key=lambda r: r["call"]["volume"] + r["put"]["volume"])
    stock_price = atm["strike"]

    # Add moneyness + premium%
    for row in chains:
        s = row["strike"]
        row["moneyness_pct"] = round((s - stock_price) / stock_price * 100, 2)
        row["call"]["premium_pct"] = round(row["call"]["mid"] / stock_price * 100, 3)
        row["put"]["premium_pct"]  = round(row["put"]["mid"]  / stock_price * 100, 3)

    print(f"     📊 Parsed {len(chains)} strikes | ATM est: ${stock_price}")
    return {
        "expiration": expiry.strftime("%Y-%m-%d"),
        "dte": dte,
        "est_stock_price": stock_price,
        "total_strikes": len(chains),
        "chain": chains
    }


# ── 2. IV / HV / IVR — barchart.com ─────────────────────────────────────────

def fetch_barchart_iv(ticker, session):
    """Scrape IV, HV, IVR, IVP from barchart.com overview page."""
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    print(f"  📡 Barchart IV: {url}")

    result = {
        "iv_30": None, "hv_30": None,
        "ivr": None, "ivp": None,
        "source": "barchart"
    }

    try:
        # Barchart needs a real browser session — use session with cookies
        session.get("https://www.barchart.com/", headers=HEADERS_BASIC, timeout=15)
        time.sleep(1)

        resp = session.get(url, headers=HEADERS_BASIC, timeout=30)
        resp.raise_for_status()
        html = resp.text
        print(f"     ✅ Got {len(html):,} chars")

        soup = BeautifulSoup(html, "html.parser")

        # Barchart puts options data in specific spans/divs
        # Look for "Implied Volatility", "Historical Volatility", "IV Rank", "IV Percentile"
        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        def extract_pct_after(keyword, lines):
            """Find a percentage value after a keyword line."""
            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    # Check next few lines for a percentage
                    for j in range(i+1, min(i+5, len(lines))):
                        val = lines[j].replace("%", "").replace(",", "").strip()
                        try:
                            return float(val)
                        except:
                            continue
            return None

        result["iv_30"] = extract_pct_after("Implied Volatility", lines)
        result["hv_30"] = extract_pct_after("Historical Volatility", lines)
        result["ivr"]   = extract_pct_after("IV Rank", lines)
        result["ivp"]   = extract_pct_after("IV Percentile", lines)

        print(f"     📊 IV:{result['iv_30']}% HV:{result['hv_30']}% IVR:{result['ivr']} IVP:{result['ivp']}")

        # If all None, barchart likely blocked us
        if all(v is None for v in [result["iv_30"], result["hv_30"], result["ivr"]]):
            print(f"     ⚠️  Could not parse IV data — barchart may require login")
            # Save for inspection
            (OUTPUT_DIR / f"{ticker}_barchart.html").write_text(html, encoding="utf-8")
            print(f"     💾 Saved barchart HTML for inspection")

    except requests.exceptions.HTTPError as e:
        print(f"     ❌ HTTP {e.response.status_code} — barchart blocked or rate limited")
    except requests.exceptions.ConnectionError:
        print(f"     ❌ Connection error — barchart unreachable")
    except requests.exceptions.Timeout:
        print(f"     ❌ Timeout — barchart slow")
    except Exception as e:
        print(f"     ❌ Unexpected: {e}")

    return result


# ── 3. PRICE / MA / EARNINGS — yfinance ──────────────────────────────────────

def fetch_yfinance_data(ticker):
    """Get current price, 21-day MA, and next earnings date from yfinance."""
    print(f"  📡 yfinance: {ticker}")

    result = {
        "current_price": None,
        "ma_21": None,
        "price_vs_ma_pct": None,
        "next_earnings_date": None,
        "next_earnings_days": None,
        "source": "yfinance"
    }

    try:
        stock = yf.Ticker(ticker)

        # Get 30 days of history for MA calculation
        hist = stock.history(period="60d")
        if hist.empty:
            print(f"     ❌ No price history returned")
            return result

        current_price = round(float(hist["Close"].iloc[-1]), 2)
        ma_21 = round(float(hist["Close"].tail(21).mean()), 2)
        price_vs_ma_pct = round((current_price - ma_21) / ma_21 * 100, 2)

        result["current_price"]    = current_price
        result["ma_21"]            = ma_21
        result["price_vs_ma_pct"]  = price_vs_ma_pct

        print(f"     📊 Price: ${current_price} | MA21: ${ma_21} | vs MA: {price_vs_ma_pct:+.2f}%")

        # Get earnings date
        try:
            cal = stock.calendar
            if cal is not None and not cal.empty:
                # Calendar is a DataFrame — earnings date is in columns
                if "Earnings Date" in cal.index:
                    earn_date = cal.loc["Earnings Date"].iloc[0]
                elif hasattr(cal, 'columns') and len(cal.columns) > 0:
                    earn_date = cal.iloc[0, 0]
                else:
                    earn_date = None

                if earn_date is not None:
                    if hasattr(earn_date, 'date'):
                        earn_date = earn_date.date()
                    elif isinstance(earn_date, str):
                        earn_date = datetime.strptime(earn_date[:10], "%Y-%m-%d").date()

                    days_to_earnings = (earn_date - date.today()).days
                    result["next_earnings_date"] = str(earn_date)
                    result["next_earnings_days"] = days_to_earnings
                    print(f"     📅 Earnings: {earn_date} ({days_to_earnings} days)")
        except Exception as e:
            print(f"     ⚠️  Earnings date unavailable: {e}")

    except Exception as e:
        print(f"     ❌ yfinance error: {e}")

    return result


# ── 4. MAIN COLLECTOR ─────────────────────────────────────────────────────────

def collect(ticker):
    """
    Collect ALL data for one ticker and save as JSON.
    Returns the combined data dict, or None if critical data missing.
    """
    ticker = ticker.upper()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔍 Collecting data for {ticker}")
    print(f"{'='*60}")

    expiry, dte = find_closest_to_45dte()
    print(f"📅 Target expiry: {expiry} ({dte} DTE)\n")

    session = requests.Session()

    # Step 1: Options chain
    print(f"[1/3] Options Chain")
    options_data = fetch_options_chain(ticker, expiry, dte, session)

    # Step 2: IV data from Barchart
    print(f"\n[2/3] IV / HV / IVR")
    time.sleep(1)
    iv_data = fetch_barchart_iv(ticker, session)

    # Step 3: Price + MA + Earnings from yfinance
    print(f"\n[3/3] Price / MA / Earnings")
    yf_data = fetch_yfinance_data(ticker)

    # ── Combine everything ────────────────────────────────────────────────────
    combined = {
        "ticker":       ticker,
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "date":         str(date.today()),

        # Stock metrics
        "current_price":       yf_data["current_price"],
        "ma_21":               yf_data["ma_21"],
        "price_vs_ma_pct":     yf_data["price_vs_ma_pct"],
        "next_earnings_date":  yf_data["next_earnings_date"],
        "next_earnings_days":  yf_data["next_earnings_days"],

        # IV metrics
        "iv_30":  iv_data["iv_30"],
        "hv_30":  iv_data["hv_30"],
        "ivr":    iv_data["ivr"],
        "ivp":    iv_data["ivp"],

        # Options chain
        "options": options_data,

        # Data quality flags
        "has_options":  options_data is not None,
        "has_iv":       iv_data["iv_30"] is not None,
        "has_price":    yf_data["current_price"] is not None,
    }

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_file = OUTPUT_DIR / f"{ticker}_collected.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"✅ Data collection complete for {ticker}")
    print(f"   Options  : {'✅' if combined['has_options'] else '❌ MISSING'}")
    print(f"   IV/HV    : {'✅' if combined['has_iv'] else '⚠️  MISSING (will estimate)'}")
    print(f"   Price/MA : {'✅' if combined['has_price'] else '❌ MISSING'}")
    print(f"💾 Saved: {out_file}")

    return combined


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "MSFT"
    result = collect(ticker)
    if result:
        print(f"\n📋 Quick summary:")
        print(f"   Price     : ${result['current_price']}")
        print(f"   MA21      : ${result['ma_21']}")
        print(f"   vs MA     : {result['price_vs_ma_pct']:+.2f}%")
        print(f"   IV        : {result['iv_30']}%")
        print(f"   HV        : {result['hv_30']}%")
        print(f"   IVR       : {result['ivr']}")
        print(f"   Earnings  : {result['next_earnings_date']} ({result['next_earnings_days']} days)")
        if result['options']:
            print(f"   Strikes   : {result['options']['total_strikes']}")
