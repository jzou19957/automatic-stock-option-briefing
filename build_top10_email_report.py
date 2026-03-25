import os
import json
import time
from pathlib import Path
from datetime import datetime

import requests

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
MODEL   = "gemini-2.0-flash-lite"
URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM PROMPT
#  Grounded in Tastytrade 45-DTE framework. Gemini's job is to write
#  SHORT, DIRECT, DESK-QUALITY commentary — not explain mechanics.
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_PROMPT = """
You are a senior options desk analyst at a proprietary trading firm.
Your audience is experienced premium sellers who know Tastytrade mechanics cold.
Do not explain what IV, IVR, delta, or theta mean. Just use them.

════════════════════════════════════════════════════
TASTYTRADE FRAMEWORK — USE THIS TO DRIVE COMMENTARY
════════════════════════════════════════════════════

STRATEGY SELECTION DOCTRINE (45 DTE, 3rd Friday monthly expiry):
  • Short Put Spread      → Bullish bias. Defined risk. Collect >= 1/3 spread width. Short delta ~0.25-0.30.
  • Short Call Spread     → Bearish bias. Defined risk. Collect >= 1/3 spread width. Short delta ~0.25-0.30.
  • Iron Condor           → Neutral. Defined both sides. Collect >= 1/3 widest spread. Match short deltas.
  • Short Strangle        → Neutral + IVR >= 55. Undefined risk. Both legs ~0.25 delta OTM.
  • Jade Lizard           → Bullish + put skew elevated. Naked put + OTM call spread. Total credit > call spread width = ZERO upside risk.
  • SKIP                  → Score < 2.0 or earnings within DTE.

STRIKE SELECTION:
  • Tastytrade target short delta: ~0.25 (16-30 delta range acceptable).
  • Never sell strikes closer than 1 sigma expected move unless IVR > 70.
  • Jade lizard: put strike just below 1-sigma downside; call spread just above ATM straddle level.
  • Iron condor: short strikes beyond ATM straddle; long wings approx 5 pts wide.

MANAGEMENT RULES (hardcoded discipline — do not deviate):
  • PROFIT TARGET : Close at 50% of original credit received.
  • TIME STOP     : Close at 21 DTE regardless of P/L — this is the management date.
  • EARLY ALERT   : If stock crosses 0.5 sigma toward tested side, begin watching for roll.
  • ROLLING       : Roll tested side out in time for a net credit if still OTM. Never roll ITM short for a debit.
  • EARNINGS      : If earnings fall within DTE window, use DEFINED risk only (spreads/condor). Never straddle/strangle through earnings.

EDGE QUALITY:
  • IV/HV >= 1.5 = strong edge. IV/HV 1.1-1.3 = marginal. < 1.1 = no edge, say so bluntly.
  • IVR >= 65 = elevated; >= 85 = top quartile — mention this explicitly.
  • IV/HV drives "are options rich vs. realized?" IVR drives "is this elevated for THIS ticker?"
  • Skew: put skew > 1.3 means OTM puts are rich — favors jade lizard or put spread.

════════════════════════════════════════════════════
OUTPUT SCHEMA — valid JSON only, no markdown, no preamble
════════════════════════════════════════════════════
{
  "ticker": str,
  "report_date": str,
  "top_row": {
    "premium_score": str,        // e.g. "7.2/10"
    "strategy_name": str,        // exact strategy or "SKIP — [one-line reason]"
    "current_price": float
  },
  "basic_stats": {
    "iv": str,                   // e.g. "34.2%"
    "ivr": str,                  // e.g. "71"
    "iv_hv": str,                // e.g. "1.48x"
    "bias": str,
    "recommended_strikes": str,  // e.g. "SELL $185P / BUY $180P · SELL $215C / BUY $220C" — use actual strikes from input
    "dte": int,
    "credit": str,               // e.g. "$2.35 cr"
    "management_date": str       // 21-DTE date
  },
  "management_notes": str,       // 2-3 sentences MAX. Specific prices and dates only.
  "position_summary": str        // 5-7 sentences. See style rules below.
}

════════════════════════════════════════════════════
CONTENT RULES
════════════════════════════════════════════════════

management_notes (2-3 sentences, hard limit):
  - State the take-profit credit target ($X).
  - State the management date and the action on that date.
  - State one specific alert level (0.5 sigma or breakeven). Nothing else.
  GOOD: "Take off at $X credit or better. Close the position by [mgmt_date] at 21 DTE regardless.
         Watch $Y on the downside — a close there starts the roll clock."
  BAD:  "Traders should consider taking profits when the position reaches approximately 50% of maximum potential..."

position_summary (5-7 sentences, desk voice):
  Sentence 1: Lead with the vol edge — state IV/HV and IVR as a single verdict. Is the premium rich or not?
  Sentence 2: What specifically drove the strategy choice — bias, skew, earnings safety, liquidity.
  Sentence 3: Describe the structure in one line: strikes, credit, defined/undefined, breakeven(s).
  Sentence 4: The primary risk factor (what kills this trade). Be specific.
  Sentence 5: Expected move context — does 1 sigma clear the short strike? By how much?
  Sentence 6 (optional): Skew or earnings note if relevant.
  Sentence 7: One-line conviction statement. Blunt. E.g. "This is a clean structure at fair credit — take it." or "Marginal setup; size accordingly."
  Final line: Always close with a one-sentence Lossdog/Tastytrade framework reference. Vary the wording naturally but always name both.
  Examples:
    "Per Lossdog/Tastytrade mechanics: 50% profit target, hard close at 21 DTE, roll only OTM for net credit."
    "Lossdog/Tastytrade rule: defined risk mandatory with earnings inside the window — no exceptions."
    "Standard Lossdog/Tastytrade pipeline: take 50% and step aside by [mgmt_date], no holding through expiry."
    "Lossdog/Tastytrade framework: size small, sell premium at elevated IV/HV, let theta do the work."

  NEVER use: "it is important to note", "traders should be aware", "in the context of",
             "should be considered", "provides an opportunity", "this setup represents"

STYLE EXAMPLES:
  BAD:  "NVDA currently carries a premium attractiveness score which should be judged in the
         context of its implied-versus-realized volatility relationship..."
  GOOD: "IV/HV at 1.61x with IVR 74 — options are meaningfully rich relative to recent realized,
         and the rank confirms this isn't a one-day spike. Bullish MA posture and flat put skew
         pushed this toward a short put spread rather than a condor..."

INPUT JSON:
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_latest_date_folder():
    date_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], reverse=True)
    if not date_dirs:
        raise FileNotFoundError("No dated data folders found")
    return date_dirs[0]


def build_email_llm_input(data):
    rec         = data.get("recommendation", {}) or {}
    stock_price = data.get("stock_price")
    one_sd_move = safe_get(data, "expected_move_levels", "1sd", "move", default=None)
    iv_30       = data.get("iv_30")
    hv_30       = data.get("hv_30")
    iv_hv       = round(iv_30 / hv_30, 2) if iv_30 and hv_30 and hv_30 > 0 else None

    upside_alert = downside_alert = None
    if isinstance(stock_price, (int, float)) and isinstance(one_sd_move, (int, float)):
        half_move      = one_sd_move / 2.0
        upside_alert   = round(stock_price + half_move, 2)
        downside_alert = round(stock_price - half_move, 2)

    return {
        "ticker":               data.get("ticker"),
        "report_date":          data.get("analysis_date"),
        "stock_price":          stock_price,
        "iv_30":                iv_30,
        "hv_30":                hv_30,
        "iv_hv_ratio":          iv_hv,
        "ivr":                  data.get("ivr"),
        "ivp":                  data.get("ivp"),
        "skew_ratio":           data.get("skew_ratio"),
        "bias":                 data.get("bias"),
        "attractiveness_score": data.get("attractiveness_score"),
        "attractiveness_label": data.get("attractiveness_label"),
        "earnings_note":        data.get("earnings_note"),
        "earnings_within_dte":  data.get("earnings_safe") is False,
        "next_earnings_date":   data.get("next_earnings_date"),
        "next_earnings_days":   data.get("next_earnings_days"),
        "target_expiry":        data.get("target_expiry"),
        "target_dte":           data.get("target_dte"),
        "management_date":      data.get("management_date"),
        "one_sd_move":          safe_get(data, "expected_move_levels", "1sd", "move"),
        "one_sd_move_pct":      safe_get(data, "expected_move_levels", "1sd", "move_pct"),
        "one_sd_upside":        safe_get(data, "expected_move_levels", "1sd", "upside"),
        "one_sd_downside":      safe_get(data, "expected_move_levels", "1sd", "downside"),
        "atm_straddle":         safe_get(data, "expected_move", "atm_straddle"),
        "atm_straddle_pct":     safe_get(data, "expected_move", "atm_straddle_pct"),
        "half_1sd_upside_alert":   upside_alert,
        "half_1sd_downside_alert": downside_alert,
        "recommended_strategy": {
            "strategy":         rec.get("strategy_name"),
            "reason":           rec.get("reason"),
            "risk_constraint":  rec.get("risk_constraint"),
            "expiry":           rec.get("expiry"),
            "dte":              rec.get("dte"),
            "legs":             rec.get("legs"),
            "credit":           rec.get("credit"),
            "max_risk":         rec.get("max_risk"),
            "roi_pct":          rec.get("roi_pct"),
            "breakeven":        rec.get("breakeven"),
            "breakeven_pct":    rec.get("breakeven_pct"),
            "take_profit_at":   rec.get("take_profit_at"),
            "management_date":  rec.get("management_date"),
            "liquidity_ok":     rec.get("liquidity_ok"),
            "meets_one_third":  rec.get("meets_one_third"),
            "short_delta":      safe_get(rec, "strategy_data", "short_delta"),
            "short_prob_otm":   safe_get(rec, "strategy_data", "short_prob_otm"),
            "em_safety_note":   safe_get(
                rec, "strategy_data", "management_rules", "safety_vs_em", "note", default=None
            ),
        },
    }


def fallback_email_json(clean_input):
    """Rule-based fallback when Gemini is unavailable — terse, specific, no filler."""
    strat      = clean_input.get("recommended_strategy", {}) or {}
    score      = clean_input.get("attractiveness_score")
    score_text = f"{score}/10" if score is not None else "N/A"
    iv_30      = clean_input.get("iv_30")
    hv_30      = clean_input.get("hv_30")
    iv_hv      = clean_input.get("iv_hv_ratio")
    ivr        = clean_input.get("ivr")
    bias       = clean_input.get("bias", "N/A")
    ticker     = clean_input.get("ticker", "")

    strategy_name = strat.get("strategy") or "NONE — SKIP"
    legs          = strat.get("legs")     or "N/A"
    tp            = strat.get("take_profit_at")
    mgmt_date     = strat.get("management_date")
    credit        = strat.get("credit")
    breakeven     = strat.get("breakeven")
    em_note       = strat.get("em_safety_note", "")
    down_alert    = clean_input.get("half_1sd_downside_alert")
    up_alert      = clean_input.get("half_1sd_upside_alert")
    one_sd_down   = clean_input.get("one_sd_downside")
    one_sd_up     = clean_input.get("one_sd_upside")

    # ── management_notes: specific, max 3 sentences ──────────────────────────
    mgmt_parts = []
    if tp is not None and credit is not None:
        mgmt_parts.append(f"Take off at ${tp} credit (50% of ${credit}).")
    elif tp is not None:
        mgmt_parts.append(f"Take off at ${tp} credit target.")
    if mgmt_date:
        mgmt_parts.append(f"Hard close by {mgmt_date} at 21 DTE.")
    if down_alert is not None:
        mgmt_parts.append(
            f"Alert at ${down_alert} downside / ${up_alert} upside — 0.5-sigma breach triggers roll review."
        )
    elif em_note:
        mgmt_parts.append(em_note)
    management_notes = " ".join(mgmt_parts) if mgmt_parts else \
        "Close at 50% credit. Hard stop at 21 DTE. Roll only if still OTM and for net credit."

    # ── position_summary: structured, blunt ─────────────────────────────────
    iv_hv_str = f"{iv_hv}x" if iv_hv else "N/A"
    edge_line = (
        f"IV/HV {iv_hv_str} with IVR {ivr} — "
        + ("options are rich relative to realized vol, edge favors selling." if iv_hv and iv_hv >= 1.3
           else "vol premium is marginal; edge is thin.")
    )
    be_line   = f" Breakeven ${breakeven}." if breakeven else ""
    em_line   = (
        f" 1-sigma expected move puts the downside at ${one_sd_down} and upside at ${one_sd_up}."
        if one_sd_down else ""
    )
    earn_line = ""
    if clean_input.get("earnings_within_dte"):
        earn_line = f" Earnings fall within the {strat.get('dte', '')} DTE window — defined risk mandatory."
    elif clean_input.get("next_earnings_days"):
        earn_line = f" Next earnings in {clean_input['next_earnings_days']} days — clear of this cycle."

    conviction = (
        "Clean structure at current vol levels — execute at mid or better."
        if score and score >= 6.5
        else ("Marginal setup — size down, manage early." if score and score >= 4.0
              else "Low edge — skip or wait for a better vol entry.")
    )

    framework_line = (
        f"Per Lossdog/Tastytrade mechanics: 50% profit target, hard close at 21 DTE ({mgmt_date}), roll only OTM for net credit."
        if mgmt_date
        else "Per Lossdog/Tastytrade mechanics: 50% profit target, hard close at 21 DTE, roll only OTM for net credit."
    )

    position_summary = (
        f"{edge_line} "
        f"{bias} posture drove selection of {strategy_name}."
        f" {legs}.{be_line}{em_line}{earn_line} {conviction} {framework_line}"
    )

    return {
        "ticker":      ticker,
        "report_date": clean_input.get("report_date"),
        "top_row": {
            "premium_score":  score_text,
            "strategy_name":  strategy_name,
            "current_price":  clean_input.get("stock_price"),
        },
        "basic_stats": {
            "iv":                  f"{iv_30}%" if iv_30 else "N/A",
            "ivr":                 str(ivr) if ivr else "N/A",
            "iv_hv":               iv_hv_str,
            "bias":                bias,
            "recommended_strikes": legs,
            "dte":                 strat.get("dte"),
            "credit":              f"${credit} cr" if credit else "N/A",
            "management_date":     mgmt_date or "N/A",
        },
        "management_notes": management_notes,
        "position_summary": position_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini call
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(clean_data, max_retries=5):
    if not API_KEY:
        return {"status": "ERROR", "ticker": clean_data.get("ticker"), "error": "Missing GEMINI_API_KEY"}

    payload = {
        "contents": [{
            "parts": [{"text": EMAIL_PROMPT + json.dumps(clean_data, ensure_ascii=False)}]
        }],
        "generationConfig": {
            "temperature":     0.45,   # raised from 0.25 — prevents generic register
            "maxOutputTokens": 2400
        },
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{URL}?key={API_KEY}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                if attempt < max_retries:
                    wait = min(2 ** attempt, 20)
                    print(f"  Gemini {response.status_code} for {clean_data.get('ticker')} — retry {attempt}/{max_retries} in {wait}s")
                    time.sleep(wait)
                    continue
                return {
                    "status": "ERROR",
                    "ticker": clean_data.get("ticker"),
                    "error":  f"Gemini failed after {max_retries} attempts",
                    "details": last_error,
                }

            response.raise_for_status()
            raw   = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = str(e)
            if attempt < max_retries:
                wait = min(2 ** attempt, 20)
                print(f"  Gemini connection error for {clean_data.get('ticker')} — retry {attempt}/{max_retries} in {wait}s")
                time.sleep(wait)
                continue
            return {
                "status": "ERROR",
                "ticker": clean_data.get("ticker"),
                "error":  f"Gemini failed after {max_retries} attempts",
                "details": last_error,
            }

        except Exception as e:
            return {
                "status":  "ERROR",
                "ticker":  clean_data.get("ticker"),
                "error":   "Unexpected Gemini error",
                "details": str(e),
            }


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_top10():
    latest     = get_latest_date_folder()
    calculated = []
    for ticker_dir in sorted(latest.iterdir()):
        path = ticker_dir / "calculated.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("attractiveness_score")
        if isinstance(score, (int, float)):
            calculated.append(data)
    ranked = sorted(calculated, key=lambda x: x.get("attractiveness_score", 0), reverse=True)
    return latest, ranked[:10]


# ─────────────────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_num(value, decimals=2, prefix=""):
    if isinstance(value, (int, float)):
        return f"{prefix}{value:.{decimals}f}"
    return str(value) if value else "—"


def score_to_pct(score_text):
    try:
        return max(0, min(float(str(score_text).split("/")[0]) * 10, 100))
    except Exception:
        return 0.0


def score_band(score_text):
    """Return (bar_color, tier_label) based on score."""
    pct = score_to_pct(score_text)
    if pct >= 75:
        return "#10b981", "EXCELLENT"
    if pct >= 60:
        return "#3b82f6", "GOOD"
    if pct >= 45:
        return "#f59e0b", "MODERATE"
    if pct >= 30:
        return "#f97316", "WEAK"
    return "#94a3b8", "SKIP"


def strategy_label_style(strategy_name):
    """Background / foreground / border for strategy chip."""
    name = (strategy_name or "").upper()
    if any(x in name for x in ("SKIP", "NONE")):
        return "#f1f5f9", "#64748b", "#cbd5e1"
    if "CONDOR" in name:
        return "#eff6ff", "#1d4ed8", "#bfdbfe"
    if "STRANGLE" in name or "STRADDLE" in name:
        return "#fdf4ff", "#7e22ce", "#e9d5ff"
    if "JADE" in name or "LIZARD" in name:
        return "#f0fdf4", "#15803d", "#bbf7d0"
    return "#fff7ed", "#c2410c", "#fed7aa"

# ─────────────────────────────────────────────────────────────────────────────
#  Card builder  —  pure <table> layout, matches approved preview
# ─────────────────────────────────────────────────────────────────────────────

def build_card(item, rank):
    top    = item.get("top_row", {})
    stats  = item.get("basic_stats", {})
    ticker = item.get("ticker", "???")

    score_text                = top.get("premium_score", "—")
    strategy_name             = top.get("strategy_name", "—")
    score_pct                 = score_to_pct(score_text)
    bar_color, tier           = score_band(score_text)
    chip_bg, chip_fg, chip_bd = strategy_label_style(strategy_name)
    price_display             = fmt_num(top.get("current_price"), 2, "$")

    is_skip = any(x in strategy_name.upper() for x in ("SKIP", "NONE"))

    # ── stat cell helper ─────────────────────────────────────────────────────
    def sc(label, value, accent=False):
        vc = "#1d4ed8" if accent else "#0f172a"
        return (
            f'<tr>'
            f'<td style="padding:7px 12px 7px 0;font-size:10.5px;color:#64748b;'
            f'font-family:\'Courier New\',monospace;border-bottom:1px solid #f1f5f9;'
            f'white-space:nowrap;">{label}</td>'
            f'<td style="padding:7px 0 7px 12px;font-size:10.5px;font-weight:700;color:{vc};'
            f'font-family:\'Courier New\',monospace;text-align:right;'
            f'border-bottom:1px solid #f1f5f9;">{value}</td>'
            f'</tr>'
        )

    # ── left stats block (IV, IVR, IV/HV, BIAS) ──────────────────────────────
    left_stats = (
        sc("IV (30d)", stats.get("iv",    "—")) +
        sc("IVR",      stats.get("ivr",   "—")) +
        sc("IV / HV",  stats.get("iv_hv", "—"), accent=True) +
        sc("BIAS",     stats.get("bias",  "—"))
    )

    # ── right stats block (DTE, CREDIT, BREAKEVEN, MGMT DATE) ────────────────
    breakeven = stats.get("breakeven", "")
    be_cells  = sc("BREAKEVEN", breakeven) if breakeven and not is_skip else ""
    right_stats = (
        sc("DTE",       stats.get("dte",             "—")) +
        sc("CREDIT",    stats.get("credit",          "—"), accent=True) +
        be_cells +
        sc("MGMT DATE", stats.get("management_date", "—"))
    )

    # ── score bar (table-based, no CSS width tricks) ──────────────────────────
    filled_w = max(1, int(score_pct))
    empty_w  = 100 - filled_w
    score_bar = (
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;">'
        f'<tr>'
        f'<td style="font-size:9px;font-family:\'Courier New\',monospace;'
        f'color:#94a3b8;letter-spacing:1px;">PREMIUM SCORE</td>'
        f'<td style="font-size:9.5px;font-family:\'Courier New\',monospace;'
        f'font-weight:700;color:{bar_color};text-align:right;">'
        f'{score_text} &middot; {tier}</td>'
        f'</tr>'
        f'<tr><td colspan="2" style="padding-top:5px;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="background:#e2e8f0;">'
        f'<tr>'
        f'<td style="width:{filled_w}%;height:3px;background:{bar_color};font-size:0;">&nbsp;</td>'
        f'<td style="width:{empty_w}%;height:3px;font-size:0;">&nbsp;</td>'
        f'</tr></table>'
        f'</td></tr>'
        f'</table>'
    )

    # ── structure block ───────────────────────────────────────────────────────
    strikes = stats.get("recommended_strikes", "—")
    structure_block = "" if is_skip else (
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:#f8fafc;border:1px solid #e2e8f0;margin-bottom:10px;">'
        f'<tr><td style="padding:8px 12px 3px 12px;font-size:9px;'
        f'font-family:\'Courier New\',monospace;color:#94a3b8;letter-spacing:1.5px;">STRUCTURE</td></tr>'
        f'<tr><td style="padding:2px 12px 10px 12px;font-size:12px;'
        f'font-family:\'Courier New\',Courier,monospace;color:#0f172a;line-height:1.6;">'
        f'{strikes}</td></tr>'
        f'</table>'
    )

    # ── management block ──────────────────────────────────────────────────────
    mgmt_notes = item.get("management_notes", "")
    mgmt_block = "" if is_skip else (
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="border-left:3px solid #cbd5e1;background:#f8fafc;margin-bottom:10px;">'
        f'<tr><td style="padding:8px 12px 3px 12px;font-size:9px;'
        f'font-family:\'Courier New\',monospace;color:#94a3b8;letter-spacing:1.5px;">MANAGEMENT</td></tr>'
        f'<tr><td style="padding:2px 12px 10px 12px;font-size:12px;'
        f'line-height:1.65;color:#334155;">{mgmt_notes}</td></tr>'
        f'</table>'
    )

    # ── analyst note block ────────────────────────────────────────────────────
    analyst_note = item.get("position_summary", "—")
    note_block = (
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="border-left:3px solid #3b82f6;background:#f8fafc;">'
        f'<tr><td style="padding:8px 12px 3px 12px;font-size:9px;'
        f'font-family:\'Courier New\',monospace;color:#94a3b8;letter-spacing:1.5px;">ANALYST NOTE</td></tr>'
        f'<tr><td style="padding:2px 12px 12px 12px;font-size:12px;'
        f'line-height:1.7;color:#1e293b;">{analyst_note}</td></tr>'
        f'</table>'
    )

    # ── assemble card ─────────────────────────────────────────────────────────
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:#ffffff;border:1px solid #e2e8f0;margin-bottom:12px;">'

        # header row
        f'<tr><td style="padding:14px 20px 12px 20px;border-bottom:2px solid #0f172a;">'
        f'<table width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td>'
        f'<span style="font-size:9px;font-family:\'Courier New\',monospace;color:#94a3b8;'
        f'font-weight:700;letter-spacing:1px;vertical-align:middle;">#{rank}&nbsp;&nbsp;</span>'
        f'<span style="font-size:24px;font-weight:900;color:#0f172a;'
        f'font-family:\'Courier New\',Courier,monospace;letter-spacing:-0.5px;'
        f'vertical-align:middle;">{ticker}</span>'
        f'<span style="font-size:13px;font-weight:600;color:#64748b;'
        f'font-family:\'Courier New\',monospace;vertical-align:middle;">'
        f'&nbsp;&nbsp;{price_display}</span>'
        f'</td>'
        f'<td style="text-align:right;vertical-align:middle;">'
        f'<span style="background:{chip_bg};color:{chip_fg};border:1px solid {chip_bd};'
        f'padding:4px 12px;font-size:10.5px;font-weight:700;'
        f'font-family:\'Courier New\',monospace;letter-spacing:0.3px;white-space:nowrap;">'
        f'{strategy_name}</span>'
        f'</td>'
        f'</tr></table>'
        f'</td></tr>'

        # two-column stats row
        f'<tr><td style="padding:0 20px;">'
        f'<table width="100%" cellpadding="0" cellspacing="0">'
        f'<tr>'
        f'<td style="width:50%;vertical-align:top;padding:0;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        f'{left_stats}</table></td>'
        f'<td style="width:50%;vertical-align:top;padding:0;">'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        f'{right_stats}</table></td>'
        f'</tr></table>'
        f'</td></tr>'

        # score bar
        f'<tr><td style="padding:0 20px 12px 20px;">{score_bar}</td></tr>'

        # structure + management + analyst note
        f'<tr><td style="padding:0 20px 16px 20px;">'
        f'{structure_block}{mgmt_block}{note_block}'
        f'</td></tr>'

        f'</table>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Email wrapper  —  clean header matching approved preview
# ─────────────────────────────────────────────────────────────────────────────

def build_html_email(top10_email_json, as_of_date):
    cards_html = "".join(
        build_card(item, rank=i + 1) for i, item in enumerate(top10_email_json)
    )

    # top 3 tickers for header
    top3 = [item.get("ticker", "") for item in top10_email_json[:3]]
    rest = len(top10_email_json) - 3
    symbols_line = " &nbsp;&middot;&nbsp; ".join(top3)
    if rest > 0:
        symbols_line += f" &nbsp;&middot;&nbsp; +{rest} more"

    # human-readable date
    try:
        dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        date_display = dt.strftime("%B %-d, %Y")
    except Exception:
        date_display = as_of_date

    # email subject line (used by send_top10_email.py via file read — kept as HTML comment)
    subject_tickers = ", ".join(item.get("ticker", "") for item in top10_email_json[:3])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Top 10 Premium Selling: {subject_tickers} &amp; more · {as_of_date}</title>
</head>
<body style="margin:0;padding:0;background:#e2e8f0;font-family:system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;">
<div style="max-width:700px;margin:28px auto 48px auto;">

  <!-- HEADER -->
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;">
    <tr>
      <td style="padding:26px 32px 8px 32px;">
        <div style="font-size:20px;font-weight:800;color:#f8fafc;letter-spacing:-0.3px;line-height:1.2;">
          Lossdog Top 10 Premium Selling Setups
        </div>
        <div style="font-size:11px;color:#64748b;font-family:'Courier New',monospace;margin-top:5px;">
          From 50 most liquid mega-cap &amp; ETF universe &nbsp;&middot;&nbsp; 45 DTE monthly expiry
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding:10px 32px;">
        <div style="height:1px;background:#1e293b;"></div>
      </td>
    </tr>
    <tr>
      <td style="padding:0 32px 14px 32px;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="font-size:11px;font-family:'Courier New',monospace;color:#94a3b8;">
              {date_display}
            </td>
            <td style="text-align:right;">
              <span style="font-size:10px;font-family:'Courier New',monospace;color:#475569;">
                {symbols_line}
              </span>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    <tr>
      <td style="padding:0 32px;">
        <div style="height:2px;background:linear-gradient(90deg,#3b82f6 0%,#1e40af 60%,transparent 100%);"></div>
      </td>
    </tr>
    <tr>
      <td style="padding:8px 32px 10px 32px;">
        <table cellpadding="0" cellspacing="0">
          <tr>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#475569;padding-right:14px;">SCORE:</td>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#10b981;padding-right:12px;">&#9646; &ge;7.5 EXCELLENT</td>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#3b82f6;padding-right:12px;">&#9646; 6&ndash;7.5 GOOD</td>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#f59e0b;padding-right:12px;">&#9646; 4.5&ndash;6 MODERATE</td>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#f97316;padding-right:12px;">&#9646; 3&ndash;4.5 WEAK</td>
            <td style="font-size:9px;font-family:'Courier New',monospace;color:#64748b;">&#9646; &lt;3 SKIP</td>
          </tr>
        </table>
      </td>
    </tr>
  </table>

  <!-- CARDS -->
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#e8ecf0;">
    <tr><td style="padding:16px 20px 4px 20px;">
      {cards_html}
    </td></tr>
  </table>

  <!-- FOOTER -->
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;border-top:1px solid #1e293b;">
    <tr>
      <td style="padding:12px 32px;">
        <div style="font-size:9px;font-family:'Courier New',monospace;color:#334155;">
          Lossdog Research &nbsp;&middot;&nbsp; Not investment advice &nbsp;&middot;&nbsp; For personal use only
        </div>
      </td>
    </tr>
  </table>

</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    latest, top10 = load_top10()
    if not top10:
        raise RuntimeError("No ranked calculated.json files found")

    email_json     = []
    failed_tickers = []

    for data in top10:
        clean_input = build_email_llm_input(data)
        ticker      = clean_input.get("ticker", "?")
        print(f"  Gemini -> {ticker}")
        out = call_gemini(clean_input)

        if isinstance(out, dict) and out.get("status") == "ERROR":
            print(f"  WARNING: Gemini failed for {ticker} — using rule-based fallback")
            failed_tickers.append(ticker)
            out = fallback_email_json(clean_input)

        email_json.append(out)

    if not email_json:
        raise RuntimeError("No email content available")

    today_str = datetime.now().strftime("%Y-%m-%d")
    html      = build_html_email(email_json, today_str)

    # dynamic email subject for send_top10_email.py
    top3_symbols = ", ".join(item.get("ticker", "") for item in email_json[:3])
    subject_hint = f"Top 10 Premium Selling: {top3_symbols} & more — {today_str}"

    (REPORT_DIR / "top10_email_payload.json").write_text(
        json.dumps(email_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORT_DIR / "top10_email.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "top10_email_subject.txt").write_text(subject_hint, encoding="utf-8")
    (REPORT_DIR / "top10_email_debug.json").write_text(
        json.dumps({
            "source_date_folder":           str(latest),
            "failed_tickers_used_fallback": failed_tickers,
            "generated_at":                 today_str,
            "count":                        len(email_json),
            "email_subject":                subject_hint,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved to {REPORT_DIR}")
    print(f"  top10_email.html        ({(REPORT_DIR / 'top10_email.html').stat().st_size:,} bytes)")
    print(f"  top10_email_subject.txt  {subject_hint}")
    if failed_tickers:
        print(f"  WARNING fallback used:  {', '.join(failed_tickers)}")


if __name__ == "__main__":
    main()