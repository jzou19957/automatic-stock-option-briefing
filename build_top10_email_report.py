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

    position_summary = (
        f"{edge_line} "
        f"{bias} posture drove selection of {strategy_name}."
        f" {legs}.{be_line}{em_line}{earn_line} {conviction}"
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
#  Card builder  ·  Institutional-grade, Bloomberg terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────

def build_card(item, rank):
    top    = item.get("top_row", {})
    stats  = item.get("basic_stats", {})
    ticker = item.get("ticker", "???")

    score_text            = top.get("premium_score", "—")
    strategy_name         = top.get("strategy_name", "—")
    score_pct             = score_to_pct(score_text)
    bar_color, tier       = score_band(score_text)
    chip_bg, chip_fg, chip_bd = strategy_label_style(strategy_name)
    price_display         = fmt_num(top.get("current_price"), 2, "$")

    def stat_row(label, value, accent=False):
        val_color = "#1d4ed8" if accent else "#0f172a"
        return f"""
        <tr>
          <td style="padding:6px 0 6px 0;font-size:11px;color:#64748b;
                     font-family:'Courier New',Courier,monospace;letter-spacing:0.5px;
                     border-bottom:1px solid #f1f5f9;white-space:nowrap;width:50%;">{label}</td>
          <td style="padding:6px 0 6px 0;font-size:11.5px;font-weight:700;color:{val_color};
                     text-align:right;border-bottom:1px solid #f1f5f9;
                     font-family:'Courier New',Courier,monospace;">{value}</td>
        </tr>"""

    stats_table = f"""
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      {stat_row("IV (30d)",   stats.get("iv",   "—"))}
      {stat_row("IVR",        stats.get("ivr",  "—"))}
      {stat_row("IV / HV",    stats.get("iv_hv","—"), accent=True)}
      {stat_row("BIAS",       stats.get("bias", "—"))}
      {stat_row("DTE",        stats.get("dte",  "—"))}
      {stat_row("CREDIT",     stats.get("credit","—"), accent=True)}
      {stat_row("MGMT DATE",  stats.get("management_date","—"))}
    </table>"""

    score_bar = f"""
    <div style="margin-top:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
        <span style="font-size:9.5px;font-family:'Courier New',monospace;
                     color:#94a3b8;letter-spacing:1.5px;">PREMIUM SCORE</span>
        <span style="font-size:10.5px;font-family:'Courier New',monospace;
                     font-weight:700;color:{bar_color};">{score_text} · {tier}</span>
      </div>
      <div style="height:3px;background:#e2e8f0;border-radius:2px;overflow:hidden;">
        <div style="width:{score_pct}%;height:100%;background:{bar_color};border-radius:2px;"></div>
      </div>
    </div>"""

    strikes_html = f"""
    <div style="margin-bottom:14px;padding:11px 14px;background:#0f172a;border-radius:3px;">
      <div style="font-size:9.5px;font-family:'Courier New',monospace;color:#475569;
                  letter-spacing:1.5px;margin-bottom:5px;">STRUCTURE</div>
      <div style="font-size:12px;font-family:'Courier New',Courier,monospace;
                  color:#e2e8f0;line-height:1.65;word-break:break-word;">
        {stats.get("recommended_strikes", "—")}
      </div>
    </div>"""

    mgmt_html = f"""
    <div style="margin-bottom:14px;padding:12px 14px;background:#f8fafc;
                border-left:3px solid #cbd5e1;">
      <div style="font-size:9.5px;font-family:'Courier New',monospace;color:#94a3b8;
                  letter-spacing:1.5px;margin-bottom:6px;">MANAGEMENT</div>
      <div style="font-size:12.5px;line-height:1.65;color:#334155;">
        {item.get("management_notes", "—")}
      </div>
    </div>"""

    summary_html = f"""
    <div style="padding:12px 14px;background:#f8fafc;
                border-left:3px solid #3b82f6;">
      <div style="font-size:9.5px;font-family:'Courier New',monospace;color:#94a3b8;
                  letter-spacing:1.5px;margin-bottom:6px;">ANALYST NOTE</div>
      <div style="font-size:12.5px;line-height:1.75;color:#1e293b;">
        {item.get("position_summary", "—")}
      </div>
    </div>"""

    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:3px;
                padding:22px 26px;margin-bottom:14px;">

      <!-- Header row -->
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  flex-wrap:wrap;gap:10px;border-bottom:2px solid #0f172a;
                  padding-bottom:12px;margin-bottom:16px;">
        <div style="display:flex;align-items:baseline;gap:10px;">
          <span style="font-size:10px;font-family:'Courier New',monospace;
                       color:#94a3b8;font-weight:700;letter-spacing:1px;">#{rank}</span>
          <span style="font-size:26px;font-weight:900;color:#0f172a;
                       font-family:'Courier New',Courier,monospace;letter-spacing:-0.5px;">{ticker}</span>
          <span style="font-size:14px;font-weight:600;color:#475569;
                       font-family:'Courier New',monospace;">{price_display}</span>
        </div>
        <div style="background:{chip_bg};color:{chip_fg};border:1px solid {chip_bd};
                    border-radius:2px;padding:5px 12px;font-size:11px;font-weight:700;
                    font-family:'Courier New',monospace;letter-spacing:0.5px;
                    white-space:nowrap;align-self:center;">{strategy_name}</div>
      </div>

      <!-- Two-col body -->
      <div style="display:flex;gap:22px;flex-wrap:wrap;">
        <!-- Stats col -->
        <div style="min-width:190px;flex:0 0 190px;">
          {stats_table}
          {score_bar}
        </div>
        <!-- Analysis col -->
        <div style="flex:1;min-width:240px;">
          {strikes_html}
          {mgmt_html}
          {summary_html}
        </div>
      </div>

    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Email wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_html_email(top10_email_json, as_of_date):
    cards_html = "".join(build_card(item, rank=i + 1) for i, item in enumerate(top10_email_json))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Premium Selling Report · {as_of_date}</title>
</head>
<body style="margin:0;padding:0;background:#e2e8f0;
             font-family:system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;">

  <div style="max-width:860px;margin:28px auto 48px auto;">

    <!-- HEADER -->
    <div style="background:#0f172a;padding:32px 40px 26px 40px;">
      <div style="font-size:10px;font-family:'Courier New',monospace;
                  color:#475569;letter-spacing:2.5px;margin-bottom:10px;">
        LOSSDOG RESEARCH &nbsp;&middot;&nbsp; OPTIONS PIPELINE &nbsp;&middot;&nbsp; {as_of_date}
      </div>
      <div style="font-size:24px;font-weight:800;color:#f8fafc;
                  letter-spacing:-0.2px;line-height:1.1;">
        Top 10 Premium Selling Setups
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:6px;
                  font-family:'Courier New',monospace;letter-spacing:0.3px;">
        Liquid Mega-Cap / ETF Universe &nbsp;&middot;&nbsp; 45 DTE Monthly Expiry &nbsp;&middot;&nbsp; Tastytrade Framework
      </div>
      <div style="height:2px;background:linear-gradient(90deg,#3b82f6 0%,#1e40af 55%,transparent 100%);
                  margin-top:20px;"></div>
    </div>

    <!-- LEGEND -->
    <div style="background:#1e293b;padding:9px 40px;display:flex;gap:20px;flex-wrap:wrap;
                align-items:center;">
      <span style="font-size:9.5px;font-family:'Courier New',monospace;
                   color:#64748b;letter-spacing:1px;">SCORE:</span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#10b981;">&#9646; &ge;7.5 EXCELLENT</span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#3b82f6;">&#9646; 6&ndash;7.5 GOOD</span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#f59e0b;">&#9646; 4.5&ndash;6 MODERATE</span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#f97316;">&#9646; 3&ndash;4.5 WEAK</span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#94a3b8;">&#9646; &lt;3 SKIP</span>
    </div>

    <!-- CARDS -->
    <div style="background:#f1f5f9;padding:20px 24px;">
      {cards_html}
    </div>

    <!-- FOOTER -->
    <div style="background:#0f172a;padding:14px 40px;display:flex;
                justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#334155;">
        LOSSDOG RESEARCH &nbsp;&middot;&nbsp; For internal use only &nbsp;&middot;&nbsp; Not investment advice
      </span>
      <span style="font-size:9.5px;font-family:'Courier New',monospace;color:#334155;">
        {as_of_date}
      </span>
    </div>

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

    (REPORT_DIR / "top10_email_payload.json").write_text(
        json.dumps(email_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORT_DIR / "top10_email.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "top10_email_debug.json").write_text(
        json.dumps({
            "source_date_folder":           str(latest),
            "failed_tickers_used_fallback": failed_tickers,
            "generated_at":                 today_str,
            "count":                        len(email_json),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved reports to {REPORT_DIR}")
    print(f"  top10_email.html  ({(REPORT_DIR / 'top10_email.html').stat().st_size:,} bytes)")
    print(f"  top10_email_payload.json")
    if failed_tickers:
        print(f"  WARNING: Fallback used for: {', '.join(failed_tickers)}")


if __name__ == "__main__":
    main()