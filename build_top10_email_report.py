import os
import json
import time
from pathlib import Path
from datetime import datetime

import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
MODEL = "gemini-3.1-flash-lite-preview"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

EMAIL_PROMPT = """
You are a structured options email formatter.

You receive one stock's calculated options analysis in semi-clean JSON.
Convert it into a concise, trader-friendly, email-ready JSON.

RULES:
- Output valid JSON only
- No markdown
- No explanation outside JSON
- No extra keys
- Use only provided values
- Keep language concise and trader-friendly
- The result should be rich enough for an email card
- Keep risk flags short
- Keep summary_for_traders informative but concise

OUTPUT SCHEMA:
{
  "ticker": str,
  "highlights": {
    "premium_attractiveness_score": str,
    "iv": float,
    "ivr": float,
    "bias": str,
    "recommended_strategy": str,
    "expiry_date": str,
    "dte": int,
    "strike_details": str,
    "management_date": str
  },
  "risk_flags": [str],
  "management_triggers": {
    "take_profit": str,
    "time_based_management": str,
    "upside_alert_price": float,
    "downside_alert_price": float,
    "alert_note": str
  },
  "summary_for_traders": str
}

CONTENT RULES:
- premium_attractiveness_score should look like "2.6/10"
- include IV and IVR in highlights
- strike_details should be simple and human-readable
- if strategy is skip / none, make that explicit
- use management rules from input
- use upside/downside alert levels from input
- summary_for_traders should explain:
  1. what the setup looks like,
  2. why the strategy was chosen,
  3. what the trader should watch

INPUT JSON:
"""


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
    rec = data.get("recommendation", {}) or {}
    stock_price = data.get("stock_price")
    one_sd_move = safe_get(data, "expected_move_levels", "1sd", "move", default=None)

    upside_alert = None
    downside_alert = None
    if isinstance(stock_price, (int, float)) and isinstance(one_sd_move, (int, float)):
        half_move = one_sd_move / 2.0
        upside_alert = round(stock_price + half_move, 2)
        downside_alert = round(stock_price - half_move, 2)

    return {
        "ticker": data.get("ticker"),
        "analysis_date": data.get("analysis_date"),
        "stock_price": data.get("stock_price"),
        "iv_30": data.get("iv_30"),
        "ivr": data.get("ivr"),
        "ivp": data.get("ivp"),
        "hv_30": data.get("hv_30"),
        "bias": data.get("bias"),
        "attractiveness_score": data.get("attractiveness_score"),
        "attractiveness_label": data.get("attractiveness_label"),
        "earnings_note": data.get("earnings_note"),
        "next_earnings_date": data.get("next_earnings_date"),
        "next_earnings_days": data.get("next_earnings_days"),
        "target_expiry": data.get("target_expiry"),
        "target_dte": data.get("target_dte"),
        "management_date": data.get("management_date"),
        "one_sd_move": safe_get(data, "expected_move_levels", "1sd", "move"),
        "one_sd_move_pct": safe_get(data, "expected_move_levels", "1sd", "move_pct"),
        "one_sd_upside": safe_get(data, "expected_move_levels", "1sd", "upside"),
        "one_sd_downside": safe_get(data, "expected_move_levels", "1sd", "downside"),
        "atm_straddle": safe_get(data, "expected_move", "atm_straddle"),
        "atm_straddle_pct": safe_get(data, "expected_move", "atm_straddle_pct"),
        "half_one_sd_upside_alert": upside_alert,
        "half_one_sd_downside_alert": downside_alert,
        "recommended_strategy": {
            "strategy": rec.get("strategy_name"),
            "reason": rec.get("reason"),
            "risk_constraint": rec.get("risk_constraint"),
            "expiry": rec.get("expiry"),
            "dte": rec.get("dte"),
            "legs": rec.get("legs"),
            "credit": rec.get("credit"),
            "max_risk": rec.get("max_risk"),
            "roi_pct": rec.get("roi_pct"),
            "breakeven": rec.get("breakeven"),
            "breakeven_pct": rec.get("breakeven_pct"),
            "take_profit_at": rec.get("take_profit_at"),
            "management_date": rec.get("management_date"),
            "liquidity_ok": rec.get("liquidity_ok"),
            "meets_one_third": rec.get("meets_one_third"),
            "expected_move_safety_note": safe_get(
                rec, "strategy_data", "management_rules", "safety_vs_em", "note", default=None
            )
        }
    }


def fallback_email_json(clean_input):
    strat = clean_input.get("recommended_strategy", {}) or {}
    score = clean_input.get("attractiveness_score")
    score_text = f"{score}/10" if score is not None else "N/A"

    strategy_name = strat.get("strategy") or "NONE - SKIP"
    strike_details = strat.get("legs") or "N/A - NO EDGE"

    risk_flags = []
    if clean_input.get("earnings_note"):
        risk_flags.append(clean_input["earnings_note"])
    if strat.get("expected_move_safety_note"):
        risk_flags.append(strat["expected_move_safety_note"])
    if strat.get("meets_one_third") is False:
        risk_flags.append("Does not meet the one-third credit rule")
    if not risk_flags:
        risk_flags.append("Fallback summary used because Gemini was temporarily unavailable")

    return {
        "ticker": clean_input.get("ticker"),
        "highlights": {
            "premium_attractiveness_score": score_text,
            "iv": clean_input.get("iv_30"),
            "ivr": clean_input.get("ivr"),
            "bias": clean_input.get("bias"),
            "recommended_strategy": strategy_name,
            "expiry_date": strat.get("expiry"),
            "dte": strat.get("dte"),
            "strike_details": strike_details,
            "management_date": strat.get("management_date") or clean_input.get("management_date"),
        },
        "risk_flags": risk_flags[:4],
        "management_triggers": {
            "take_profit": f"Close position at 50% of original credit ({strat.get('take_profit_at')})",
            "time_based_management": f"Close or roll position at 21 DTE on {strat.get('management_date') or clean_input.get('management_date')}",
            "upside_alert_price": clean_input.get("half_one_sd_upside_alert"),
            "downside_alert_price": clean_input.get("half_one_sd_downside_alert"),
            "alert_note": "These levels represent halfway to the 1 standard deviation expected move and should be used as early warning indicators.",
        },
        "summary_for_traders": (
            f"{clean_input.get('ticker')} has a premium attractiveness score of {score_text}. "
            f"Current IV is {clean_input.get('iv_30')} and IVR is {clean_input.get('ivr')}, with a {clean_input.get('bias')} bias. "
            f"The current recommendation is {strategy_name}. "
            f"Monitor the alert levels and respect the 50% profit and 21 DTE management rules."
        ),
    }


def call_gemini(clean_data, max_retries=5):
    if not API_KEY:
        return {
            "status": "ERROR",
            "ticker": clean_data.get("ticker"),
            "error": "Missing GEMINI_API_KEY"
        }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": EMAIL_PROMPT + json.dumps(clean_data, ensure_ascii=False)
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1800
        }
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{URL}?key={API_KEY}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt < max_retries:
                    sleep_seconds = min(2 ** attempt, 20)
                    print(
                        f"Gemini temporary error for {clean_data.get('ticker')} "
                        f"on attempt {attempt}/{max_retries}: {response.status_code}. "
                        f"Retrying in {sleep_seconds}s..."
                    )
                    time.sleep(sleep_seconds)
                    continue
                return {
                    "status": "ERROR",
                    "ticker": clean_data.get("ticker"),
                    "error": f"Gemini failed after {max_retries} attempts",
                    "details": last_error,
                }

            response.raise_for_status()

            raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = str(e)
            if attempt < max_retries:
                sleep_seconds = min(2 ** attempt, 20)
                print(
                    f"Gemini connection/timeout error for {clean_data.get('ticker')} "
                    f"on attempt {attempt}/{max_retries}. Retrying in {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)
                continue
            return {
                "status": "ERROR",
                "ticker": clean_data.get("ticker"),
                "error": f"Gemini failed after {max_retries} attempts",
                "details": last_error,
            }

        except Exception as e:
            last_error = str(e)
            return {
                "status": "ERROR",
                "ticker": clean_data.get("ticker"),
                "error": "Unexpected Gemini error",
                "details": last_error,
            }


def load_top10():
    latest = get_latest_date_folder()
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


def fmt_num(value, decimals=2):
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return "N/A"


def badge_style(score_text):
    try:
        score = float(str(score_text).split("/")[0])
    except Exception:
        score = 0.0

    if score >= 7:
        return ("#dcfce7", "#166534", "#86efac")
    if score >= 5:
        return ("#fef3c7", "#92400e", "#fcd34d")
    if score >= 3:
        return ("#fee2e2", "#991b1b", "#fca5a5")
    return ("#e5e7eb", "#374151", "#cbd5e1")


def strategy_style(strategy):
    s = (strategy or "").upper()
    if "SKIP" in s or "NONE" in s:
        return ("#f3f4f6", "#6b7280", "#e5e7eb")
    return ("#eff6ff", "#1d4ed8", "#bfdbfe")


def render_stat_pill(label, value):
    return f"""
    <div style="display:inline-block;background:#f8fafc;border:1px solid #e5e7eb;border-radius:999px;
                padding:6px 10px;margin:4px 6px 0 0;font-size:12px;color:#334155;">
      <span style="color:#64748b;">{label}:</span> <b>{value}</b>
    </div>
    """


def build_html_email(top10_email_json, as_of_date):
    cards = []

    for idx, item in enumerate(top10_email_json, 1):
        h = item["highlights"]
        risks = item.get("risk_flags", [])
        mgmt = item["management_triggers"]

        score_bg, score_fg, score_bd = badge_style(h["premium_attractiveness_score"])
        strat_bg, strat_fg, strat_bd = strategy_style(h["recommended_strategy"])

        risk_html = "".join(
            f'<li style="margin:0 0 6px 0;">{r}</li>' for r in risks[:4]
        ) or '<li style="margin:0;">No major flags provided.</li>'

        stat_row = "".join([
            render_stat_pill("Score", h["premium_attractiveness_score"]),
            render_stat_pill("IV", h["iv"]),
            render_stat_pill("IVR", h["ivr"]),
            render_stat_pill("Bias", h["bias"]),
            render_stat_pill("Expiry", h["expiry_date"]),
            render_stat_pill("DTE", h["dte"]),
        ])

        card = f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;
                    padding:20px;margin-bottom:18px;box-shadow:0 6px 18px rgba(15,23,42,0.05);">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
            <div>
              <div style="font-size:26px;font-weight:800;color:#0f172a;letter-spacing:0.2px;">#{idx} {item['ticker']}</div>
              <div style="margin-top:8px;">{stat_row}</div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
              <div style="background:{score_bg};color:{score_fg};border:1px solid {score_bd};
                          border-radius:999px;padding:8px 12px;font-size:13px;font-weight:700;">
                Premium Score {h['premium_attractiveness_score']}
              </div>
              <div style="background:{strat_bg};color:{strat_fg};border:1px solid {strat_bd};
                          border-radius:999px;padding:8px 12px;font-size:13px;font-weight:700;">
                {h['recommended_strategy']}
              </div>
            </div>
          </div>

          <div style="margin-top:16px;background:#0f172a;border-radius:14px;padding:16px;">
            <div style="font-size:12px;font-weight:700;letter-spacing:0.7px;color:#93c5fd;text-transform:uppercase;">
              Recommended Trade
            </div>
            <div style="margin-top:8px;font-size:18px;font-weight:700;color:#ffffff;line-height:1.5;">
              {h['strike_details']}
            </div>
            <div style="margin-top:10px;font-size:14px;color:#cbd5e1;line-height:1.8;">
              <b>Expiry:</b> {h['expiry_date']} ({h['dte']} DTE)&nbsp;&nbsp;|&nbsp;&nbsp;
              <b>Management Date:</b> {h['management_date']}
            </div>
          </div>

          <div style="margin-top:16px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;">
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:14px;">
              <div style="font-size:12px;font-weight:700;letter-spacing:0.7px;color:#475569;text-transform:uppercase;">
                Risk Flags
              </div>
              <ul style="margin:10px 0 0 18px;padding:0;color:#334155;font-size:14px;line-height:1.55;">
                {risk_html}
              </ul>
            </div>

            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:14px;">
              <div style="font-size:12px;font-weight:700;letter-spacing:0.7px;color:#475569;text-transform:uppercase;">
                Management
              </div>
              <div style="margin-top:10px;font-size:14px;color:#334155;line-height:1.75;">
                <b>Take profit:</b> {mgmt['take_profit']}<br>
                <b>Manage by:</b> {mgmt['time_based_management']}<br>
                <b>Upside alert:</b> <span style="font-weight:700;color:#111827;">{fmt_num(mgmt['upside_alert_price'])}</span><br>
                <b>Downside alert:</b> <span style="font-weight:700;color:#111827;">{fmt_num(mgmt['downside_alert_price'])}</span><br>
                <span style="color:#64748b;">{mgmt['alert_note']}</span>
              </div>
            </div>
          </div>

          <div style="margin-top:16px;background:#fafafa;border-left:4px solid #3b82f6;border-radius:12px;
                      padding:14px 16px;color:#111827;font-size:14px;line-height:1.75;">
            <div style="font-size:12px;font-weight:700;letter-spacing:0.7px;color:#475569;text-transform:uppercase;margin-bottom:6px;">
              Trader Takeaway
            </div>
            {item['summary_for_traders']}
          </div>
        </div>
        """
        cards.append(card)

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#eef2f7;font-family:Inter,Arial,Helvetica,sans-serif;">
        <div style="max-width:980px;margin:0 auto;padding:24px;">
          <div style="background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);
                      color:#ffffff;border-radius:20px;padding:28px 28px 24px 28px;margin-bottom:24px;
                      box-shadow:0 10px 30px rgba(15,23,42,0.18);">
            <div style="font-size:13px;letter-spacing:1px;text-transform:uppercase;color:#bfdbfe;font-weight:700;">
              Daily Options Briefing
            </div>
            <div style="font-size:30px;font-weight:800;line-height:1.2;margin-top:8px;">
              Top 10 Premium Selling Setups
            </div>
            <div style="font-size:15px;line-height:1.7;color:#dbeafe;margin-top:10px;max-width:760px;">
              {as_of_date} • ranked from your liquid mega-cap / ETF universe • sorted by premium attractiveness score
            </div>
          </div>
          {''.join(cards)}
          <div style="text-align:center;color:#64748b;font-size:12px;padding:8px 0 18px 0;">
            Generated automatically from your Google Sheet universe, options pipeline, and Gemini trade summarizer.
          </div>
        </div>
      </body>
    </html>
    """
    return html


def main():
    latest, top10 = load_top10()
    if not top10:
        raise RuntimeError("No ranked calculated.json files found")

    email_json = []
    failed_tickers = []

    for data in top10:
        clean_input = build_email_llm_input(data)
        out = call_gemini(clean_input)

        if isinstance(out, dict) and out.get("status") == "ERROR":
            print(f"Gemini failed for {clean_input.get('ticker')}, using fallback summary")
            failed_tickers.append(clean_input.get("ticker"))
            out = fallback_email_json(clean_input)

        email_json.append(out)

    if not email_json:
        raise RuntimeError("No email content available")

    today_str = datetime.now().strftime("%Y-%m-%d")
    html = build_html_email(email_json, today_str)

    with open(REPORT_DIR / "top10_email_payload.json", "w", encoding="utf-8") as f:
        json.dump(email_json, f, indent=2, ensure_ascii=False)

    with open(REPORT_DIR / "top10_email.html", "w", encoding="utf-8") as f:
        f.write(html)

    with open(REPORT_DIR / "top10_email_debug.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_date_folder": str(latest),
                "failed_tickers_used_fallback": failed_tickers,
                "generated_at": today_str,
                "count": len(email_json),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {REPORT_DIR / 'top10_email_payload.json'}")
    print(f"Saved: {REPORT_DIR / 'top10_email.html'}")
    print(f"Saved: {REPORT_DIR / 'top10_email_debug.json'}")
    print(f"Source date folder: {latest}")


if __name__ == "__main__":
    main()