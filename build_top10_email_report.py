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
You are a structured options research formatter.

You receive one stock's calculated options analysis in semi-clean JSON.
Convert it into a professional, desk-style, email-ready JSON output.

RULES:
- Output valid JSON only
- No markdown
- No explanation outside JSON
- No extra keys
- Use only provided values and facts
- Sound professional, financial, and concise
- management_notes should be natural language and practically useful
- position_summary should be comprehensive and 5 to 8 sentences
- Do not sound educational or generic
- Make the output read like internal research / trader commentary

OUTPUT SCHEMA:
{
  "ticker": str,
  "report_date": str,
  "top_row": {
    "premium_score": str,
    "strategy_name": str,
    "current_price": float
  },
  "basic_stats": {
    "iv": float,
    "ivr": float,
    "bias": str,
    "recommended_strikes": str,
    "dte": int
  },
  "management_notes": str,
  "position_summary": str
}

CONTENT RULES:
- premium_score should look like "5.9/10"
- strategy_name should be explicit; if no trade, say "NONE - SKIP"
- recommended_strikes should be direct and human-readable; if no trade, say "N/A - NO EDGE"
- management_notes should describe profit-taking, time-based management, and key alert/risk behavior naturally
- position_summary should explain:
  1. the volatility / premium setup,
  2. why the strategy was chosen,
  3. the main risk factors,
  4. what the trader should watch,
  5. the overall conviction / quality of the setup

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
        "report_date": data.get("analysis_date"),
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

    management_bits = []
    if strat.get("take_profit_at") is not None:
        management_bits.append(
            f"Take profits at 50% of original credit when the position value reaches {strat.get('take_profit_at')}."
        )
    if strat.get("management_date"):
        management_bits.append(
            f"Do not let the trade drift past 21 DTE management; review or close by {strat.get('management_date')}."
        )
    if clean_input.get("half_one_sd_upside_alert") is not None and clean_input.get("half_one_sd_downside_alert") is not None:
        management_bits.append(
            f"Use {clean_input.get('half_one_sd_upside_alert')} on the upside and {clean_input.get('half_one_sd_downside_alert')} on the downside as early warning levels."
        )
    if strat.get("expected_move_safety_note"):
        management_bits.append(strat.get("expected_move_safety_note"))

    management_notes = " ".join(management_bits).strip()
    if not management_notes:
        management_notes = "Respect profit-taking discipline, manage risk early, and monitor expected-move warning levels."

    position_summary = (
        f"{clean_input.get('ticker')} currently carries a premium attractiveness score of {score_text}, "
        f"with IV at {clean_input.get('iv_30')} and IVR at {clean_input.get('ivr')}. "
        f"The setup leans {clean_input.get('bias')}, and the current strategy recommendation is {strategy_name}. "
        f"This name should be judged in the context of its event risk, implied-versus-realized volatility relationship, and expected-move profile. "
        f"The trade structure is being chosen to align with the directional bias while keeping the risk framework explicit. "
        f"Traders should watch whether price starts moving toward the half-standard-deviation alert levels, because that can be an early sign that the position is becoming less comfortable. "
        f"Overall, this should be treated according to the stated premium score rather than as a blanket high-conviction opportunity."
    )

    return {
        "ticker": clean_input.get("ticker"),
        "report_date": clean_input.get("report_date"),
        "top_row": {
            "premium_score": score_text,
            "strategy_name": strategy_name,
            "current_price": clean_input.get("stock_price"),
        },
        "basic_stats": {
            "iv": clean_input.get("iv_30"),
            "ivr": clean_input.get("ivr"),
            "bias": clean_input.get("bias"),
            "recommended_strikes": strike_details,
            "dte": strat.get("dte"),
        },
        "management_notes": management_notes,
        "position_summary": position_summary,
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
            "temperature": 0.25,
            "maxOutputTokens": 2200
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


def fmt_num(value, decimals=2, prefix=""):
    if isinstance(value, (int, float)):
        return f"{prefix}{value:.{decimals}f}"
    return "N/A"


def score_to_pct(score_text):
    try:
        score = float(str(score_text).split("/")[0])
    except Exception:
        score = 0.0
    pct = max(0, min(score * 10, 100))
    return pct


def score_colors(score_text):
    pct = score_to_pct(score_text)
    if pct >= 70:
        return ("#16a34a", "#22c55e")
    if pct >= 50:
        return ("#ca8a04", "#eab308")
    if pct >= 30:
        return ("#ea580c", "#f59e0b")
    return ("#64748b", "#94a3b8")


def strategy_chip_colors(strategy_name):
    name = (strategy_name or "").upper()
    if "SKIP" in name or "NONE" in name:
        return ("#f8fafc", "#64748b", "#e2e8f0")
    return ("#eff6ff", "#1d4ed8", "#bfdbfe")


# ====================== IMPROVED PROFESSIONAL CARD ======================
def build_card(item):
    top = item["top_row"]
    stats = item["basic_stats"]

    score_text = top["premium_score"]
    score_pct = score_to_pct(score_text)
    bar_start, bar_end = score_colors(score_text)
    chip_bg, chip_fg, chip_bd = strategy_chip_colors(top["strategy_name"])

    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:20px;padding:28px;margin-bottom:24px;box-shadow:0 10px 30px -8px rgb(15 23 42 / 0.08);">
      
      <!-- Header -->
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
        <div style="font-size:32px;font-weight:800;color:#0f172a;letter-spacing:-0.5px;">
          {item['ticker']}
        </div>
        
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="background:#0f172a;color:#ffffff;border-radius:9999px;padding:8px 18px;font-size:14px;font-weight:700;">
            Score {score_text}
          </div>
          <div style="background:{chip_bg};color:{chip_fg};border:1px solid {chip_bd};border-radius:9999px;padding:8px 18px;font-size:14px;font-weight:700;">
            {top['strategy_name']}
          </div>
        </div>
      </div>

      <!-- Stats Grid -->
      <div style="margin-top:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;">
        <div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:14px;padding:14px 16px;">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Current Price</div>
          <div style="margin-top:4px;font-size:19px;font-weight:700;color:#0f172a;">{fmt_num(top['current_price'], 2, '$')}</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:14px;padding:14px 16px;">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Report Date</div>
          <div style="margin-top:4px;font-size:19px;font-weight:700;color:#0f172a;">{item['report_date']}</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:14px;padding:14px 16px;">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">IV</div>
          <div style="margin-top:4px;font-size:19px;font-weight:700;color:#0f172a;">{stats['iv']}</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:14px;padding:14px 16px;">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">IVR</div>
          <div style="margin-top:4px;font-size:19px;font-weight:700;color:#0f172a;">{stats['ivr']}</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #f1f5f9;border-radius:14px;padding:14px 16px;">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Bias</div>
          <div style="margin-top:4px;font-size:19px;font-weight:700;color:#0f172a;">{stats['bias']}</div>
        </div>
      </div>

      <!-- Recommended Strikes + DTE -->
      <div style="margin-top:20px;display:grid;grid-template-columns:3fr 1fr;gap:16px;">
        <div style="background:#0f172a;border-radius:16px;padding:18px 20px;color:#fff;">
          <div style="font-size:11px;color:#93c5fd;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Recommended Strikes</div>
          <div style="margin-top:8px;font-size:19px;line-height:1.4;font-weight:700;">{stats['recommended_strikes']}</div>
        </div>
        <div style="background:#0f172a;border-radius:16px;padding:18px 20px;color:#fff;text-align:center;">
          <div style="font-size:11px;color:#93c5fd;text-transform:uppercase;letter-spacing:1px;font-weight:700;">DTE</div>
          <div style="margin-top:8px;font-size:32px;font-weight:800;">{stats['dte']}</div>
        </div>
      </div>

      <!-- Management Notes -->
      <div style="margin-top:24px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:18px 20px;">
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Management Notes</div>
        <div style="margin-top:10px;font-size:14.5px;line-height:1.65;color:#0f172a;">
          {item['management_notes']}
        </div>
      </div>

      <!-- Position Summary -->
      <div style="margin-top:24px;background:#fafafa;border-left:5px solid #1d4ed8;border-radius:16px;padding:20px 22px;">
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Position Summary</div>
        <div style="margin-top:10px;font-size:14.5px;line-height:1.75;color:#111827;">
          {item['position_summary']}
        </div>
      </div>

      <!-- Premium Score Bar -->
      <div style="margin-top:24px;">
        <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:700;color:#475569;margin-bottom:8px;">
          <span>Premium Score</span>
          <span>{score_text}</span>
        </div>
        <div style="height:12px;background:#e2e8f0;border-radius:9999px;overflow:hidden;">
          <div style="width:{score_pct}%;height:100%;background:linear-gradient(90deg,{bar_start} 0%,{bar_end} 100%);"></div>
        </div>
      </div>

    </div>
    """


# ====================== IMPROVED EMAIL TEMPLATE ======================
def build_html_email(top10_email_json, as_of_date):
    cards_html = "".join(build_card(item) for item in top10_email_json)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Top 10 Premium Selling Setups • Lossdog Research</title>
    </head>
    <body style="margin:0;padding:0;background:#f1f5f9;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
      
      <div style="max-width:1100px;margin:40px auto;background:#ffffff;border-radius:24px;overflow:hidden;box-shadow:0 25px 50px -12px rgb(15 23 42 / 0.15);">
        
        <!-- HEADER -->
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e40af 100%);color:#fff;padding:52px 48px 36px 48px;">
          <div style="font-size:13px;font-weight:600;letter-spacing:2.5px;opacity:0.9;margin-bottom:6px;">
            LOSSDOG RESEARCH • OPTIONS PIPELINE
          </div>
          <h1 style="margin:0;font-size:38px;font-weight:800;line-height:1.05;letter-spacing:-1px;">
            Top 10 Premium Selling Setups
          </h1>
          <div style="margin-top:12px;font-size:17px;opacity:0.85;">
            {as_of_date} &nbsp;&nbsp;|&nbsp;&nbsp; Liquid Mega-Cap / ETF Universe
          </div>
        </div>

        <!-- CARDS -->
        <div style="padding:40px 48px 48px 48px;">
          {cards_html}
        </div>

        <!-- FOOTER -->
        <div style="text-align:center;padding:24px 0 32px 0;color:#64748b;font-size:13px;">
          Generated automatically from your Google Sheet universe, options pipeline, and Gemini research formatter.
        </div>

      </div>
    </body>
    </html>
    """


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