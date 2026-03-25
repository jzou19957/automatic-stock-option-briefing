import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

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
Your job is to convert it into a SIMPLE but information-dense email-friendly JSON output for traders.

RULES:
- Output valid JSON only
- No markdown
- No explanation outside JSON
- No extra keys beyond schema
- Do not omit keys from schema
- Use only provided numbers and facts
- Do not invent values
- Keep language concise, clear, and email-friendly
- summary_for_traders must be no more than 2 short paragraphs total
- Each paragraph should be compact and easy to scan
- highlights must contain the most important facts a trader needs immediately

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
- highlights.premium_attractiveness_score should look like "2.6/10"
- highlights must always include IV, IVR, strategy, expiry date, strike details, and management date
- strike_details should be simple and human-readable, for example:
  "SELL 400 CALL / BUY 405 CALL"
- risk_flags should contain 2 to 5 short items covering the biggest caution or context points
- management_triggers.take_profit must clearly describe the 50% credit rule
- management_triggers.time_based_management must clearly describe the 21 DTE rule
- management_triggers.upside_alert_price and downside_alert_price are already precomputed in input
- management_triggers.alert_note should explain these are halfway-to-1 standard deviation monitoring levels
- summary_for_traders should explain:
  1. what the volatility/setup looks like,
  2. why the strategy is being recommended,
  3. what the trader should watch or do
- Keep summary useful for email readers, not generic

INPUT JSON:
"""


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_latest_date_folder() -> Path:
    date_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], reverse=True)
    if not date_dirs:
        raise FileNotFoundError("No dated data folders found in data/")
    return date_dirs[0]


def build_email_llm_input(data: dict[str, Any]) -> dict[str, Any]:
    rec = data.get("recommendation", {}) or {}
    attr = data.get("attractiveness_breakdown", {}) or {}

    stock_price = data.get("stock_price")
    one_sd_move = safe_get(data, "expected_move_levels", "1sd", "move", default=None)

    upside_alert_price = None
    downside_alert_price = None
    if isinstance(stock_price, (int, float)) and isinstance(one_sd_move, (int, float)):
        half_move = one_sd_move / 2.0
        upside_alert_price = round(stock_price + half_move, 2)
        downside_alert_price = round(stock_price - half_move, 2)

    return {
        "ticker": data.get("ticker"),
        "highlights": {
            "premium_attractiveness_score": data.get("attractiveness_score"),
            "premium_attractiveness_label": data.get("attractiveness_label"),
            "iv": data.get("iv_30"),
            "ivr": data.get("ivr"),
            "ivp": data.get("ivp"),
            "hv": data.get("hv_30"),
            "bias": data.get("bias"),
            "recommended_strategy": rec.get("strategy_name"),
            "expiry_date": rec.get("expiry"),
            "dte": rec.get("dte"),
            "strike_details": rec.get("legs"),
            "management_date": rec.get("management_date"),
            "credit": rec.get("credit"),
            "max_risk": rec.get("max_risk"),
            "breakeven": rec.get("breakeven"),
            "roi_pct": rec.get("roi_pct"),
        },
        "context": {
            "analysis_date": data.get("analysis_date"),
            "stock_price": data.get("stock_price"),
            "target_expiry": data.get("target_expiry"),
            "target_dte": data.get("target_dte"),
            "management_date": data.get("management_date"),
            "edge_summary": data.get("edge_summary"),
            "earnings_note": data.get("earnings_note"),
            "next_earnings_date": data.get("next_earnings_date"),
            "next_earnings_days": data.get("next_earnings_days"),
            "earnings_safe": data.get("earnings_safe"),
        },
        "expected_move": {
            "one_sd_move": safe_get(data, "expected_move_levels", "1sd", "move"),
            "one_sd_move_pct": safe_get(data, "expected_move_levels", "1sd", "move_pct"),
            "one_sd_upside": safe_get(data, "expected_move_levels", "1sd", "upside"),
            "one_sd_downside": safe_get(data, "expected_move_levels", "1sd", "downside"),
            "two_sd_move": safe_get(data, "expected_move_levels", "2sd", "move"),
            "two_sd_move_pct": safe_get(data, "expected_move_levels", "2sd", "move_pct"),
            "atm_straddle": safe_get(data, "expected_move", "atm_straddle"),
            "atm_straddle_pct": safe_get(data, "expected_move", "atm_straddle_pct"),
            "half_one_sd_upside_alert": upside_alert_price,
            "half_one_sd_downside_alert": downside_alert_price,
        },
        "premium_selling": {
            "attractiveness_score": data.get("attractiveness_score"),
            "attractiveness_label": data.get("attractiveness_label"),
            "raw_score": attr.get("raw_score"),
            "earnings_modifier": attr.get("earnings_modifier"),
            "modifier_note": attr.get("modifier_note"),
        },
        "recommended_strategy": {
            "strategy": rec.get("strategy_name"),
            "reason": rec.get("reason"),
            "risk_constraint": rec.get("risk_constraint"),
            "legs": rec.get("legs"),
            "expiry": rec.get("expiry"),
            "dte": rec.get("dte"),
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
            ),
        },
        "management_framework": {
            "take_profit_rule": f"Take profit at 50% of original credit (target value: {rec.get('take_profit_at')})",
            "time_rule": f"Manage or close at 21 DTE on {rec.get('management_date')}",
            "upside_alert_price": upside_alert_price,
            "downside_alert_price": downside_alert_price,
            "alert_note": "These prices are halfway to the 1 standard deviation expected move and should be used as early warning levels.",
        },
    }


def call_gemini(clean_data: dict[str, Any]) -> dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")

    payload = {
        "contents": [{"parts": [{"text": EMAIL_PROMPT + json.dumps(clean_data, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1800},
    }

    response = requests.post(
        f"{URL}?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()

    raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def load_top10() -> tuple[Path, list[dict[str, Any]]]:
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


def build_html_email(top10_email_json: list[dict[str, Any]], as_of_date: str) -> str:
    cards = []
    for item in top10_email_json:
        h = item["highlights"]
        risks = "".join(f"<li>{r}</li>" for r in item.get("risk_flags", []))
        mgmt = item["management_triggers"]
        cards.append(f"""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:18px;margin-bottom:18px;">
          <div style="font-size:22px;font-weight:700;color:#111827;margin-bottom:8px;">{item['ticker']}</div>
          <div style="font-size:14px;color:#374151;line-height:1.7;">
            <b>Premium attractiveness:</b> {h['premium_attractiveness_score']}<br>
            <b>IV / IVR:</b> {h['iv']} / {h['ivr']}<br>
            <b>Bias:</b> {h['bias']}<br>
            <b>Recommended strategy:</b> {h['recommended_strategy']}<br>
            <b>Expiry:</b> {h['expiry_date']} ({h['dte']} DTE)<br>
            <b>Strikes:</b> {h['strike_details']}<br>
            <b>Management date:</b> {h['management_date']}
          </div>
          <div style="margin-top:12px;">
            <div style="font-weight:700;color:#111827;margin-bottom:6px;">Risk flags</div>
            <ul style="margin:0;padding-left:20px;color:#374151;">{risks}</ul>
          </div>
          <div style="margin-top:12px;font-size:14px;color:#374151;line-height:1.7;">
            <b>Take profit:</b> {mgmt['take_profit']}<br>
            <b>Time-based management:</b> {mgmt['time_based_management']}<br>
            <b>Upside alert:</b> {mgmt['upside_alert_price']}<br>
            <b>Downside alert:</b> {mgmt['downside_alert_price']}<br>
            <b>Alert note:</b> {mgmt['alert_note']}
          </div>
          <div style="margin-top:14px;background:#f9fafb;border-radius:10px;padding:12px;color:#111827;line-height:1.7;">
            {item['summary_for_traders']}
          </div>
        </div>
        """)

    return f"""
    <html>
      <body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:920px;margin:0 auto;padding:24px;">
          <div style="background:#111827;color:#ffffff;border-radius:16px;padding:24px;margin-bottom:24px;">
            <div style="font-size:28px;font-weight:800;">Top 10 Best Premium Selling Setups</div>
            <div style="font-size:16px;margin-top:8px;color:#d1d5db;">
              {as_of_date} • ranked from your liquid mega-cap / ETF universe
            </div>
          </div>
          {''.join(cards)}
        </div>
      </body>
    </html>
    """


def main() -> None:
    latest, top10 = load_top10()
    if not top10:
        raise RuntimeError("No ranked calculated.json files found")

    email_json = []
    for data in top10:
        clean_input = build_email_llm_input(data)
        out = call_gemini(clean_input)
        email_json.append(out)

    today_str = datetime.now().strftime("%Y-%m-%d")
    html = build_html_email(email_json, today_str)

    with open(REPORT_DIR / "top10_email_payload.json", "w", encoding="utf-8") as f:
        json.dump(email_json, f, indent=2, ensure_ascii=False)
    with open(REPORT_DIR / "top10_email.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved: {REPORT_DIR / 'top10_email_payload.json'}")
    print(f"Saved: {REPORT_DIR / 'top10_email.html'}")
    print(f"Source date folder: {latest}")


if __name__ == "__main__":
    main()
