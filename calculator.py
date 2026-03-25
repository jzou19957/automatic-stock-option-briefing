"""
calculator.py — Pure math. Zero opinions. Zero LLM.

Takes the 3 parsed JSON files for a ticker and calculates EVERYTHING
needed for the LLM to make a final recommendation:

  - IV/HV ratio, edge quality
  - Expected move (1σ and 2σ) at 45 DTE
  - Black-Scholes delta for every strike
  - Tasty score for every strike (delta + liquidity + premium)
  - Strategy candidates: put spread, call spread, iron condor, jade lizard, strangle
  - For each candidate: credit, max risk, ROI, breakeven, 1/3 rule check
  - Management date (21 DTE from expiry = take profit target date)
  - Earnings safety check
  - Final scored summary ready for LLM

Usage:
    python calculator.py data/2026-03-25/MSFT
    python calculator.py                        (all tickers today)

Output:
    data/2026-03-25/MSFT/calculated.json
"""

import sys
import json
import math
import glob
from datetime import date, datetime, timedelta
from pathlib import Path
from expiry_utils import find_best_monthly_expiry

try:
    from scipy.stats import norm
    def _ncdf(x): return norm.cdf(x)
except ImportError:
    def _ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

DATA_DIR = Path("./data")
TODAY    = str(date.today())

# ── Thresholds ────────────────────────────────────────────────────────────────

IV_HV_MIN       = 1.10   # minimum edge to consider selling
IV_HV_GOOD      = 1.30
IV_HV_STRONG    = 1.50
IV_HV_GREAT     = 2.00

IVR_LOW         = 30
IVR_MODERATE    = 50
IVR_HIGH        = 65

MIN_VOLUME      = 50
MIN_OI          = 100
MIN_CREDIT      = 0.10   # lowered — better to flag low credit than skip entirely
ONE_THIRD_RULE  = True   # credit must be >= spread_width / 3

DELTA_TARGET    = 0.25   # ideal short delta
DELTA_MIN       = 0.10
DELTA_MAX       = 0.40


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_delta(S, K, T, sigma, opt_type="put"):
    """Black-Scholes delta. sigma = decimal (0.28 not 28%)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        if opt_type == "call":
            return round(_ncdf(d1), 4)
        else:
            return round(_ncdf(d1) - 1, 4)
    except:
        return None


def bs_prob_otm(S, K, T, sigma, opt_type="put"):
    """Probability of expiring OTM."""
    d = bs_delta(S, K, T, sigma, opt_type)
    if d is None:
        return None
    return round(1 - abs(d), 4)


def expected_move(S, iv_decimal, T_days):
    """
    Expected move using Black-Scholes lognormal model.
    Returns (1sigma_move, 2sigma_move, 1sigma_pct, 2sigma_pct)
    """
    T = T_days / 365
    move_1sd = round(S * iv_decimal * math.sqrt(T), 2)
    move_2sd = round(S * iv_decimal * math.sqrt(T) * 2, 2)
    pct_1sd  = round(move_1sd / S * 100, 2)
    pct_2sd  = round(move_2sd / S * 100, 2)
    return move_1sd, move_2sd, pct_1sd, pct_2sd


def atm_straddle_move(chain, stock_price):
    """
    Expected move from ATM straddle price.
    The straddle price (call mid + put mid at ATM) = market's expected move.
    """
    atm = min(chain, key=lambda r: abs(r["strike"] - stock_price))
    straddle = round((atm["call"]["mid"] or 0) + (atm["put"]["mid"] or 0), 2)
    pct = round(straddle / stock_price * 100, 2) if stock_price else None
    return straddle, pct, atm["strike"]


# ── Strike enrichment ─────────────────────────────────────────────────────────

def enrich_chain(chain, S, iv_decimal, T):
    """
    Add delta, prob_otm, bid_ask_spread_pct, tasty_score to every strike.
    Returns enriched chain + summary stats.
    """
    max_vol = max((max(r["call"]["volume"] or 0, r["put"]["volume"] or 0)
                   for r in chain), default=1) or 1
    max_oi  = max((max(r["call"]["oi"] or 0,     r["put"]["oi"] or 0)
                   for r in chain), default=1) or 1

    enriched = []
    put_candidates  = []
    call_candidates = []

    for row in chain:
        K   = row["strike"]
        mon = row["moneyness"]
        new = {"strike": K, "moneyness": mon, "itm": row.get("itm", False)}

        for side in ["call", "put"]:
            opt  = row[side]
            bid  = opt["bid"] or 0
            ask  = opt["ask"] or 0
            mid  = round((bid + ask) / 2, 2)
            vol  = opt["volume"] or 0
            oi   = opt["oi"] or 0

            # Fix zero bid
            if bid == 0 and ask > 0:
                bid = round(ask * 0.85, 2)
                mid = round((bid + ask) / 2, 2)

            spread_pct = round((ask - bid) / mid * 100, 1) if mid > 0 else 999
            vol_rank   = round(vol / max_vol, 3)
            oi_rank    = round(oi  / max_oi,  3)
            prem_pct   = round(mid / S * 100, 3) if S else 0

            opt_type   = "call" if side == "call" else "put"
            delta      = bs_delta(S, K, T, iv_decimal, opt_type)
            prob_otm   = bs_prob_otm(S, K, T, iv_decimal, opt_type)

            # ── Tasty score (0–100) ───────────────────────────────────────────
            # Delta score: peak at 0.25 delta
            delta_score = 0
            if delta is not None:
                abs_d = abs(delta)
                if 0.20 <= abs_d <= 0.30:   delta_score = 100
                elif 0.15 <= abs_d < 0.20:  delta_score = 70
                elif 0.30 < abs_d <= 0.35:  delta_score = 70
                elif 0.10 <= abs_d < 0.15:  delta_score = 35
                elif 0.35 < abs_d <= 0.45:  delta_score = 35
                else:                        delta_score = 5

            # Liquidity score
            spread_score  = max(0, 100 - spread_pct * 4)
            volume_score  = min(100, vol_rank * 300)
            oi_score      = min(100, oi_rank  * 200)
            liq_score     = round(spread_score*0.4 + volume_score*0.35 + oi_score*0.25, 1)

            # Premium score (meaningful credit)
            prem_score    = min(100, prem_pct * 400)

            tasty_score   = round(
                delta_score * 0.40 +
                liq_score   * 0.35 +
                prem_score  * 0.25,
                1
            )

            new[side] = {
                "bid":          bid,
                "ask":          ask,
                "mid":          mid,
                "volume":       vol,
                "oi":           oi,
                "premium_pct":  prem_pct,
                "spread_pct":   spread_pct,
                "vol_rank":     vol_rank,
                "delta":        delta,
                "prob_otm":     prob_otm,
                "liquidity_score": liq_score,
                "tasty_score":  tasty_score,
                "liquid":       vol >= MIN_VOLUME or oi >= MIN_OI,
            }

            if delta is not None:
                entry = (tasty_score, K, delta, new[side])
                if side == "put":
                    put_candidates.append(entry)
                else:
                    call_candidates.append(entry)

        enriched.append(new)

    # Sort by tasty score descending
    put_candidates.sort(key=lambda x: x[0], reverse=True)
    call_candidates.sort(key=lambda x: x[0], reverse=True)

    stats = {
        "best_put":  {"strike": put_candidates[0][1],
                      "delta":  put_candidates[0][2],
                      "score":  put_candidates[0][0]} if put_candidates else None,
        "best_call": {"strike": call_candidates[0][1],
                      "delta":  call_candidates[0][2],
                      "score":  call_candidates[0][0]} if call_candidates else None,
        "put_candidates":  [(s, k, round(d,3)) for s,k,d,_ in put_candidates[:5]],
        "call_candidates": [(s, k, round(d,3)) for s,k,d,_ in call_candidates[:5]],
    }

    return enriched, stats


# ── Strike lookup ─────────────────────────────────────────────────────────────

def find_by_tasty(chain, side, min_delta=0.10, max_delta=0.40, require_liquid=True):
    """Find best strike for a side filtered by delta range and liquidity."""
    candidates = []
    for row in chain:
        opt   = row[side]
        delta = opt.get("delta")
        if delta is None:
            continue
        abs_d = abs(delta)
        if not (min_delta <= abs_d <= max_delta):
            continue
        if require_liquid and not opt.get("liquid"):
            continue
        candidates.append((opt["tasty_score"], row))

    if not candidates:
        # Relax liquidity
        candidates = [
            (row[side]["tasty_score"], row)
            for row in chain
            if row[side].get("delta") is not None
            and min_delta <= abs(row[side]["delta"]) <= max_delta
        ]

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_spread_long(chain, side, short_strike, width=5):
    """Find long leg of spread: short_strike ± width."""
    if side == "put":
        target = short_strike - width
        candidates = sorted(
            [r for r in chain if r["strike"] <= short_strike - 2],
            key=lambda r: abs(r["strike"] - target)
        )
    else:
        target = short_strike + width
        candidates = sorted(
            [r for r in chain if r["strike"] >= short_strike + 2],
            key=lambda r: abs(r["strike"] - target)
        )
    return candidates[0] if candidates else None


# ── Strategy builders ─────────────────────────────────────────────────────────

def build_management_rules(strategy_type, S, short_strike, credit,
                            em_levels, expiry_str, mgmt_date_str, ticker=""):
    """
    Build specific management rules with price targets for a strategy.
    All rules are pure math — no opinions.
    """
    tp_credit   = round(credit * 0.50, 2)
    em1_up      = em_levels["1sd"]["upside"]
    em1_down    = em_levels["1sd"]["downside"]
    em2_up      = em_levels["2sd"]["upside"]
    em2_down    = em_levels["2sd"]["downside"]

    if strategy_type == "put_spread":
        be = round(short_strike - credit, 2)
        return {
            "strategy_type":    "put_spread",
            "breakeven_price":  be,
            "breakeven_note":   f"{ticker} must stay ABOVE ${be} to be profitable",
            "take_profit":      {
                "trigger":      f"Close spread when spread value = ${tp_credit}",
                "credit_target": tp_credit,
                "note":         f"50% of ${credit} original credit"
            },
            "manage_at_21dte":  {
                "date":         mgmt_date_str,
                "action":       "Close or roll regardless of price"
            },
            "safety_vs_em": {
                "1sd_downside":  em1_down,
                "safe_68pct":    em1_down > be,
                "2sd_downside":  em2_down,
                "safe_95pct":    em2_down > be,
                "note": (
                    f"1σ downside ${em1_down} is {'ABOVE' if em1_down > be else 'BELOW'} "
                    f"breakeven ${be} — "
                    f"{'68% of moves stay safe ✅' if em1_down > be else '68% move tests this spread ⚠️'}"
                )
            }
        }

    elif strategy_type == "call_spread":
        be = round(short_strike + credit, 2)
        return {
            "strategy_type":    "call_spread",
            "breakeven_price":  be,
            "breakeven_note":   f"{ticker} must stay BELOW ${be} to be profitable",
            "take_profit":      {
                "trigger":      f"Close spread when spread value = ${tp_credit}",
                "credit_target": tp_credit,
                "note":         f"50% of ${credit} original credit"
            },
            "manage_at_21dte":  {
                "date":         mgmt_date_str,
                "action":       "Close or roll regardless of price"
            },
            "safety_vs_em": {
                "1sd_upside":    em1_up,
                "safe_68pct":    em1_up < be,
                "2sd_upside":    em2_up,
                "safe_95pct":    em2_up < be,
                "note": (
                    f"1σ upside ${em1_up} is {'BELOW' if em1_up < be else 'ABOVE'} "
                    f"breakeven ${be} — "
                    f"{'68% of moves stay safe ✅' if em1_up < be else '68% move tests this spread ⚠️'}"
                )
            }
        }

    elif strategy_type == "iron_condor":
        put_be  = round(short_strike - credit, 2)
        # For condor, short_strike is put short, credit is total
        return {
            "strategy_type":       "iron_condor",
            "profit_zone":         f"{ticker} between ${put_be} and ${round(put_be + credit + 10, 2)} at expiry",
            "take_profit":         {
                "trigger":         f"Close entire condor when total value = ${tp_credit}",
                "credit_target":   tp_credit,
            },
            "manage_at_21dte":     {
                "date":            mgmt_date_str,
                "action":          "Close or roll untested side"
            },
        }

    return {}


def build_spread(chain, side, S, dte, expiry_str, mgmt_date_str, spread_width=5):
    """Build put spread or call spread."""
    short_row = find_by_tasty(chain, side)
    if not short_row:
        return None

    long_row  = find_spread_long(chain, side, short_row["strike"], spread_width)
    if not long_row:
        return None

    short_opt = short_row[side]
    long_opt  = long_row[side]

    credit    = round(short_opt["mid"] - long_opt["mid"], 2)
    if credit <= 0:
        # try wider spread
        long_row = find_spread_long(chain, side, short_row["strike"], spread_width * 2)
        if not long_row:
            return None
        long_opt = long_row[side]
        credit   = round(short_opt["mid"] - long_opt["mid"], 2)
    if credit < MIN_CREDIT:
        return None

    width     = abs(short_row["strike"] - long_row["strike"])
    max_risk  = round(width - credit, 2)
    roi       = round(credit / max_risk * 100, 1) if max_risk > 0 else 0
    one_third = round(width / 3, 2)

    if side == "put":
        breakeven     = round(short_row["strike"] - credit, 2)
        be_pct        = round((breakeven - S) / S * 100, 2)
        strategy_name = "Short Put Spread"
    else:
        breakeven     = round(short_row["strike"] + credit, 2)
        be_pct        = round((breakeven - S) / S * 100, 2)
        strategy_name = "Short Call Spread"

    # Flag if credit is below ideal but above absolute minimum
    credit_note = None
    IDEAL_CREDIT = 0.50
    if credit < IDEAL_CREDIT:
        credit_note = f"Low credit ${credit} — likely short-dated chain or low IV environment"

    return {
        "strategy":       strategy_name,
        "expiry":         expiry_str,
        "dte":            dte,
        "credit_note":    credit_note,
        "management_date": mgmt_date_str,
        "legs": [
            {"action": "SELL", "type": side.upper(),
             "strike": short_row["strike"],
             "bid": short_opt["bid"], "ask": short_opt["ask"], "mid": short_opt["mid"],
             "delta": short_opt["delta"], "prob_otm": short_opt["prob_otm"],
             "volume": short_opt["volume"], "oi": short_opt["oi"]},
            {"action": "BUY",  "type": side.upper(),
             "strike": long_row["strike"],
             "bid": long_opt["bid"], "ask": long_opt["ask"], "mid": long_opt["mid"],
             "delta": long_opt["delta"], "prob_otm": long_opt["prob_otm"],
             "volume": long_opt["volume"], "oi": long_opt["oi"]},
        ],
        "credit":           credit,
        "spread_width":     width,
        "max_risk":         max_risk,
        "roi_pct":          roi,
        "breakeven":        breakeven,
        "breakeven_pct":    be_pct,
        "take_profit_at":   round(credit * 0.50, 2),
        "one_third_min":    one_third,
        "meets_one_third":  credit >= one_third,
        "short_delta":      short_opt["delta"],
        "short_prob_otm":   short_opt["prob_otm"],
        "liquidity_ok":     short_opt.get("liquid", False),
    }


def build_iron_condor(chain, S, dte, expiry_str, mgmt_date_str):
    """Build iron condor: OTM put spread + OTM call spread."""
    put_spread  = build_spread(chain, "put",  S, dte, expiry_str, mgmt_date_str)
    call_spread = build_spread(chain, "call", S, dte, expiry_str, mgmt_date_str)

    if not put_spread or not call_spread:
        return None

    total_credit = round(put_spread["credit"] + call_spread["credit"], 2)
    max_risk     = round(max(put_spread["spread_width"],
                              call_spread["spread_width"]) - total_credit, 2)
    roi          = round(total_credit / max_risk * 100, 1) if max_risk > 0 else 0

    return {
        "strategy":        "Iron Condor",
        "expiry":          expiry_str,
        "dte":             dte,
        "management_date": mgmt_date_str,
        "put_spread":      put_spread,
        "call_spread":     call_spread,
        "legs":            put_spread["legs"] + call_spread["legs"],
        "credit":          total_credit,
        "max_risk":        max_risk,
        "roi_pct":         roi,
        "put_breakeven":   put_spread["breakeven"],
        "put_breakeven_pct": put_spread["breakeven_pct"],
        "call_breakeven":  call_spread["breakeven"],
        "call_breakeven_pct": call_spread["breakeven_pct"],
        "take_profit_at":  round(total_credit * 0.50, 2),
        "liquidity_ok":    put_spread["liquidity_ok"] and call_spread["liquidity_ok"],
    }


def build_jade_lizard(chain, S, dte, expiry_str, mgmt_date_str):
    """Jade Lizard: naked put + call spread. Credit > call spread width = no upside risk."""
    put_row  = find_by_tasty(chain, "put")
    if not put_row:
        return None

    call_short = find_by_tasty(chain, "call")
    if not call_short:
        return None

    call_long = find_spread_long(chain, "call", call_short["strike"], 5)
    if not call_long:
        return None

    put_credit  = put_row["put"]["mid"]
    call_credit = round(call_short["call"]["mid"] - call_long["call"]["mid"], 2)
    call_width  = abs(call_long["strike"] - call_short["strike"])
    total       = round(put_credit + call_credit, 2)

    if total < MIN_CREDIT:
        return None

    no_upside_risk = total > call_width
    breakeven      = round(put_row["strike"] - total, 2)
    be_pct         = round((breakeven - S) / S * 100, 2)

    return {
        "strategy":          "Jade Lizard",
        "expiry":            expiry_str,
        "dte":               dte,
        "management_date":   mgmt_date_str,
        "legs": [
            {"action": "SELL", "type": "PUT",
             "strike": put_row["strike"],
             "mid": put_row["put"]["mid"],
             "delta": put_row["put"]["delta"],
             "prob_otm": put_row["put"]["prob_otm"],
             "volume": put_row["put"]["volume"], "oi": put_row["put"]["oi"]},
            {"action": "SELL", "type": "CALL",
             "strike": call_short["strike"],
             "mid": call_short["call"]["mid"],
             "delta": call_short["call"]["delta"],
             "prob_otm": call_short["call"]["prob_otm"],
             "volume": call_short["call"]["volume"], "oi": call_short["call"]["oi"]},
            {"action": "BUY",  "type": "CALL",
             "strike": call_long["strike"],
             "mid": call_long["call"]["mid"],
             "delta": call_long["call"]["delta"],
             "prob_otm": call_long["call"]["prob_otm"],
             "volume": call_long["call"]["volume"], "oi": call_long["call"]["oi"]},
        ],
        "credit":            total,
        "put_credit":        put_credit,
        "call_spread_credit": call_credit,
        "call_spread_width": call_width,
        "no_upside_risk":    no_upside_risk,
        "max_risk":          f"${breakeven} downside (put side)",
        "roi_pct":           round(total / put_row["strike"] * 100, 2),
        "breakeven":         breakeven,
        "breakeven_pct":     be_pct,
        "take_profit_at":    round(total * 0.50, 2),
        "upside_risk_note":  "No upside risk ✅" if no_upside_risk else "⚠️ Credit < call spread width — upside risk exists",
        "liquidity_ok":      put_row["put"].get("liquid", False),
    }


def build_strangle(chain, S, dte, expiry_str, mgmt_date_str):
    """Short strangle: naked put + naked call (undefined risk)."""
    put_row  = find_by_tasty(chain, "put")
    call_row = find_by_tasty(chain, "call")
    if not put_row or not call_row:
        return None

    credit = round(put_row["put"]["mid"] + call_row["call"]["mid"], 2)
    if credit < MIN_CREDIT:
        return None

    put_be  = round(put_row["strike"]  - credit, 2)
    call_be = round(call_row["strike"] + credit, 2)

    return {
        "strategy":          "Short Strangle",
        "expiry":            expiry_str,
        "dte":               dte,
        "management_date":   mgmt_date_str,
        "legs": [
            {"action": "SELL", "type": "PUT",
             "strike": put_row["strike"],
             "mid": put_row["put"]["mid"],
             "delta": put_row["put"]["delta"],
             "prob_otm": put_row["put"]["prob_otm"],
             "volume": put_row["put"]["volume"], "oi": put_row["put"]["oi"]},
            {"action": "SELL", "type": "CALL",
             "strike": call_row["strike"],
             "mid": call_row["call"]["mid"],
             "delta": call_row["call"]["delta"],
             "prob_otm": call_row["call"]["prob_otm"],
             "volume": call_row["call"]["volume"], "oi": call_row["call"]["oi"]},
        ],
        "credit":            credit,
        "max_risk":          "UNLIMITED — undefined risk strategy",
        "roi_pct":           None,
        "put_breakeven":     put_be,
        "call_breakeven":    call_be,
        "put_breakeven_pct": round((put_be  - S) / S * 100, 2),
        "call_breakeven_pct":round((call_be - S) / S * 100, 2),
        "take_profit_at":    round(credit * 0.50, 2),
        "liquidity_ok":      put_row["put"].get("liquid") and call_row["call"].get("liquid"),
        "note":              "UNDEFINED RISK — requires margin/buying power",
    }



# ── Premium Attractiveness Score (0-10) ──────────────────────────────────────

def premium_attractiveness_score(iv_hv, ivr, ivp, best_oi, earnings_days, dte):
    """
    Score 0-10 representing how attractive it is to sell premium TODAY.
    Pure math — no subjective judgment.

    Components:
      IV/HV ratio  40% — core edge: are options overpriced vs realized vol?
      IVR          30% — is IV high vs its own 52wk history?
      IVP          20% — what percentile is IV in?
      Liquidity    10% — can we actually get filled?

    Earnings modifier applied last:
      Earnings within DTE   → ×0.50 (event risk cuts edge in half)
      Earnings just outside → ×0.75 (caution)
      Earnings safe         → ×1.00

    Returns: (score_float, breakdown_dict)
    """

    def interp(val, low, high, pts_low, pts_high):
        """Linear interpolation between two score bands."""
        if val <= low:  return pts_low
        if val >= high: return pts_high
        return pts_low + (val - low) / (high - low) * (pts_high - pts_low)

    # ── IV/HV component (0-10) ────────────────────────────────────────────────
    if iv_hv is None:
        iv_hv_pts = 0.0
    elif iv_hv >= 2.00: iv_hv_pts = 10.0
    elif iv_hv >= 1.75: iv_hv_pts = interp(iv_hv, 1.75, 2.00, 8.0, 10.0)
    elif iv_hv >= 1.50: iv_hv_pts = interp(iv_hv, 1.50, 1.75, 6.0,  8.0)
    elif iv_hv >= 1.25: iv_hv_pts = interp(iv_hv, 1.25, 1.50, 4.0,  6.0)
    elif iv_hv >= 1.10: iv_hv_pts = interp(iv_hv, 1.10, 1.25, 2.0,  4.0)
    elif iv_hv >= 1.00: iv_hv_pts = interp(iv_hv, 1.00, 1.10, 0.0,  2.0)
    else:               iv_hv_pts = 0.0
    iv_hv_pts = round(iv_hv_pts, 2)

    # ── IVR component (0-10) ──────────────────────────────────────────────────
    if ivr is None:
        ivr_pts = 0.0
    elif ivr >= 85: ivr_pts = 10.0
    elif ivr >= 70: ivr_pts = interp(ivr, 70, 85,  8.0, 10.0)
    elif ivr >= 55: ivr_pts = interp(ivr, 55, 70,  6.0,  8.0)
    elif ivr >= 40: ivr_pts = interp(ivr, 40, 55,  4.0,  6.0)
    elif ivr >= 25: ivr_pts = interp(ivr, 25, 40,  2.0,  4.0)
    else:           ivr_pts = interp(ivr,  0, 25,  0.0,  2.0)
    ivr_pts = round(ivr_pts, 2)

    # ── IVP component (0-10) ──────────────────────────────────────────────────
    if ivp is None:
        ivp_pts = 0.0
    elif ivp >= 80: ivp_pts = 10.0
    elif ivp >= 65: ivp_pts = interp(ivp, 65, 80,  7.0, 10.0)
    elif ivp >= 50: ivp_pts = interp(ivp, 50, 65,  5.0,  7.0)
    elif ivp >= 30: ivp_pts = interp(ivp, 30, 50,  3.0,  5.0)
    else:           ivp_pts = interp(ivp,  0, 30,  0.0,  3.0)
    ivp_pts = round(ivp_pts, 2)

    # ── Liquidity component (0-10) ────────────────────────────────────────────
    if best_oi is None:
        liq_pts = 0.0
    elif best_oi >= 5000: liq_pts = 10.0
    elif best_oi >= 1000: liq_pts = interp(best_oi, 1000, 5000, 7.0, 10.0)
    elif best_oi >= 200:  liq_pts = interp(best_oi,  200, 1000, 4.0,  7.0)
    else:                 liq_pts = interp(best_oi,    0,  200, 0.0,  4.0)
    liq_pts = round(liq_pts, 2)

    # ── Weighted raw score ────────────────────────────────────────────────────
    raw = round(
        iv_hv_pts * 0.40 +
        ivr_pts   * 0.30 +
        ivp_pts   * 0.20 +
        liq_pts   * 0.10,
        2
    )

    # ── Earnings modifier ─────────────────────────────────────────────────────
    if earnings_days is not None and dte is not None:
        if earnings_days < dte:
            modifier      = 0.50
            modifier_note = f"earnings in {earnings_days}d within {dte} DTE → ×0.50"
        elif earnings_days < dte + 14:
            modifier      = 0.75
            modifier_note = f"earnings in {earnings_days}d just outside DTE → ×0.75"
        else:
            modifier      = 1.00
            modifier_note = "earnings safe → ×1.00"
    else:
        modifier      = 0.85   # unknown earnings = slight caution
        modifier_note = "earnings unknown → ×0.85"

    final = round(raw * modifier, 1)
    final = max(0.0, min(10.0, final))

    # ── Attractiveness label ──────────────────────────────────────────────────
    if   final >= 8.0: label = "EXCELLENT — strong edge to sell premium"
    elif final >= 6.5: label = "GOOD — solid edge, worth trading"
    elif final >= 5.0: label = "MODERATE — acceptable edge"
    elif final >= 3.5: label = "WEAK — marginal edge, be selective"
    elif final >= 2.0: label = "POOR — little statistical edge"
    else:              label = "SKIP — no meaningful edge right now"

    breakdown = {
        "iv_hv_component":  {"value": iv_hv,     "pts": iv_hv_pts, "weight": 0.40},
        "ivr_component":    {"value": ivr,        "pts": ivr_pts,   "weight": 0.30},
        "ivp_component":    {"value": ivp,        "pts": ivp_pts,   "weight": 0.20},
        "liquidity_component": {"value": best_oi, "pts": liq_pts,   "weight": 0.10},
        "raw_score":        raw,
        "earnings_modifier":modifier,
        "modifier_note":    modifier_note,
        "final_score":      final,
        "label":            label,
    }

    return final, breakdown


# ── Strategy recommendation ───────────────────────────────────────────────────

def recommend_strategy(bias, iv_hv, ivr, skew_signal, strategies,
                       attractiveness_score, earnings_safe):
    """
    Pick the single best strategy from available candidates.
    Pure rule-based — no AI needed.

    Rules (in priority order):
    1. Earnings unsafe → prefer DEFINED risk only
    2. Score < 3.5 → recommend SKIP
    3. Bias + skew → guide direction
    4. IVR level → guide aggressiveness
    """

    if attractiveness_score < 2.0:
        return {
            "recommended":  "SKIP",
            "reason":       f"Attractiveness score {attractiveness_score}/10 too low — no meaningful edge",
            "strategy_data": None,
        }

    available = list(strategies.keys())
    if not available:
        return {
            "recommended":  "SKIP",
            "reason":       "No strategies built — insufficient data or credit",
            "strategy_data": None,
        }

    # Must be defined risk if earnings within DTE
    defined_only = (earnings_safe is False)
    defined_strategies = ["put_spread", "call_spread", "iron_condor"]
    undefined_strategies = ["strangle", "jade_lizard"]

    # Filter by risk type if needed
    if defined_only:
        candidates = [s for s in available if s in defined_strategies]
        risk_note = "defined risk only (earnings within DTE)"
    else:
        candidates = available
        risk_note = "all strategies available"

    if not candidates:
        candidates = available  # fallback

    # Pick based on bias + IVR
    pick = None
    reason = ""

    if bias == "NEUTRAL":
        if ivr and ivr >= 55 and "strangle" in candidates and not defined_only:
            pick   = "strangle"
            reason = f"Neutral bias + IVR {ivr} elevated → undefined strangle maximizes premium"
        elif "iron_condor" in candidates:
            pick   = "iron_condor"
            reason = f"Neutral bias → iron condor collects premium both sides with defined risk"

    elif bias == "BULLISH":
        if skew_signal == "PUT_SKEW" and "jade_lizard" in candidates and not defined_only:
            pick   = "jade_lizard"
            reason = f"Bullish + put skew elevated → jade lizard sells overpriced puts + call spread"
        elif "put_spread" in candidates:
            pick   = "put_spread"
            reason = f"Bullish bias → short put spread profits if stock stays flat or rises"

    elif bias == "BEARISH":
        if "call_spread" in candidates:
            pick   = "call_spread"
            reason = f"Bearish bias → short call spread profits if stock stays flat or falls"

    # Fallback: best by ROI among defined strategies
    if pick is None:
        best_roi = -1
        for name in candidates:
            s = strategies.get(name, {})
            roi = s.get("roi_pct") or 0
            if isinstance(roi, (int, float)) and roi > best_roi:
                best_roi = roi
                pick     = name
        reason = f"Best ROI ({best_roi}%) among available strategies"

    if pick is None:
        pick   = candidates[0]
        reason = "Default selection"

    strat_data = strategies.get(pick, {})

    # Build a clean trade summary
    legs_str = " / ".join(
        f"{l['action']} ${l['strike']} {l['type']} [Δ{l.get('delta','?')}]"
        for l in strat_data.get("legs", [])
    )

    return {
        "recommended":    pick,
        "strategy_name":  strat_data.get("strategy", pick),
        "reason":         reason,
        "risk_constraint": risk_note,
        "legs":           legs_str,
        "expiry":         strat_data.get("expiry"),
        "dte":            strat_data.get("dte"),
        "credit":         strat_data.get("credit"),
        "max_risk":       strat_data.get("max_risk"),
        "roi_pct":        strat_data.get("roi_pct"),
        "breakeven":      strat_data.get("breakeven") or strat_data.get("put_breakeven"),
        "breakeven_pct":  strat_data.get("breakeven_pct") or strat_data.get("put_breakeven_pct"),
        "take_profit_at": strat_data.get("take_profit_at"),
        "management_date":strat_data.get("management_date"),
        "meets_one_third":strat_data.get("meets_one_third", True),
        "liquidity_ok":   strat_data.get("liquidity_ok", True),
        "strategy_data":  strat_data,
    }

# ── Edge + scoring ────────────────────────────────────────────────────────────

def calc_edge(iv_30, hv_30, ivr, ivp, skew_ratio, earnings_safe, dte):
    """Calculate IV edge score 0–100 and one-line summary."""
    score   = 0
    factors = []

    iv_hv = round(iv_30 / hv_30, 3) if hv_30 and hv_30 > 0 else None

    # IV/HV ratio — up to 35 pts
    if iv_hv:
        if iv_hv >= IV_HV_GREAT:   score += 35; factors.append(f"IV/HV {iv_hv} exceptional")
        elif iv_hv >= IV_HV_STRONG: score += 28; factors.append(f"IV/HV {iv_hv} strong")
        elif iv_hv >= IV_HV_GOOD:   score += 20; factors.append(f"IV/HV {iv_hv} good")
        elif iv_hv >= IV_HV_MIN:    score += 10; factors.append(f"IV/HV {iv_hv} weak edge")
        else:                        factors.append(f"IV/HV {iv_hv} no edge")

    # IVR — up to 25 pts
    if ivr is not None:
        if ivr >= 75:   score += 25; factors.append(f"IVR {ivr} top quartile")
        elif ivr >= 60: score += 18; factors.append(f"IVR {ivr} elevated")
        elif ivr >= 40: score += 10; factors.append(f"IVR {ivr} moderate")
        elif ivr >= 30: score += 5;  factors.append(f"IVR {ivr} low-moderate")
        else:            factors.append(f"IVR {ivr} low")

    # IVP — up to 15 pts
    if ivp is not None:
        if ivp >= 80:   score += 15; factors.append(f"IVP {ivp}% high")
        elif ivp >= 60: score += 10; factors.append(f"IVP {ivp}% above median")
        elif ivp >= 40: score += 5;  factors.append(f"IVP {ivp}% near median")

    # Skew — up to 10 pts
    if skew_ratio > 1.3:  score += 10; factors.append(f"put skew {skew_ratio} strong")
    elif skew_ratio > 1.1: score += 5;  factors.append(f"put skew {skew_ratio} mild")

    # Earnings safety — up to 15 pts or penalty
    if earnings_safe is True:
        score += 15; factors.append("earnings safe")
    elif earnings_safe is False:
        score -= 25; factors.append("⚠️ EARNINGS WITHIN DTE — high risk")

    score = max(0, min(100, score))

    if score >= 80:   quality = "EXCELLENT"
    elif score >= 60: quality = "GOOD"
    elif score >= 40: quality = "MODERATE"
    else:             quality = "POOR"

    premium_above_hv = round((iv_hv - 1) * 100, 1) if iv_hv else 0
    summary = (
        f"IVR {ivr} + IV/HV {iv_hv} + skew {skew_ratio} → "
        f"selling premium at {premium_above_hv}% above realized vol [{quality}]"
    )

    return score, quality, summary, iv_hv


# ── Main calculator ───────────────────────────────────────────────────────────

def calculate(ticker_dir):
    ticker_dir = Path(ticker_dir)
    ticker     = ticker_dir.name.upper()

    print(f"\n{'='*60}")
    print(f"🔢 Calculating: {ticker}")

    try:
        return _calculate_inner(ticker_dir, ticker)
    except Exception as e:
        import traceback
        issue = f"{type(e).__name__}: {e}"
        print(f"  💥 Fatal error: {issue}")
        err = {
            "ticker": ticker, "status": "ERROR",
            "data_issues": [issue, traceback.format_exc()[-400:]],
            "strategies": {}, "edge_score": 0, "edge_quality": "FAILED",
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        try:
            (ticker_dir / "calculated_error.json").write_text(json.dumps(err, indent=2))
        except Exception:
            pass
        return err


def _calculate_inner(ticker_dir, ticker):
    # ── Load parsed JSONs ─────────────────────────────────────────────────────
    ov_file = ticker_dir / "optioncharts_overview.json"
    ch_file = ticker_dir / "optioncharts_chain.json"
    yf_file = ticker_dir / "yfinance.json"

    if not ov_file.exists() or not ch_file.exists():
        print(f"  ❌ Missing parsed JSON — run parser.py first")
        return {
            "ticker": ticker, "status": "SKIP",
            "data_issues": ["Missing optioncharts_overview.json or optioncharts_chain.json"],
            "strategies": {}, "edge_score": 0, "edge_quality": "NO_DATA",
        }

    ov = json.loads(ov_file.read_text())
    ch = json.loads(ch_file.read_text())
    yf = json.loads(yf_file.read_text()) if yf_file.exists() else {}

    # ── Key inputs ────────────────────────────────────────────────────────────
    S            = yf.get("current_price") or ch.get("stock_price")
    ma_21        = yf.get("ma_21")
    price_vs_ma  = yf.get("price_vs_ma_pct")
    iv_30        = ov.get("iv_30")
    hv_30        = ov.get("hv_30")
    ivr          = ov.get("iv_rank")
    ivp          = ov.get("iv_pct")
    earn_days    = yf.get("next_earnings_days")
    earn_date    = yf.get("next_earnings_date")
    chain        = ch.get("chain", [])
    chain_dte    = ch.get("dte")

    # ── Chain DTE validation ──────────────────────────────────────────────────
    # If chain DTE is far from 45 days, flag it — calculations will be off
    if chain_dte is not None and chain_dte < 10:
        print(f"  ⚠️  CHAIN DTE WARNING: Chain is {chain_dte} DTE (expected ~45)")
        print(f"      Strategies will show minimal credit — re-fetch with correct date")

    # ── Find best monthly expiry (~45 DTE, always 3rd Friday) ───────────────
    expiry_date, dte, expiry_str, _ = find_best_monthly_expiry(target_dte=45)
    today = date.today()

    # Management date = expiry - 21 days (take profit window)
    mgmt_date    = expiry_date - timedelta(days=21)
    mgmt_str     = mgmt_date.strftime("%Y-%m-%d")
    mgmt_days    = (mgmt_date - today).days

    print(f"  📅 Expiry target : {expiry_str} ({dte} DTE)")
    print(f"  📅 Management    : {mgmt_str} (21 DTE, {mgmt_days} days from today)")

    # ── Earnings safety ───────────────────────────────────────────────────────
    if earn_days is None:
        earnings_safe = None
        earnings_note = "Earnings date unknown — treat as CAUTION"
    elif earn_days < dte:
        earnings_safe = False
        earnings_note = f"⚠️ Earnings in {earn_days} days — WITHIN {dte} DTE window"
    elif earn_days < dte + 14:
        earnings_safe = None   # caution zone
        earnings_note = f"Earnings in {earn_days} days — just outside DTE, monitor closely"
    else:
        earnings_safe = True
        earnings_note = f"Earnings in {earn_days} days — safely outside DTE ✅"

    print(f"  📋 Earnings      : {earnings_note}")

    # ── IV validation ─────────────────────────────────────────────────────────
    iv_note = None
    if iv_30 and iv_30 < 8:
        # Known Barchart parsing bug — estimate from HV + IVR
        if hv_30 and ivr:
            ratio    = 1.0 + (ivr / 100) * 0.8
            iv_30    = round(hv_30 * ratio, 2)
            iv_note  = f"IV estimated as HV×{ratio:.2f} (raw IV was suspicious)"
        else:
            iv_30    = hv_30  # fallback
            iv_note  = "IV estimated from HV (raw IV suspicious, no IVR to adjust)"

    if not iv_30 or not hv_30:
        msg = f"IV({iv_30}) or HV({hv_30}) missing or zero — cannot calculate edge"
        print(f"  ❌ {msg}")
        return {
            "ticker": ticker, "status": "SKIP",
            "data_issues": [msg],
            "strategies": {}, "edge_score": 0, "edge_quality": "NO_DATA",
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    iv_decimal = iv_30 / 100
    T          = dte / 365

    # ── Expected moves ────────────────────────────────────────────────────────
    em1, em2, em1_pct, em2_pct = expected_move(S, iv_decimal, dte)
    straddle_move, straddle_pct, atm_strike = atm_straddle_move(chain, S)

    print(f"  📊 Expected move : ±${em1} (1σ={em1_pct}%) | ±${em2} (2σ={em2_pct}%)")
    print(f"  📊 ATM straddle  : ${straddle_move} (±{straddle_pct}%) at ${atm_strike}")

    # ── Expected move price levels (all 3 sigma levels) ───────────────────────
    em3       = round(S * iv_decimal * math.sqrt(T) * 3, 2)
    em3_pct   = round(em3 / S * 100, 2)
    em_levels = {
        "1sd": {
            "sigma": 1, "probability_pct": 68.27,
            "move":  em1,  "move_pct":  em1_pct,
            "upside":   round(S + em1, 2),
            "downside": round(S - em1, 2),
            "note": "Stock stays in this range 68% of the time by expiry"
        },
        "2sd": {
            "sigma": 2, "probability_pct": 95.45,
            "move":  em2,  "move_pct":  em2_pct,
            "upside":   round(S + em2, 2),
            "downside": round(S - em2, 2),
            "note": "Stock stays in this range 95% of the time by expiry"
        },
        "3sd": {
            "sigma": 3, "probability_pct": 99.73,
            "move":  em3,  "move_pct":  em3_pct,
            "upside":   round(S + em3, 2),
            "downside": round(S - em3, 2),
            "note": "Stock stays in this range 99.7% of the time by expiry"
        },
    }

    # ── Directional bias ──────────────────────────────────────────────────────
    if price_vs_ma is None:
        bias = "UNKNOWN"
    elif price_vs_ma > 1.0:
        bias = "BULLISH"
    elif price_vs_ma < -1.0:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # ── Put/call skew ─────────────────────────────────────────────────────────
    put5  = next((r for r in chain if abs(r["moneyness"] - (-5)) < 2.5), None)
    call5 = next((r for r in chain if abs(r["moneyness"] - (+5)) < 2.5), None)
    skew_ratio = 1.0
    if put5 and call5:
        pm = put5["put"]["mid"] or 0
        cm = call5["call"]["mid"] or 0
        skew_ratio  = round(pm / cm, 3) if cm > 0 else 1.0
    skew_signal = "PUT_SKEW" if skew_ratio > 1.2 else ("CALL_SKEW" if skew_ratio < 0.8 else "NEUTRAL_SKEW")

    # ── Edge score ────────────────────────────────────────────────────────────
    edge_score, edge_quality, edge_summary, iv_hv_ratio = calc_edge(
        iv_30, hv_30, ivr, ivp, skew_ratio, earnings_safe, dte
    )

    print(f"  📊 Edge score    : {edge_score}/100 [{edge_quality}]")
    print(f"  📊 {edge_summary}")

    # ── Enrich chain with delta + tasty scores ────────────────────────────────
    enriched_chain, chain_stats = enrich_chain(chain, S, iv_decimal, T)

    # ── Build all strategy candidates ────────────────────────────────────────
    strategies = {}
    data_issues = []

    def try_build(name, fn, *args):
        try:
            result = fn(*args)
            if result:
                # Attach management rules with price targets
                try:
                    if name == "put_spread":
                        result["management_rules"] = build_management_rules(
                            "put_spread", S,
                            result["legs"][0]["strike"],
                            result["credit"], em_levels, expiry_str, mgmt_str, ticker
                        )
                    elif name == "call_spread":
                        result["management_rules"] = build_management_rules(
                            "call_spread", S,
                            result["legs"][0]["strike"],
                            result["credit"], em_levels, expiry_str, mgmt_str, ticker
                        )
                    elif name == "iron_condor":
                        result["management_rules"] = build_management_rules(
                            "iron_condor", S,
                            result["put_spread"]["legs"][0]["strike"],
                            result["credit"], em_levels, expiry_str, mgmt_str, ticker
                        )
                except Exception as e:
                    data_issues.append(f"Management rules for {name} failed: {e}")
                strategies[name] = result
        except Exception as e:
            data_issues.append(f"Strategy {name} failed: {type(e).__name__}: {e}")

    try_build("put_spread",   build_spread,       enriched_chain, "put",  S, dte, expiry_str, mgmt_str)
    try_build("call_spread",  build_spread,       enriched_chain, "call", S, dte, expiry_str, mgmt_str)
    try_build("iron_condor",  build_iron_condor,  enriched_chain, S, dte, expiry_str, mgmt_str)
    try_build("jade_lizard",  build_jade_lizard,  enriched_chain, S, dte, expiry_str, mgmt_str)
    try_build("strangle",     build_strangle,     enriched_chain, S, dte, expiry_str, mgmt_str)

    print(f"  📋 Strategies    : {list(strategies.keys())}")

    # ── Premium Attractiveness Score ──────────────────────────────────────────
    best_oi = None
    if chain_stats.get("best_put"):
        # Get OI from the best put strike in enriched chain
        for row in enriched_chain:
            if row["strike"] == chain_stats["best_put"]["strike"]:
                best_oi = row["put"].get("oi")
                break

    attract_score, attract_breakdown = premium_attractiveness_score(
        iv_hv_ratio, ivr, ivp, best_oi, earn_days, dte
    )
    print(f"  ⭐ Attractiveness : {attract_score}/10 [{attract_breakdown['label']}]")

    # ── Strategy Recommendation ───────────────────────────────────────────────
    recommendation = recommend_strategy(
        bias, iv_hv_ratio, ivr, skew_signal,
        strategies, attract_score, earnings_safe
    )
    print(f"  🎯 Recommended   : {recommendation['recommended']} — {recommendation['reason'][:60]}")

    # ── Assemble final output ─────────────────────────────────────────────────
    result = {
        "ticker":          ticker,
        "data_issues":     data_issues,
        "calculated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analysis_date":   TODAY,

        # ── Target expiry ──
        "target_expiry":   expiry_str,
        "target_dte":      dte,
        "management_date": mgmt_str,
        "management_days_from_today": mgmt_days,

        # ── Stock context ──
        "stock_price":     S,
        "ma_21":           ma_21,
        "price_vs_ma_pct": price_vs_ma,
        "bias":            bias,

        # ── IV metrics ──
        "iv_30":           iv_30,
        "hv_30":           hv_30,
        "iv_hv_ratio":     iv_hv_ratio,
        "ivr":             ivr,
        "ivp":             ivp,
        "iv_note":         iv_note,

        # ── Earnings ──
        "next_earnings_date": earn_date,
        "next_earnings_days": earn_days,
        "earnings_safe":      earnings_safe,
        "earnings_note":      earnings_note,

        # ── Expected moves ──
        "expected_move": {
            "method":           "black_scholes_lognormal",
            "dte":              dte,
            "stock_price":      S,
            "expiry":           expiry_str,
            "atm_straddle":     straddle_move,
            "atm_straddle_pct": straddle_pct,
            "atm_strike_used":  atm_strike,
            "levels":           em_levels,
        },

        # ── Skew ──
        "skew_ratio":      skew_ratio,
        "skew_signal":     "PUT_SKEW" if skew_ratio > 1.2 else (
                           "CALL_SKEW" if skew_ratio < 0.8 else "NEUTRAL"),

        # ── Edge ──
        "edge_score":      edge_score,
        "edge_quality":    edge_quality,
        "edge_summary":    edge_summary,

        # ── Expected move levels ──
        "expected_move_levels": em_levels,

        # ── Best strikes ──
        "best_put_strike":  chain_stats["best_put"],
        "best_call_strike": chain_stats["best_call"],
        "top_put_candidates":  chain_stats["put_candidates"],
        "top_call_candidates": chain_stats["call_candidates"],

        # ── Strategy candidates ──
        "strategies":      strategies,
        "strategy_count":  len(strategies),

        # ── Premium attractiveness ──
        "attractiveness_score":     attract_score,
        "attractiveness_label":     attract_breakdown["label"],
        "attractiveness_breakdown": attract_breakdown,

        # ── Recommendation ──
        "recommendation": recommendation,

        # ── LLM context block (compact summary for Gemini) ──
        "llm_context": build_llm_context(
            ticker, S, ma_21, price_vs_ma, bias,
            iv_30, hv_30, iv_hv_ratio, ivr, ivp, skew_ratio,
            em1, em1_pct, em2, em2_pct, straddle_move, straddle_pct,
            earn_date, earn_days, earnings_safe,
            expiry_str, dte, mgmt_str, mgmt_days,
            edge_score, edge_quality, edge_summary,
            strategies, attract_score, attract_breakdown, recommendation
        )
    }

    # Save
    out = ticker_dir / "calculated.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  💾 Saved: {out} ({out.stat().st_size:,} bytes)")
    return result


# ── LLM context builder ───────────────────────────────────────────────────────

def build_llm_context(ticker, S, ma_21, price_vs_ma, bias,
                       iv_30, hv_30, iv_hv, ivr, ivp, skew,
                       em1, em1_pct, em2, em2_pct, straddle, straddle_pct,
                       earn_date, earn_days, earn_safe,
                       expiry, dte, mgmt_date, mgmt_days,
                       edge_score, edge_quality, edge_summary,
                       strategies, attract_score=None,
                       attract_breakdown=None, recommendation=None):
    """
    Compact text block for LLM — contains ONLY numbers + labels.
    No HTML, no raw chain, no noise. Just what the LLM needs to decide.
    """
    lines = [
        f"=== {ticker} OPTIONS ANALYSIS ===",
        f"Date: {TODAY} | Score: {edge_score}/100 [{edge_quality}]",
        f"Edge: {edge_summary}",
        "",
        "--- STOCK ---",
        f"Price: ${S} | MA21: ${ma_21} | vs MA: {price_vs_ma:+.2f}% | Bias: {bias}",
        "",
        "--- VOLATILITY ---",
        f"IV(30d): {iv_30}% | HV(30d): {hv_30}% | IV/HV: {iv_hv}",
        f"IV Rank: {ivr}% | IV Pct: {ivp}% | Put/Call Skew: {skew}",
        "",
        f"--- EXPECTED MOVE BY {expiry} ({dte} DTE) ---",
        f"68% of time (1σ): stays ${round(S-em1,2)} to ${round(S+em1,2)} (±${em1} = ±{em1_pct}%)",
        f"95% of time (2σ): stays ${round(S-em2,2)} to ${round(S+em2,2)} (±${em2} = ±{em2_pct}%)",
        f"99% of time (3σ): stays ${round(S-em1*3,2)} to ${round(S+em1*3,2)} (±${round(em1*3,2)} = ±{round(em1_pct*3,2)}%)",
        f"ATM Straddle: ${straddle} (market implied ±{straddle_pct}%)",
        "",
        "--- EARNINGS ---",
        f"Next earnings: {earn_date} ({earn_days} days away)" if earn_date else "Next earnings: UNKNOWN",
        f"Earnings within DTE: {'YES ⚠️' if earn_safe is False else ('CAUTION' if earn_safe is None else 'NO ✅')}",
        "",
        "--- TRADE MANAGEMENT ---",
        f"Target expiry  : {expiry} ({dte} DTE)",
        f"Enter          : Today {TODAY}",
        f"Management date: {mgmt_date} (21 DTE = 50% profit target date, {mgmt_days} days from today)",
        "",
        "--- STRATEGY CANDIDATES ---",
    ]

    for name, strat in strategies.items():
        legs_str = " / ".join(
            f"{l['action']} ${l['strike']} {l['type']} [Δ{l.get('delta','?')}]"
            for l in strat.get("legs", [])
        )
        credit   = strat.get("credit", "?")
        max_risk = strat.get("max_risk", "?")
        roi      = strat.get("roi_pct")
        be       = strat.get("breakeven") or strat.get("put_breakeven")
        be_pct   = strat.get("breakeven_pct") or strat.get("put_breakeven_pct")
        tp       = strat.get("take_profit_at", "?")
        liq      = "✅" if strat.get("liquidity_ok") else "⚠️ low liquidity"
        one3     = "✅" if strat.get("meets_one_third", True) else f"⚠️ below 1/3 rule (min ${strat.get('one_third_min','?')})"

        lines.append(f"\n[{strat['strategy'].upper()}]")
        lines.append(f"Legs     : {legs_str}")
        lines.append(f"Credit   : ${credit} | Max Risk: ${max_risk}" +
                     (f" | ROI: {roi}%" if roi else ""))
        if be:
            lines.append(f"Breakeven: ${be} ({be_pct:+.1f}%)")
        lines.append(f"Take 50% : ${tp} by {mgmt_date} OR when spread = ${tp}")
        lines.append(f"Manage   : Close at 21 DTE ({mgmt_date}) regardless of price")
        lines.append(f"Liquidity: {liq} | 1/3 Rule: {one3}")

        # Add safety check vs expected move
        mgmt = strat.get("management_rules", {})
        safety = mgmt.get("safety_vs_em", {})
        if safety.get("note"):
            lines.append(f"EM Safety: {safety['note']}")
        if mgmt.get("breakeven_note"):
            lines.append(f"BE Rule  : {mgmt['breakeven_note']}")
        if strat.get("note"):
            lines.append(f"Note     : {strat['note']}")
        if strat.get("upside_risk_note"):
            lines.append(f"Note     : {strat['upside_risk_note']}")

    # ── Attractiveness score ─────────────────────────────────────────────────
    if attract_score is not None and attract_breakdown is not None:
        lines.append(f"\n--- PREMIUM ATTRACTIVENESS ---")
        lines.append(f"Score    : {attract_score}/10 [{attract_breakdown['label']}]")
        lines.append(f"IV/HV    : {attract_breakdown['iv_hv_component']['value']} → {attract_breakdown['iv_hv_component']['pts']}/10 pts (×0.40)")
        lines.append(f"IVR      : {attract_breakdown['ivr_component']['value']} → {attract_breakdown['ivr_component']['pts']}/10 pts (×0.30)")
        lines.append(f"IVP      : {attract_breakdown['ivp_component']['value']} → {attract_breakdown['ivp_component']['pts']}/10 pts (×0.20)")
        lines.append(f"Liquidity: OI {attract_breakdown['liquidity_component']['value']} → {attract_breakdown['liquidity_component']['pts']}/10 pts (×0.10)")
        lines.append(f"Raw      : {attract_breakdown['raw_score']}/10")
        lines.append(f"Modifier : {attract_breakdown['modifier_note']}")
        lines.append(f"FINAL    : {attract_score}/10")

    # ── Recommended strategy ──────────────────────────────────────────────────
    if recommendation is not None:
        lines.append(f"\n--- RECOMMENDED STRATEGY ---")
        if recommendation["recommended"] == "SKIP":
            lines.append(f"RECOMMENDATION: SKIP")
            lines.append(f"Reason: {recommendation['reason']}")
        else:
            r = recommendation
            lines.append(f"RECOMMENDATION: {r['strategy_name']}")
            lines.append(f"Reason   : {r['reason']}")
            lines.append(f"Legs     : {r['legs']}")
            lines.append(f"Expiry   : {r['expiry']} ({r['dte']} DTE)")
            lines.append(f"Credit   : ${r['credit']}")
            lines.append(f"Max Risk : ${r['max_risk']}")
            if r.get("roi_pct"):
                lines.append(f"ROI      : {r['roi_pct']}%")
            if r.get("breakeven"):
                lines.append(f"Breakeven: ${r['breakeven']} ({r.get('breakeven_pct',0):+.1f}%)")
            lines.append(f"Take 50% : ${r['take_profit_at']} by {r['management_date']}")
            lines.append(f"Liquidity: {'✅' if r['liquidity_ok'] else '⚠️'} | 1/3 Rule: {'✅' if r['meets_one_third'] else '⚠️ below min'}")
            if r.get("risk_constraint"):
                lines.append(f"Note     : {r['risk_constraint']}")

    lines.append(f"\n=== END {ticker} ===")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        # Can pass ticker name or full path
        arg = sys.argv[1]
        if Path(arg).exists():
            dirs = [Path(arg)]
        else:
            # Treat as ticker name, find in today's folder
            dirs = [DATA_DIR / TODAY / arg.upper()]
    else:
        # All tickers in today's folder that have parsed JSON
        day_dir = DATA_DIR / TODAY
        if not day_dir.exists():
            print(f"❌ No data for {TODAY}")
            sys.exit(1)
        dirs = sorted([
            d for d in day_dir.iterdir()
            if d.is_dir() and (d / "optioncharts_chain.json").exists()
        ])

    print(f"\n{'='*60}")
    print(f"🔢 Calculator — {TODAY} | {len(dirs)} ticker(s)")
    print(f"{'='*60}")

    results = []
    for d in dirs:
        r = calculate(d)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print(f"✅ Calculated: {len(results)}/{len(dirs)}")
    for r in results:
        attract = r.get('attractiveness_score', '?')
        rec     = r.get('recommendation', {}).get('recommended', '?')
        print(f"   {r.get('ticker','?')}: "
              f"attractiveness={attract}/10 | "
              f"strategies={r.get('strategy_count','?')} | "
              f"recommend={rec} | "
              f"expiry={r.get('target_expiry','?')} ({r.get('target_dte','?')} DTE)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()