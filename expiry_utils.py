"""
expiry_utils.py — Monthly expiry date logic.

Rule: Always use 3rd Friday (monthly expiry) with :m suffix.
      Count N months forward from today — simple, deterministic, always liquid.

Default: +2 months forward (~45-60 DTE, Tasty sweet spot)
         If that gives < 35 DTE (early in month), use +3 months instead.
"""

from datetime import datetime, timedelta, date


def get_third_friday(year, month):
    """Return the 3rd Friday of a given year/month."""
    d = datetime(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_until_friday)
    return (first_friday + timedelta(weeks=2)).date()


def get_monthly_expiry(months_ahead):
    """
    Get the 3rd Friday exactly N calendar months from today.
    Returns (expiry_date, dte).
    """
    today        = date.today()
    target_month = today.month + months_ahead
    target_year  = today.year + (target_month - 1) // 12
    target_month = ((target_month - 1) % 12) + 1
    expiry       = get_third_friday(target_year, target_month)
    dte          = (expiry - today).days
    return expiry, dte


def find_best_monthly_expiry(target_dte=45, min_dte=30):
    """
    Find the best monthly expiry by counting forward months.
    
    Logic:
      - Try +2 months first (usually 45-60 DTE)
      - If result < min_dte (we're late in the month), use +3 months
      - Always returns a 3rd Friday with :m suffix
    
    Returns: expiry_date, dte, date_str, suffix
    """
    # Try 2 months ahead first
    expiry, dte = get_monthly_expiry(2)

    # If too close (e.g. today is March 14, April expiry is only 30 DTE)
    if dte < min_dte:
        expiry, dte = get_monthly_expiry(3)

    date_str = expiry.strftime('%Y-%m-%d')
    return expiry, dte, date_str, ":m"


def build_chain_urls(ticker, target_dte=45):
    """
    Build full page URL and async URL for the chain request.
    Always monthly expiry, always :m suffix.
    """
    expiry, dte, date_str, suffix = find_best_monthly_expiry(target_dte)
    mgmt_date = expiry - timedelta(days=21)

    return {
        "expiry":        expiry,
        "dte":           dte,
        "date_str":      date_str,
        "suffix":        suffix,
        "expiry_type":   "monthly",
        "full_url": (
            f"https://optioncharts.io/options/{ticker}/option-chain"
            f"?option_type=all&expiration_dates={date_str}{suffix}"
            f"&view=straddle&strike_range=all"
        ),
        "async_url": (
            f"https://optioncharts.io/async/option_chain"
            f"?expiration_dates={date_str}{suffix}"
            f"&option_type=all&strike_range=all"
            f"&ticker={ticker}&view=straddle"
        ),
        "mgmt_date":     mgmt_date,
        "mgmt_date_str": mgmt_date.strftime('%Y-%m-%d'),
        "mgmt_days":     (mgmt_date - date.today()).days,
    }


if __name__ == "__main__":
    today = date.today()
    print(f"Today: {today}\n")
    print("Monthly expiry options:")
    for m in range(1, 5):
        exp, dte = get_monthly_expiry(m)
        note = " ← SELECTED" if 30 <= (exp - today).days <= 65 and m == (2 if (get_monthly_expiry(2)[1] >= 30) else 3) else ""
        print(f"  +{m} months → {exp} ({dte} DTE) :m{note}")

    print()
    info = build_chain_urls("MSFT")
    print(f"Selected for MSFT:")
    print(f"  Expiry  : {info['expiry']} ({info['dte']} DTE)")
    print(f"  Mgmt    : {info['mgmt_date_str']} (21 DTE)")
    print(f"  Full URL: {info['full_url']}")
    print(f"  Async   : {info['async_url']}")
