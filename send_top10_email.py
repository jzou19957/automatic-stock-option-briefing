import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "reports" / "top10_email.html"

GMAIL_USERNAME = os.getenv("GMAIL_USERNAME", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
TO_EMAIL = "jzou1995@gmail.com"


def main() -> None:
    if not GMAIL_USERNAME or not GMAIL_APP_PASSWORD:
        raise RuntimeError("Missing GMAIL_USERNAME or GMAIL_APP_PASSWORD")
    if not REPORT_PATH.exists():
        raise RuntimeError(f"Email HTML not found: {REPORT_PATH}")

    html = REPORT_PATH.read_text(encoding="utf-8")
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{today_str} — Top 10 Best Premium Selling of 50 Liquid Megacap/ETF"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USERNAME
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USERNAME, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USERNAME, [TO_EMAIL], msg.as_string())

    print(f"Email sent to {TO_EMAIL}")


if __name__ == "__main__":
    main()
