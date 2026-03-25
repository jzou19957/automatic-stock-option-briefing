import os
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "reports" / "top10_email.html"

GMAIL_USERNAME = os.getenv("GMAIL_USERNAME", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()

TO_EMAILS = ["jzou1995@gmail.com"]


def main():
    if not GMAIL_USERNAME or not GMAIL_APP_PASSWORD:
        raise RuntimeError("Missing GMAIL_USERNAME or GMAIL_APP_PASSWORD")

    if not REPORT_PATH.exists():
        raise RuntimeError(f"Email HTML not found: {REPORT_PATH}")

    html = REPORT_PATH.read_text(encoding="utf-8")
    if not html.strip():
        raise RuntimeError("Email HTML file is empty")

    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{today_str} — Top 10 Best Premium Selling of 50 Liquid Megacap/ETF"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USERNAME
    msg["To"] = ", ".join(TO_EMAILS)

    plain_text = (
        f"{subject}\n\n"
        f"This email contains the HTML version of today's top 10 premium selling setups.\n"
        f"If your mail client does not render HTML, open the GitHub Actions artifact instead."
    )

    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as server:
            server.login(GMAIL_USERNAME, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USERNAME, TO_EMAILS, msg.as_string())

        print(f"Email sent to: {', '.join(TO_EMAILS)}")

    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            "SMTP authentication failed. Check GMAIL_USERNAME and GMAIL_APP_PASSWORD secrets."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to send email: {e}") from e


if __name__ == "__main__":
    main()