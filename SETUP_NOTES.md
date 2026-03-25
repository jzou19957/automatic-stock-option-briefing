# GitHub Actions starter bundle

## Put these files in your repo
- `.github/workflows/daily-options-email.yml`
- `download_universe_csv.py`
- `build_top10_email_report.py`
- `send_top10_email.py`
- `requirements-actions.txt`

## Required existing project files
These files are assumed to already exist in your repo root:
- `fetch.py`
- `parser.py`
- `calculator.py`
- `universe.csv` is generated automatically by `download_universe_csv.py`

## GitHub Secrets to add
- `UNIVERSE_CSV_URL`
- `GEMINI_API_KEY`
- `GMAIL_USERNAME`
- `GMAIL_APP_PASSWORD`

## Google Sheet CSV URL format
Replace gid if your sheet tab is not the first tab:

`https://docs.google.com/spreadsheets/d/1U-mYdJ5kObYRXrgOW40yedpEgzhY7_U4T-TcbrEQuSs/export?format=csv&gid=0`

## Expected flow
1. Workflow downloads `universe.csv` from Google Sheets.
2. `fetch.py` runs for all symbols in that CSV.
3. `parser.py` parses the fetched data.
4. `calculator.py` creates `calculated.json` per ticker.
5. `build_top10_email_report.py` sorts all `calculated.json` files by `attractiveness_score`, keeps the top 10, asks Gemini to format each one, and writes `reports/top10_email.html`.
6. `send_top10_email.py` emails the HTML report to `jzou1995@gmail.com`.

## Recommended first test
Temporarily comment out the `Send email` step in the workflow and run the action manually.
Check the uploaded artifact for:
- `reports/top10_email_payload.json`
- `reports/top10_email.html`
- your latest `data/...` output

Then turn the email step back on.
