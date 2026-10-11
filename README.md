# MoveInSync employee app: Play Store ratings monitor

Answers one question, every time it is asked: **"How are our Play Store ratings?"**

It downloads the newest public Play Store reviews of the employee app (`com.moveinsync.ets`),
tags each complaint by theme (booking, login/OTP, crashes, tracking, ...), and writes a one-page
scorecard in about five minutes.

## What the scorecard shows

- Play score and written-review average for the last 30, 90 and 365 days, with margins of error
- Share of 1-2 star and 1-star reviews
- **Latest release health** (flags a release whose average is more than 0.2 below the 12-month baseline)
- Developer reply rate on 1-2 star reviews, by half-year
- Themes ranked by "max gain" (how much the written-review average would rise if that theme's
  1-2 star reviews had been 4 stars), with an owner: App, Ops or Mixed
- Recent example quotes for the top themes

## Run it

```bash
pip install -r requirements.txt
python moveinsync_ratings_analysis.py                 # download fresh reviews, then analyse
python moveinsync_ratings_analysis.py --from-csv output/employee_app_reviews_raw.csv   # re-analyse a saved download
```

Works in Google Colab too (`!pip install ...` then `!python ...`).

A GitHub Action (`.github/workflows/ratings.yml`) runs it on the 1st of every month and attaches the
scorecard and charts to the run (Actions tab, "Artifacts"). It can also be started by hand.

## Outputs (`output/`)

| File | What it is |
| --- | --- |
| `*_scorecard.md` | The one-page answer |
| `*_theme_table.csv` | Theme sizes and max gain (last 365 and 90 days) |
| `*_version_ratings.csv` | Rating by app version |
| `*_reply_rate_by_half.csv` | Reply rate to 1-2 star reviews over time |
| `*_monthly.csv`, `*.png` | Monthly series and charts |
| `*_validation_sample.csv` | 100 random recent low-star reviews to check the tags by hand |
| `*_lowstar_to_read.csv` | Newest 300 low-star reviews to read |

## Limits

- Written reviews skew negative and ignore users who only tap stars.
- Themes are keyword rules (English). About 85% of tags were judged reasonable in an informal spot-check
  of 40 reviews; use theme shares as directional.
- The download returned no Hindi-script reviews, so Hindi reviews are not covered.
- A new release has few reviews in its first weeks; treat early gaps as suggestive.
- No internal data (crash rates, OTP logs, booking failures) is used.

Built with Python (google-play-scraper, pandas, matplotlib). Written with help from Claude, an AI assistant.
