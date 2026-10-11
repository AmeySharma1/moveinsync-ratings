"""
MoveInSync Play Store ratings analysis (reusable scorecard)  -  v3

Setup (once, e.g. in Colab):
    pip install google-play-scraper tabulate pandas matplotlib

Run (downloads fresh reviews, then analyses):
    python moveinsync_ratings_analysis.py

Re-analyse an existing download without fetching again (fast):
    python moveinsync_ratings_analysis.py --from-csv output/employee_app_reviews_raw.csv

Outputs go to ./output/ :
    <app>_scorecard.md            the one-page answer to "how are our ratings?"
    <app>_theme_table.csv         theme sizes + rating drag (last 365d and last 90d)
    <app>_version_ratings.csv     rating by app version
    <app>_reply_rate_by_half.csv  developer reply rate on 1-2 star reviews over time
    <app>_monthly.csv, charts (.png)
    <app>_validation_sample.csv   100 random recent 1-2 star reviews: check the tags by hand
    <app>_lowstar_to_read.csv     newest 300 1-2 star reviews: read these
    <app>_reviews_raw.csv / _tagged.csv
"""
import argparse
import os
import time
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*match groups.*")

# ---------------- CONFIG ----------------
APP_IDS = {
    "employee_app": "com.moveinsync.ets",
    # "driver_app": "REPLACE_WITH_DRIVER_APP_ID",
}
COUNTRY = "in"
# language -> max reviews to download (Hindi reviews are not returned when lang="en")
LANG_LIMITS = {"en": 20000, "hi": 5000}
OUT = "output"
RECENT_DAYS = 90
LONG_DAYS = 365

# ---------------- THEMES ----------------
# Built from reading real 1-2 star reviews (Oct 2026). Multi-label: one review can hit several.
# owner: App = product/engineering can fix; Ops = vendor/routing/transport desk; Mixed = both.
THEMES = {
    "App won't open / crashes / blank screen": ("App",
        r"\bcrash|\bnot (?:able to )?(?:open|launch|load)\b|\bnot opening\b|"
        r"\b(?:won'?t|wont|doesn'?t|does not|can'?t|cannot|unable to) (?:open|launch|load|start)\b|"
        r"\bopen(?:s|ing)? (?:and|then) clos|\bclos(?:es|ing) (?:automatically|itself|by itself|as soon)|"
        r"\bautomatic(?:ally)? clos|\bblank screen\b|\bwhite screen\b|\bhangs?\b|\bfreez|\bstuck\b|"
        r"\bforce ?clos|\bnot working\b|\bapp error\b|\buninstall|\breinstall|\bslow\b|\bloading\b|\bbug|"
        r"\blag(?:s|gy|ging)?\b|\blatency\b|\btak(?:es?|ing) (?:too )?(?:long|much time|forever)\b|\bsluggish|"
        r"\bglitch|\bnot stable\b|\bsomething went wrong\b"),
    "False 'not installed from Play Store' error": ("App",
        r"\bnot (?:been )?(?:downloaded|installed)\b\W+(?:\w+\W+){0,2}?(?:from|in|thru|through|via|on)\W+"
        r"(?:the )?(?:google )?play ?store"),
    "OTP not received / login fails / logged out": ("App",
        r"\botp\b|\blog ?in\b|\blogin\b|\blog(?:s|ging)? (?:me |us |you )?out\b|\blogged out\b|\bsign ?in\b|"
        r"\bnot verif|\bverification\b|\bregist|\bpassword\b|\bvalidate my identity\b|"
        r"\bsession expired\b|\bunauthori[sz]ed\b"),
    "Forced rating / no skip": ("App",
        r"(?:forc\w*|compuls\w*|mandatory|keeps? (?:on )?asking|always asking|asks? (?:me )?(?:for|to)|"
        r"asking (?:for|to)|no (?:option|way) to skip|skip)\W+(?:\w+\W+){0,7}?rat(?:e|ing|ings)\b|"
        r"\brat(?:e|ing)\b\W+(?:\w+\W+){0,7}?(?:skip|forc\w*|compuls\w*|every ride|each ride|every trip)"),
    "Intrusive pop-ups / notifications / permissions": ("App",
        r"\bnotification|\bnotif\b|\bpop ?-?ups?\b|\bpermission|\bspam|\balerts?\b|\bmessage comes\b|"
        r"\bterms and (?:conditions|agreements?)\b|\btransport policy\b|\bagree and agree\b"),
    "Seat-belt / safety-check friction": ("App",
        r"seat ?-?belt|\bbuckle\b|safety check"),
    "New update / redesign / confusing UX": ("App",
        r"\bnew update\b|\blatest update\b|\bafter (?:the )?(?:last |every |each |recent |new )?updates?\b|"
        r"\bupdate (?:made|is|was|has)\b|\bold version\b|\bold (?:ui|look)\b|"
        r"\bnew (?:feature|ui|version|design|look)s?\b|\bconfus|\bcomplicat|\bsimpl(?:ify|e|icity)\b|"
        r"\bui\b|\bux\b|\binterface\b|\buser[- ]friendly\b|\bnot intuitive\b|\bhard to (?:use|navigate)\b|"
        r"\bnavigat|\bdesign"),
    "Booking / cancel / no-show / scheduling": ("App",
        r"\bbook(?:ing|ings|ed|s)?\b|\bschedul|\broster\b|\bslot\b|\bcut ?-?off\b|\bcancel|\bno[- ]?show\b|"
        r"\bwait ?-?list\b|\bcalendar\b|\bseat (?:number|selection|booking)\b|\bad-?hoc\b|"
        r"\bchange (?:a |the )?(?:destination|address)\b"),
    "Live tracking stale / wrong location": ("App",
        r"\btrack|\bgps\b|\blive (?:location|boarding)\b|\blocation\b|\bmaps?\b|\bmapping\b|"
        r"\bnot updat|\bwhere is my\b"),
    "Device / OS compatibility": ("App",
        r"\bandroid ?\d*|\bpixel\b|\boneplus\b|\bone plus\b|\bvivo\b|\bsamsung\b|\boppo\b|\brealme\b|"
        r"\bxiaomi\b|\bredmi\b|\bhuawei\b|\bmotorola\b|\biphone\b|\bios\b|\bmobile data\b|\bwi-?fi\b"),
    "ETA wrong / cab late": ("Ops",
        r"\blate\b|\bdelay|\beta\b|\bon[- ]?time\b|\bwaiting\b|\bwait\b|\bnever (?:came|arrived|shows?)\b|"
        r"\bdon'?t come\b|\bdidn'?t (?:come|arrive)\b|\bnot (?:come|arrive)\b|\bpunctual|\bnot reach"),
    "Routing / clubbing / long commute / pickup point": ("Ops",
        r"\brout(?:e|es|ing)\b|\bclub|\bpick ?-?up (?:point|location|time|spot)s?\b|"
        r"\bdrop ?-?(?:off )?(?:point|location)s?\b|\bwrong (?:place|location|address|stop|route)\b|"
        r"\bmiddle of nowhere\b|\bdetour|\blong (?:route|way)\b|\b\d(?:\.\d)? hours? (?:to|for|in)\b|"
        r"\btoo early\b|\bshorter\b|\bnodal\b|\bstops?\b|\bdrop sequence\b"),
    "Cab allocation / transport service quality": ("Ops",
        r"\bno (?:cab|vehicle|ride|shuttle|bus|car)s?\b|\bnot (?:assigned|allocated)\b|\ballocat|"
        r"\btransport(?:ation)? (?:team|service|services|management|system|facility)\b|"
        r"\bcab (?:service|services|facility|management)\b|\bmonopoly\b|\bfacility\b"),
    "Helpdesk / support / feedback unreachable": ("Mixed",
        r"\bhelp ?-?desk\b|\b(?:customer|tech(?:nical)?|no|zero|poor|bad|useless) support\b|"
        r"\bsupport (?:team|desk|staff|executive|person)\b|\bcustomer (?:care|service)\b|\bcomplaint|"
        r"\bgrievance|\bescalat|\bnot helping\b|\bno (?:one|body) (?:is )?(?:help|respond|reply|call)|"
        r"\bno response\b|\bnot responding\b|\bunreachable\b|\bnot reachable\b|\bno feedback\b|"
        r"\bfeedback (?:option|link|form|box|button|text ?box)\b|"
        r"\b(?:can'?t|cannot|unable to|not able to) (?:give|send|write|submit) feedback"),
    "Driver behaviour / safety": ("Ops",
        r"\brude\b|\bmisbehav|\babusive\b|\bdisrespect|\brash\b|\bunprofessional|\bunsafe\b|\bsafety\b|"
        r"\bfemale\b|\bwomen\b|\bgirl\b|\bharass|\bmanners\b|\bdrunk\b|\bdriver (?:behaviou?r|attitude)\b|"
        r"\bdrivers? (?:is|are|was|were) (?:very )?(?:bad|rude|careless)\b|\bdirty\b|\bsmell"),
}
GENERIC_MAX_WORDS = 6   # short low-star reviews with no theme = "no actionable detail"


# ---------------- DATA ----------------
def fetch_lang(app_id, lang, limit):
    """Download newest reviews for one language in pages of 200, with retries."""
    from google_play_scraper import reviews, Sort
    rows, token, fails = [], None, 0
    while len(rows) < limit:
        try:
            batch, token = reviews(app_id, lang=lang, country=COUNTRY,
                                   sort=Sort.NEWEST, count=200, continuation_token=token)
        except Exception as e:
            fails += 1
            print(f"  [{lang}] request failed ({type(e).__name__}); retry {fails}/5")
            if fails >= 5:
                break
            time.sleep(2 * fails)
            continue
        fails = 0
        rows.extend(batch)
        print(f"  [{lang}] fetched {len(rows)} reviews")
        if not batch or token is None:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if len(df):
        df["fetch_lang"] = lang
    return df


def fetch(label, app_id):
    from google_play_scraper import app as gp_app
    parts = [fetch_lang(app_id, lang, lim) for lang, lim in LANG_LIMITS.items()]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise SystemExit("No reviews fetched. Check internet / package id and try again.")
    df = pd.concat(parts, ignore_index=True)
    df["app"] = label
    return df, get_meta(app_id)


def get_meta(app_id):
    try:
        from google_play_scraper import app as gp_app
        return gp_app(app_id, lang="en", country=COUNTRY)
    except Exception as e:
        print(f"  (could not fetch live app info: {type(e).__name__})")
        return {}


def clean(df):
    for col, default in [("replyContent", None), ("appVersion", None),
                         ("thumbsUpCount", 0), ("content", "")]:
        if col not in df.columns:
            df[col] = default
    df = df.drop_duplicates(subset="reviewId").copy()
    df["at"] = pd.to_datetime(df["at"])
    df["month"] = df["at"].dt.to_period("M").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["appVersion"] = df["appVersion"].fillna("unknown").astype(str)
    df["has_reply"] = df["replyContent"].notna()
    df["is_devanagari"] = df["content"].str.contains(r"[ऀ-ॿ]", regex=True)
    df["words"] = df["content"].str.split().str.len().fillna(0).astype(int)
    return df.sort_values("at", ascending=False).reset_index(drop=True)


def tag(df):
    text = df["content"].str.lower()
    for theme, (_, pat) in THEMES.items():
        df[theme] = text.str.contains(pat, regex=True).fillna(False).astype(bool)
    df["any_theme"] = df[list(THEMES)].any(axis=1)
    df["generic_negative"] = (df["score"] <= 2) & (~df["any_theme"]) & (df["words"] <= GENERIC_MAX_WORDS)
    return df


# ---------------- ANALYSIS ----------------
def window(df, days):
    return df[df["at"] >= df["at"].max() - pd.Timedelta(days=days)]


def theme_table(df, window_label):
    n = len(df)
    low = df[df["score"] <= 2]
    rows = []
    for theme, (owner, _) in THEMES.items():
        m, m_low = df[theme], low[theme]
        uplift = ((4 - low.loc[m_low, "score"]).sum() / n) if n else 0
        rows.append({
            "window": window_label,
            "theme": theme,
            "owner": owner,
            "low_star_reviews": int(m_low.sum()),
            "pct_of_1_2_star": round(100 * m_low.mean(), 1) if len(low) else 0,
            "pct_of_all_reviews": round(100 * m.mean(), 1) if n else 0,
            "rating_drag_if_fixed(+stars)": round(float(uplift), 3),
        })
    return pd.DataFrame(rows).sort_values("rating_drag_if_fixed(+stars)", ascending=False)


def owner_split(df):
    low = df[df["score"] <= 2]
    out = {}
    for owner in ["App", "Ops", "Mixed"]:
        cols = [t for t, (o, _) in THEMES.items() if o == owner]
        out[owner] = round(100 * low[cols].any(axis=1).mean(), 1) if len(low) else 0
    out["No theme"] = round(100 * (~low["any_theme"]).mean(), 1) if len(low) else 0
    out["of which generic (<=6 words)"] = round(100 * low["generic_negative"].mean(), 1) if len(low) else 0
    return out


def version_table(df):
    g = df.groupby("appVersion").agg(
        reviews=("score", "size"),
        avg_rating=("score", "mean"),
        pct_1_2_star=("score", lambda s: 100 * (s <= 2).mean()),
        first_seen=("at", "min"),
        last_seen=("at", "max"),
    ).reset_index()
    g["avg_rating"] = g["avg_rating"].round(2)
    g["pct_1_2_star"] = g["pct_1_2_star"].round(1)
    return g.sort_values("first_seen", ascending=False)


def reply_rate_by_half(df):
    low = df[df["score"] <= 2].copy()
    low["half"] = low["at"].dt.year.astype(str) + "H" + ((low["at"].dt.month > 6) + 1).astype(str)
    t = low.groupby("half").agg(low_star_reviews=("score", "size"),
                                reply_pct=("has_reply", lambda s: round(100 * s.mean(), 1)))
    return t.reset_index()


def stat_line(w):
    n = len(w)
    if n < 2:
        return "n/a"
    ci = 1.96 * w["score"].std() / np.sqrt(n)
    return (f"n={n} | avg {w['score'].mean():.2f} (+/-{ci:.2f}) | 1-2 star {100 * (w['score'] <= 2).mean():.1f}% "
            f"| 1 star {100 * (w['score'] == 1).mean():.1f}% | 5 star {100 * (w['score'] == 5).mean():.1f}%")


# ---------------- CHARTS ----------------
def charts(df, label, vt):
    os.makedirs(OUT, exist_ok=True)
    m = df.groupby("month").agg(avg=("score", "mean"),
                                low_share=("score", lambda s: 100 * (s <= 2).mean()),
                                n=("score", "size")).reset_index()
    m = m[m["n"] >= 20].tail(36)
    if len(m):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(m["month"], m["avg"], marker="o")
        axes[0].set_ylim(1, 5)
        axes[0].set_title(f"{label}: monthly average of written reviews (last 36 months)")
        axes[1].plot(m["month"], m["low_share"], marker="o", color="tab:red")
        axes[1].set_title("Share of 1-2 star reviews (%)")
        axes[1].tick_params(axis="x", rotation=90)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{label}_monthly_rating.png", dpi=150)
        plt.close()

    v = vt[(vt["reviews"] >= 30) & (vt["first_seen"] >= df["at"].max() - pd.Timedelta(days=548))]
    v = v.sort_values("first_seen")
    if len(v):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(v["appVersion"], v["avg_rating"])
        ax.set_title(f"{label}: avg rating by app version (>=30 reviews, last 18 months)")
        ax.set_ylim(1, 5)
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{label}_rating_by_version.png", dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    df["score"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"{label}: star distribution (downloaded reviews)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{label}_star_dist.png", dpi=150)
    plt.close()


# ---------------- HELPER FILES ----------------
def write_helper_files(df, label):
    themes = list(THEMES)
    recent_low = window(df, LONG_DAYS)
    recent_low = recent_low[recent_low["score"] <= 2]
    s = recent_low.sample(min(100, len(recent_low)), random_state=42)
    s = s[["reviewId", "at", "score", "appVersion", "content"]].copy()
    s["auto_themes"] = df.loc[s.index, themes].apply(
        lambda r: "; ".join([t for t in themes if r[t]]), axis=1)
    s["tag_correct (Y/N)"] = ""
    s["right_theme_if_wrong"] = ""
    s.to_csv(f"{OUT}/{label}_validation_sample.csv", index=False, encoding="utf-8-sig")
    low = df[df["score"] <= 2].head(300)[["at", "score", "appVersion", "thumbsUpCount", "content"]]
    low.to_csv(f"{OUT}/{label}_lowstar_to_read.csv", index=False, encoding="utf-8-sig")


# ---------------- SCORECARD ----------------
def top_quotes(df, theme, k=3):
    s = window(df, LONG_DAYS)
    s = s[(s[theme]) & (s["score"] <= 2) & (s["words"].between(5, 22))]
    s = s.sort_values("thumbsUpCount", ascending=False).head(k)
    return [f'"{t.strip()}" ({sc}*, v{ver})'
            for t, sc, ver in zip(s["content"], s["score"], s["appVersion"])]


def to_md(d):
    try:
        return d.to_markdown(index=False)
    except ImportError:
        return d.to_string(index=False)


def latest_release_health(df, meta):
    ver = str(meta.get("version") or "")
    if not ver:
        vt = df[df["appVersion"] != "unknown"]
        ver = vt.iloc[0]["appVersion"] if len(vt) else ""
    cur = df[df["appVersion"] == ver]
    base = window(df[df["appVersion"] != ver], LONG_DAYS)
    if not len(cur):
        return f"- Latest version {ver or 'n/a'}: no reviews yet"
    flag = ""
    if len(cur) >= 30 and cur["score"].mean() < base["score"].mean() - 0.2:
        flag = "  **ALERT: below baseline by >0.2**"
    return (f"- Latest version **{ver}**: n={len(cur)}, avg {cur['score'].mean():.2f}, "
            f"1-2 star {100 * (cur['score'] <= 2).mean():.1f}%  vs 12-month baseline of other versions: "
            f"avg {base['score'].mean():.2f}, 1-2 star {100 * (base['score'] <= 2).mean():.1f}%{flag}")


def scorecard(label, meta, df, t_long, t_recent, rr):
    end = df["at"].max()
    score = meta.get("score")
    prev = df[(df["at"] < end - pd.Timedelta(days=RECENT_DAYS)) &
              (df["at"] >= end - pd.Timedelta(days=2 * RECENT_DAYS))]
    low12 = window(df, LONG_DAYS)
    low12 = low12[low12["score"] <= 2]
    rr90 = window(df, RECENT_DAYS)
    rr90 = rr90[rr90["score"] <= 2]
    lines = [
        f"# Ratings scorecard: {label}  (generated {dt.date.today()})",
        "",
        f"- Play Store overall rating: **{f'{score:.2f}' if score else 'n/a'}** "
        f"({meta.get('ratings', 'n/a')} star ratings) | written reviews analysed: {len(df)} "
        f"({df['at'].min().date()} to {end.date()})",
        "- Note: the Play score includes people who only tapped stars; written reviews skew negative, "
        "so their average is lower than the Play score.",
        "",
        "## Written-review windows",
        f"- Last 30d:  {stat_line(window(df, 30))}",
        f"- Last {RECENT_DAYS}d:  {stat_line(window(df, RECENT_DAYS))}",
        f"- Previous {RECENT_DAYS}d: {stat_line(prev)}",
        f"- Last {LONG_DAYS}d: {stat_line(window(df, LONG_DAYS))}",
        "",
        "## Latest release health",
        latest_release_health(df, meta),
        "",
        "## Developer replies to 1-2 star reviews",
        f"- Reply rate last {RECENT_DAYS}d: {100 * rr90['has_reply'].mean():.1f}%  | "
        f"last {LONG_DAYS}d: {100 * low12['has_reply'].mean():.1f}%  | all time: "
        f"{100 * df.loc[df['score'] <= 2, 'has_reply'].mean():.1f}%",
        to_md(rr.tail(6)),
        "",
        f"## Who owns the 1-2 star complaints? (last {LONG_DAYS}d, % of 1-2 star reviews; themes overlap)",
        " | ".join(f"{k}: {float(v):.1f}%" for k, v in owner_split(window(df, LONG_DAYS)).items()),
        "",
        f"## Theme ranking by rating drag (last {LONG_DAYS}d)",
        to_md(t_long.head(12)[["theme", "owner", "low_star_reviews", "pct_of_1_2_star",
                               "rating_drag_if_fixed(+stars)"]]),
        "",
        f"## Same, last {RECENT_DAYS}d (smaller sample, use as direction only)",
        to_md(t_recent.head(8)[["theme", "owner", "low_star_reviews", "pct_of_1_2_star",
                                "rating_drag_if_fixed(+stars)"]]),
        "",
        f"## Example recent 1-2 star quotes (last {LONG_DAYS}d) for the top themes",
    ]
    for th in t_long.head(6)["theme"]:
        lines.append(f"**{th}**")
        for q in top_quotes(df, th):
            lines.append(f"- {q}")
    return "\n".join(lines)


# ---------------- MAIN ----------------
def analyse(label, df_raw, meta):
    os.makedirs(OUT, exist_ok=True)
    df = tag(clean(df_raw))
    df.to_csv(f"{OUT}/{label}_reviews_tagged.csv", index=False, encoding="utf-8-sig")
    t_long = theme_table(window(df, LONG_DAYS), f"last_{LONG_DAYS}d")
    t_recent = theme_table(window(df, RECENT_DAYS), f"last_{RECENT_DAYS}d")
    pd.concat([t_long, t_recent]).to_csv(f"{OUT}/{label}_theme_table.csv", index=False, encoding="utf-8-sig")
    vt = version_table(df)
    vt.to_csv(f"{OUT}/{label}_version_ratings.csv", index=False)
    rr = reply_rate_by_half(df)
    rr.to_csv(f"{OUT}/{label}_reply_rate_by_half.csv", index=False)
    df.groupby("month").agg(avg=("score", "mean"), n=("score", "size")).to_csv(f"{OUT}/{label}_monthly.csv")
    charts(df, label, vt)
    write_helper_files(df, label)
    with open(f"{OUT}/{label}_scorecard.md", "w", encoding="utf-8") as f:
        f.write(scorecard(label, meta, df, t_long, t_recent, rr))
    print(f"Done {label}. See {OUT}/")
    return df


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", help="re-analyse an existing *_reviews_raw.csv (skips download)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    for label, app_id in APP_IDS.items():
        if args.from_csv:
            raw = pd.read_csv(args.from_csv)
            meta = get_meta(app_id)
        else:
            print(f"Fetching {label} ({app_id}) ...")
            raw, meta = fetch(label, app_id)
            raw.to_csv(f"{OUT}/{label}_reviews_raw.csv", index=False, encoding="utf-8-sig")
        analyse(label, raw, meta)


if __name__ == "__main__":
    main()
