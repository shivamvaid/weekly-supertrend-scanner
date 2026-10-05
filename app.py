
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Weekly Supertrend Reversal Scanner", page_icon="📈", layout="wide")

NIFTY_URLS = {
    "Nifty 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "Nifty Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(name):
    url = NIFTY_URLS[name]
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    symbol_col = next(c for c in df.columns if c.strip().lower() in ("symbol", "ticker"))
    name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), symbol_col)
    out = pd.DataFrame({
        "symbol": df[symbol_col].astype(str).str.strip(),
        "company": df[name_col].astype(str).str.strip(),
    })
    out["yf"] = out["symbol"] + ".NS"
    return out.drop_duplicates("symbol").reset_index(drop=True)

def weekly_data(ticker):
    df = yf.download(
        ticker,
        period="3y",
        interval="1wk",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.title() for c in df.columns})
    needed = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def indicators(df, atr_period=14, factor=3.0, ema_period=10):
    x = df.copy()
    prev_close = x["Close"].shift(1)
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev_close).abs(),
        (x["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()

    hl2 = (x["High"] + x["Low"]) / 2
    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st_line = pd.Series(index=x.index, dtype=float)
    direction = pd.Series(index=x.index, dtype=int)

    for i in range(len(x)):
        if i == 0 or pd.isna(atr.iloc[i]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            continue

        prev_i = i - 1
        if (basic_upper.iloc[i] < final_upper.iloc[prev_i]) or (x["Close"].iloc[prev_i] > final_upper.iloc[prev_i]):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[prev_i]

        if (basic_lower.iloc[i] > final_lower.iloc[prev_i]) or (x["Close"].iloc[prev_i] < final_lower.iloc[prev_i]):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[prev_i]

        if pd.isna(st_line.iloc[prev_i]):
            st_line.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = -1
        elif st_line.iloc[prev_i] == final_upper.iloc[prev_i]:
            if x["Close"].iloc[i] <= final_upper.iloc[i]:
                st_line.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st_line.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if x["Close"].iloc[i] >= final_lower.iloc[i]:
                st_line.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st_line.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    x["ST"] = st_line
    x["ST_dir"] = direction
    x["EMA10"] = x["Close"].ewm(span=ema_period, adjust=False).mean()
    return x.dropna(subset=["ST", "EMA10"])

def analyze(df):
    if len(df) < 40:
        return None
    x = indicators(df)
    if len(x) < 28:
        return None

    # The last two completed weekly candles must both close above ST.
    last = x.iloc[-1]
    prev = x.iloc[-2]
    confirmed = last["Close"] > last["ST"] and prev["Close"] > prev["ST"]
    if not confirmed:
        return None

    # Find the continuous bearish run immediately before the first confirmation week.
    # A bearish week is defined by Close <= Supertrend.
    first_signal_pos = len(x) - 2
    j = first_signal_pos - 1
    while j >= 0 and x["Close"].iloc[j] <= x["ST"].iloc[j]:
        j -= 1
    bearish_weeks = first_signal_pos - 1 - j

    # Minimum six months ~= 26 weekly candles.
    if bearish_weeks < 26:
        return None

    signal1 = x.iloc[-2]
    signal2 = x.iloc[-1]
    entry = float(signal2["Close"])

    # Recent weekly swing low: lowest low in the 10 weeks before/through confirmation.
    swing_window = x.iloc[max(0, len(x)-12):len(x)-2]
    swing_low = float(swing_window["Low"].min())
    st_stop = float(signal2["ST"])
    stop = min(swing_low, st_stop) * 0.995

    # Technical reference targets. T1 = prior 20-week high before the confirmation.
    prior = x.iloc[:-2]
    t1 = float(prior["High"].tail(20).max())
    t2 = float(prior["High"].tail(52).max())
    if t1 <= entry:
        t1 = entry + max(entry - stop, entry * 0.05)
    if t2 <= t1:
        t2 = t1 + max(entry - stop, entry * 0.10)

    risk = entry - stop
    rr1 = (t1 - entry) / risk if risk > 0 else np.nan
    rr2 = (t2 - entry) / risk if risk > 0 else np.nan

    ema_ok = entry > float(signal2["EMA10"])
    ema_slope = float(signal2["EMA10"] - x["EMA10"].iloc[-4]) if len(x) >= 4 else 0.0
    ema_rising = ema_slope > 0
    body = abs(float(signal2["Close"] - signal2["Open"]))
    candle_range = max(float(signal2["High"] - signal2["Low"]), 1e-9)
    strong_candle = body / candle_range >= 0.55

    # 100-point model.
    score = 0
    score += min(25, 15 + max(0, bearish_weeks - 26) * 0.5)  # duration
    score += 25  # hard confirmation
    score += 10 if ema_ok else 0
    score += 5 if ema_rising else 0
    score += 10 if strong_candle else 5
    score += 10 if rr1 >= 2 else (6 if rr1 >= 1.5 else 2)
    score += 5 if entry > signal2["ST"] else 0
    score = min(100, round(score))

    rating = "A+" if score >= 85 else ("Strong" if score >= 75 else ("Watch" if score >= 65 else "Reject"))

    return {
        "bearish_weeks": bearish_weeks,
        "bearish_months": round(bearish_weeks / 4.345, 1),
        "entry": entry,
        "st": float(signal2["ST"]),
        "ema10": float(signal2["EMA10"]),
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "rr1": rr1,
        "rr2": rr2,
        "ema_ok": ema_ok,
        "ema_rising": ema_rising,
        "strong_candle": strong_candle,
        "score": score,
        "rating": rating,
        "signal1_date": signal1.name.strftime("%Y-%m-%d"),
        "signal2_date": signal2.name.strftime("%Y-%m-%d"),
        "data": x,
    }

def scan_one(row):
    try:
        df = weekly_data(row["yf"])
        result = analyze(df)
        if result:
            return {
                "symbol": row["symbol"],
                "company": row["company"],
                **{k: v for k, v in result.items() if k != "data"},
                "_data": result["data"],
            }
    except Exception:
        return None
    return None

def money(v):
    return f"₹{v:,.2f}" if pd.notna(v) else "—"

st.title("📈 Weekly Supertrend Reversal Scanner")
st.caption("Your strategy: Weekly Supertrend (ATR 14, Factor 3) + EMA 10 • ≥6 months bearish • 2 consecutive weekly closes above Supertrend")

with st.sidebar:
    st.header("Strategy")
    st.write("**Timeframe:** Weekly only")
    st.write("**Supertrend:** ATR 14 / Factor 3")
    st.write("**EMA:** 10")
    st.write("**Bearish minimum:** 26 weeks")
    st.write("**Entry:** 2 consecutive weekly closes above ST")
    st.divider()
    universe_choice = st.selectbox("Universe", ["Combined (Nifty 500)", "Nifty 500", "Nifty Midcap 150", "Nifty Smallcap 250"])
    min_score = st.slider("Minimum score", 0, 100, 65)
    run = st.button("🔄 Scan now", type="primary", use_container_width=True)

st.info("The scanner uses completed weekly candles. It does not use Daily data for the signal.")

if run:
    if universe_choice == "Combined (Nifty 500)":
        universe = load_universe("Nifty 500")
    else:
        universe = load_universe(universe_choice)
    st.write(f"Scanning **{len(universe)} stocks**…")

    results = []
    progress = st.progress(0)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(scan_one, row) for _, row in universe.iterrows()]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r:
                results.append(r)
            progress.progress(i / len(futures))

    results = [r for r in results if r["score"] >= min_score]
    results.sort(key=lambda r: (r["score"], r["bearish_weeks"], r["rr1"]), reverse=True)
    st.session_state["results"] = results
    st.success(f"Scan complete — {len(results)} qualifying setups found.")

if "results" not in st.session_state:
    st.markdown("### What this app finds")
    st.markdown("""
    1. Stock stays **below the red Weekly Supertrend for at least 6 months**.
    2. The bearish run is measured immediately before the reversal.
    3. The stock then gives **two consecutive Weekly closes above Supertrend**.
    4. EMA 10 is used as confirmation.
    5. The app calculates a technical entry, stop, targets, R:R and a setup score.
    """)
    st.warning("For live use, verify the final weekly close after the market week has fully closed.")
    st.stop()

results = st.session_state["results"]
if not results:
    st.warning("No stocks currently meet the selected score threshold.")
    st.stop()

table = pd.DataFrame([{
    "Stock": r["symbol"],
    "Company": r["company"],
    "Bearish": f'{r["bearish_months"]:.1f}M',
    "Entry": money(r["entry"]),
    "ST": money(r["st"]),
    "EMA10": money(r["ema10"]),
    "SL": money(r["stop"]),
    "T1": money(r["t1"]),
    "T2": money(r["t2"]),
    "R:R T1": f'{r["rr1"]:.2f}',
    "Score": r["score"],
    "Rating": r["rating"],
} for r in results])

st.dataframe(table, use_container_width=True, hide_index=True)

selected = st.selectbox("Open setup", [r["symbol"] for r in results])
r = next(x for x in results if x["symbol"] == selected)
x = r["_data"]

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Score", f'{r["score"]}/100')
c2.metric("Bearish period", f'{r["bearish_months"]:.1f} months')
c3.metric("Entry", money(r["entry"]))
c4.metric("Stop", money(r["stop"]))
c5.metric("T1 R:R", f'{r["rr1"]:.2f}')
c6.metric("T2 R:R", f'{r["rr2"]:.2f}')

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=x.index, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name="Weekly"
))
fig.add_trace(go.Scatter(x=x.index, y=x["ST"], name="Supertrend", mode="lines"))
fig.add_trace(go.Scatter(x=x.index, y=x["EMA10"], name="EMA 10", mode="lines"))
fig.add_hline(y=r["entry"], line_dash="dash", annotation_text="Entry")
fig.add_hline(y=r["stop"], line_dash="dot", annotation_text="SL")
fig.add_hline(y=r["t1"], line_dash="dot", annotation_text="T1")
fig.add_hline(y=r["t2"], line_dash="dot", annotation_text="T2")
fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade checklist")
checks = [
    ("≥ 6 months below Weekly Supertrend", True),
    ("Two consecutive Weekly closes above Supertrend", True),
    ("Price above EMA 10", r["ema_ok"]),
    ("EMA 10 rising", r["ema_rising"]),
    ("Strong confirmation candle", r["strong_candle"]),
    ("T1 offers ≥ 2R", r["rr1"] >= 2),
]
for label, ok in checks:
    st.write(("✅ " if ok else "⚠️ ") + label)

st.caption(
    "This is a technical scanner, not investment advice. Stops/targets are mechanical reference levels and should be reviewed against the actual weekly structure before trading."
)
