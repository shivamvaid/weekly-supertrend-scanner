# Weekly Supertrend Reversal Scanner

A Streamlit web app built around the user's exact weekly strategy.

## Rules

- Universe: Nifty 500 / Nifty Midcap 150 / Nifty Smallcap 250
- Timeframe: Weekly only
- Supertrend: ATR 14, Factor 3
- EMA: 10
- Bearish phase: at least 26 consecutive weekly closes at/below Supertrend immediately before the reversal
- Entry confirmation: 2 consecutive completed weekly closes above Supertrend
- Output: entry, technical stop, T1/T2, R:R, EMA confirmation, score

The app pulls the current index constituent CSVs from Nifty Indices and weekly OHLC data from Yahoo Finance through `yfinance`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Create a Streamlit Community Cloud app from the repository.
4. Set the main file to `app.py`.
5. Deploy.

## Important

- Signals are based on completed weekly candles.
- Yahoo Finance is a convenience data source and can occasionally have delays, corrections, or missing symbols. Verify the final signal on your broker/charting platform before placing a trade.
- The Nifty universe is fetched from official Nifty Indices constituent CSV endpoints.
