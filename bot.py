"""Bot notifikasi Supertrend -> Telegram.

Dijalankan terjadwal (GitHub Actions). Setiap run:
1. Ambil data harga dari Yahoo Finance (yfinance)
2. Hitung Supertrend (faktor & ATR sesuai pengaturan)
3. Jika ada pergantian arah baru sejak run sebelumnya, kirim pesan Telegram
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SYMBOL = os.getenv("SYMBOL") or "GC=F"        # emas futures di Yahoo
INTERVAL = os.getenv("INTERVAL") or "15m"     # contoh: 5m, 15m, 30m, 1h
PERIOD = os.getenv("PERIOD") or "30d"
FACTOR = float(os.getenv("FACTOR") or "3")
ATR_LEN = int(os.getenv("ATR_LEN") or "15")
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "state.json"


def interval_delta(text):
    if text.endswith("m"):
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("h"):
        return pd.Timedelta(hours=int(text[:-1]))
    if text.endswith("d"):
        return pd.Timedelta(days=int(text[:-1]))
    raise ValueError("INTERVAL tidak dikenali: " + text)


def fetch():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if df.empty:
        raise RuntimeError("Data kosong dari Yahoo Finance untuk " + SYMBOL)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # buang candle yang masih berjalan, hanya pakai candle yang sudah ditutup
    step = interval_delta(INTERVAL)
    if df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]
    return df


def supertrend(df, factor, n):
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    hl2 = (h + l) / 2.0
    up = hl2 + factor * atr
    lo = hl2 - factor * atr
    fu, fl = up.copy(), lo.copy()
    direction = np.ones(len(c), dtype=int)  # 1 = uptrend (BUY), -1 = downtrend (SELL)

    for i in range(1, len(c)):
        fl[i] = lo[i] if (lo[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        if direction[i - 1] == -1:
            direction[i] = 1 if c[i] > fu[i] else -1
        else:
            direction[i] = -1 if c[i] < fl[i] else 1

    line = np.where(direction == 1, fl, fu)
    return direction, line, c


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID belum diisi di Secrets")
    r = requests.post(
        "https://api.telegram.org/bot" + TOKEN + "/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )
    r.raise_for_status()


def wib(ts):
    return ts.tz_convert("Asia/Jakarta").strftime("%d %b %Y %H:%M WIB")


def main():
    df = fetch()
    direction, line, close = supertrend(df, FACTOR, ATR_LEN)

    warmup = ATR_LEN * 3
    flips = [i for i in range(max(1, warmup), len(direction))
             if direction[i] != direction[i - 1]]
    if not flips:
        print("Belum ada pergantian arah dalam data.")
        return

    i = flips[-1]
    flip_time = df.index[i].isoformat()
    close_time = df.index[i] + interval_delta(INTERVAL)
    side = "BUY" if direction[i] == 1 else "SELL"
    now_side = "BUY (uptrend)" if direction[-1] == 1 else "SELL (downtrend)"

    state = load_state()
    if state is None:
        send_telegram(
            "Bot Supertrend aktif.\n"
            "Simbol: {} | TF: {} | Faktor {} | ATR {}\n"
            "Arah saat ini: {}".format(SYMBOL, INTERVAL, FACTOR, ATR_LEN, now_side)
        )
        save_state({"last_flip_time": flip_time})
        print("State awal dibuat.")
        return

    if state.get("last_flip_time") != flip_time:
        send_telegram(
            "Supertrend {} di {} ({})\n"
            "Candle tutup: {}\n"
            "Harga tutup: {:.2f} | Level Supertrend: {:.2f}".format(
                side, SYMBOL, INTERVAL, wib(close_time), close[i], line[i]
            )
        )
        save_state({"last_flip_time": flip_time})
        print("Sinyal terkirim:", side)
    else:
        print("Tidak ada sinyal baru.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
