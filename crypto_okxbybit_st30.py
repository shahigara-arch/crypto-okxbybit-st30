# crypto_okxbybit_st30.py — OKX/Bybit 30m ST+MACD with Hidden Strength Add-ons
import os, time, html, requests, datetime as dt, sys, traceback
import numpy as np
import pandas as pd

# ---------- Config ----------
TOP_N = 100
INTERVAL_MIN = 30
MTF_INTERVAL = "4H"   # OKX format; Bybit uses 240 minutes
MIN_PRICE = 0.05
MIN_QVOL = 10_000_000.0   # 24h quote volume USD (lower if too strict)
TIMEOUT = 20
RETRIES = 3
THROTTLE = 0.03
SRC = os.environ.get("EX_SOURCE", "okx").strip().lower()   # okx | bybit
TV_PREFIX = "OKX" if SRC == "okx" else "BYBIT"

# Ultra thresholds
ULTRA_MIN_RVOL = 2.0
ULTRA_MIN_ADX  = 20.0
ULTRA_BO_TOL   = 0.005  # within 0.5% of 55-bar high/low
ULTRA_OB_MIN   = 0.55   # orderbook bias (BUY) or 0.45 inverted for SELL

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"})

def log(*a): print(dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "-", *a, flush=True)

def get_json(url, params=None, retries=RETRIES):
    for i in range(retries):
        try:
            r = S.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            log("HTTP", r.status_code, url, "Body:", (r.text or "")[:180])
            time.sleep(1+i)
        except Exception as e:
            log("GET err:", url, e)
            time.sleep(1+i)
    raise RuntimeError(f"GET failed: {url}")

# ---------- TA ----------
def ema(a, length):
    a = np.asarray(a, dtype=float)
    if len(a) == 0: return np.array([])
    alpha = 2.0/(length+1.0)
    out = np.empty_like(a); out[:] = np.nan
    out[0] = a[0]
    for i in range(1, len(a)): out[i] = alpha*a[i] + (1-alpha)*out[i-1]
    return out

def rma(x, length):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) < length: return out
    out[length-1] = np.mean(x[:length])
    for i in range(length, len(x)):
        out[i] = (out[i-1]*(length-1) + x[i]) / length
    return out

def supertrend(h, l, c, period=18, mult=1.5):
    h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); c = np.asarray(c, dtype=float)
    n = len(c)
    if n == 0: return np.array([]), np.array([])
    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    atr = rma(tr, period)
    hl2 = (h + l) / 2.0
    upper = hl2 + mult*atr; lower = hl2 - mult*atr
    fub = np.copy(upper); flb = np.copy(lower)
    for i in range(1, n):
        fub[i] = upper[i] if (upper[i] < fub[i-1]) or (c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lower[i] if (lower[i] > flb[i-1]) or (c[i-1] < flb[i-1]) else flb[i-1]
    trend = np.ones(n, dtype=int); st = np.full(n, np.nan)
    trend[0] = 1; st[0] = flb[0]
    for i in range(1, n):
        if trend[i-1] == 1:
            if c[i] <= fub[i]: trend[i] = -1; st[i] = fub[i]
            else:              trend[i] =  1; st[i] = flb[i]
        else:
            if c[i] >= flb[i]: trend[i] =  1; st[i] = flb[i]
            else:              trend[i] = -1; st[i] = fub[i]
    return st, trend

def macd(c, fast=12, slow=26, signal=9):
    c = np.asarray(c, dtype=float)
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

def adx(h, l, c, length=14):
    h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); c = np.asarray(c, dtype=float)
    if len(c) < length+2: return np.full(len(c), np.nan)
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plusDM = np.where((up > dn) & (up > 0), up, 0.0)
    minusDM = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_close = c[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)])
    atr = rma(tr, length)
    pDI = 100 * rma(plusDM, length) / atr
    mDI = 100 * rma(minusDM, length) / atr
    dx = 100 * np.abs(pDI - mDI) / (pDI + mDI)
    adxv = rma(dx, length)
    adxv = np.concatenate(([np.nan], adxv))
    return adxv

def norm01(x, lo, hi):
    if x is None or np.isnan(x): return 0.0
    if hi == lo: return 0.0
    return float(max(0.0, min(1.0, (x - lo)/(hi - lo))))

# ---------- OKX ----------
def okx_top_usdt(top_n=100):
    j = get_json("https://www.okx.com/api/v5/market/tickers", params={"instType":"SPOT"})
    arr = j.get("data") or []
    rows = []
    for d in arr:
        inst = d.get("instId","")  # e.g., BTC-USDT
        if not inst.endswith("-USDT"): continue
        if any(x in inst for x in ["3L","3S","5L","5S","2L","2S"]): continue
        qv = float(d.get("volCcy24h") or 0.0)
        last = float(d.get("last") or 0.0)
        if last < MIN_PRICE or qv < MIN_QVOL: continue
        rows.append((inst, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    syms = [s for s,_ in rows[:top_n]]
    qmap = {s:q for s,q in rows}
    log("OKX top:", syms[:10], "…", len(syms))
    return syms, qmap

def okx_klines(inst, bar="30m", limit=300):
    j = get_json("https://www.okx.com/api/v5/market/candles", params={"instId":inst, "bar":bar, "limit":limit})
    arr = j.get("data") or []
    df = pd.DataFrame(arr, columns=["ts","open","high","low","close","vol","volCcy","volQuote"])
    if df.empty: return df
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    df["open_time_dt"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df = df.sort_values("open_time_dt").reset_index(drop=True)
    # OKX candles are end times descending; but we sorted by open
    return df

def okx_depth(inst, sz=20):
    j = get_json("https://www.okx.com/api/v5/market/books", params={"instId":inst, "sz":sz})
    data = (j.get("data") or [])
    if not data: return None
    dd = data[0]
    bids = dd.get("bids") or []
    asks = dd.get("asks") or []
    def qsum(levels):
        s = 0.0
        for p,q,_ in levels:
            s += float(p)*float(q)
        return s
    bq = qsum(bids); aq = qsum(asks)
    if bq+aq == 0: return None
    imb = (bq - aq) / (bq + aq)  # [-1..1]
    return imb

def okx_swap_metrics(inst_spot):
    # Convert BTC-USDT -> BTC-USDT-SWAP
    base = inst_spot.split("-")[0]
    inst = f"{base}-USDT-SWAP"
    # OI
    oi = None; fund = None
    try:
        j1 = get_json("https://www.okx.com/api/v5/public/open-interest", params={"instType":"SWAP", "instId":inst})
        d1 = (j1.get("data") or [])
        if d1:
            oi = float(d1[0].get("oi") or 0.0)
    except Exception: pass
    try:
        j2 = get_json("https://www.okx.com/api/v5/public/funding-rate", params={"instId":inst})
        d2 = (j2.get("data") or [])
        if d2:
            fund = float(d2[0].get("fundingRate") or 0.0)  # typically 0.0001
    except Exception: pass
    return oi, fund

# ---------- Bybit ----------
def bybit_top_usdt(top_n=100):
    j = get_json("https://api.bybit.com/v5/market/tickers", params={"category":"spot"})
    arr = (j.get("result") or {}).get("list") or []
    rows = []
    for d in arr:
        sym = d.get("symbol","")  # e.g., BTCUSDT
        if not sym.endswith("USDT"): continue
        if sym.endswith(("3LUSDT","3SUSDT","5LUSDT","5SUSDT","2LUSDT","2SUSDT")): continue
        qv = float(d.get("turnover24h") or 0.0)
        last = float(d.get("lastPrice") or 0.0)
        if last < MIN_PRICE or qv < MIN_QVOL: continue
        rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    syms = [s for s,_ in rows[:top_n]]
    qmap = {s:q for s,q in rows}
    log("Bybit top:", syms[:10], "…", len(syms))
    return syms, qmap

def bybit_klines(symbol, minutes=30, limit=300):
    interval = str(minutes)
    j = get_json("https://api.bybit.com/v5/market/kline", params={"category":"spot","symbol":symbol,"interval":interval,"limit":limit})
    arr = ((j.get("result") or {}).get("list") or [])
    # list: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(arr, columns=["start","open","high","low","close","volume","turnover"])
    if df.empty: return df
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df["open_time_dt"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True)
    df = df.sort_values("open_time_dt").reset_index(drop=True)
    return df

def bybit_depth(symbol, limit=50):
    j = get_json("https://api.bybit.com/v5/market/orderbook", params={"category":"spot","symbol":symbol,"limit":limit})
    d = (j.get("result") or {})
    bids = d.get("b") or []; asks = d.get("a") or []
    def qsum(levels):
        s = 0.0
        for p,q in levels:
            s += float(p)*float(q)
        return s
    bq = qsum(bids); aq = qsum(asks)
    if bq+aq == 0: return None
    return (bq - aq)/(bq + aq)

def bybit_linear_metrics(symbol_spot):
    # spot BTCUSDT -> linear perp BTCUSDT
    oi = None; fund = None
    try:
        j = get_json("https://api.bybit.com/v5/market/tickers", params={"category":"linear","symbol":symbol_spot})
        arr = (j.get("result") or {}).get("list") or []
        if arr:
            d = arr[0]
            oi = float(d.get("openInterest") or 0.0)
            fund = float(d.get("fundingRate") or 0.0)
    except Exception: pass
    return oi, fund

# ---------- News (CryptoPanic, optional) ----------
def news_score(base):
    token = os.environ.get("CRYPTOPANIC_TOKEN","" ).strip()
    if not token: return None
    try:
        j = get_json("https://cryptopanic.com/api/v1/posts/", params={
            "auth_token": token,
            "currencies": base,
            "kind": "news",
            "filter": "rising",
            "public": "true",
        })
        results = j.get("results") or []
        pos = neg = 0
        for r in results[:20]:
            votes = r.get("votes") or {}
            pos += int(votes.get("positive",0))
            neg += int(votes.get("negative",0))
        if pos+neg == 0: return 0.0
        score = (pos - neg) / max(1, pos + neg)  # -1..+1
        return max(0.0, min(1.0, 0.5 + 0.5*score))  # 0..1
    except Exception:
        return None

# ---------- Helpers ----------
def tv_link(sym):
    return f"https://www.tradingview.com/chart/?symbol={TV_PREFIX}:{sym.replace('-', '')}&interval={INTERVAL_MIN}"

def scan_symbol(sym, qmap):
    # 30m candles
    if SRC == "okx":
        df = okx_klines(sym, bar=f"{INTERVAL_MIN}m", limit=400)
    else:
        df = bybit_klines(sym, minutes=INTERVAL_MIN, limit=400)
    if df is None or df.empty or len(df) < 120: return None
    i = len(df) - 2  # last closed bar
    close = df["close"].values; high = df["high"].values; low = df["low"].values; vol = df["volume"].values

    st, tr = supertrend(high, low, close, 18, 1.5)
    m, s, _ = macd(close, 12, 26, 9)
    adx30 = adx(high, low, close, 14)

    st_buy  = (tr[i] == 1 and tr[i-1] == -1)
    st_sell = (tr[i] == -1 and tr[i-1] == 1)
    macd_up = (m[i-1] <= s
