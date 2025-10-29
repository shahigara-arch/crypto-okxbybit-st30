def scan_symbol(sym, qmap):
    # 30m candles
    if SRC == "okx":
        df = okx_klines(sym, bar=f"{INTERVAL_MIN}m", limit=400)
    else:
        df = bybit_klines(sym, minutes=INTERVAL_MIN, limit=400)
    if df is None or df.empty or len(df) < 120:
        return None

    i = len(df) - 2  # last closed bar
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    vol   = df["volume"].values

    # Signals
    st, tr = supertrend(high, low, close, 18, 1.5)
    m, s, _ = macd(close, 12, 26, 9)
    adx30   = adx(high, low, close, 14)

    st_buy  = (tr[i] == 1 and tr[i-1] == -1)
    st_sell = (tr[i] == -1 and tr[i-1] == 1)
    macd_up = (m[i-1] <= s[i-1]) and (m[i] > s[i])
    macd_dn = (m[i-1] >= s[i-1]) and (m[i] < s[i])

    if not ((st_buy and macd_up) or (st_sell and macd_dn)):
        return None

    # Relative volume (20-bars)
    rv = np.nan
    if i >= 21:
        avg20 = np.mean(vol[i-20:i])
        rv = vol[i] / avg20 if avg20 > 0 else np.nan

    # Breakout proximity (55 bars)
    hh = np.max(close[max(0, i-55):i+1])
    ll = np.min(close[max(0, i-55):i+1])
    if st_buy:
        bo = 1.0 - max(0.0, (hh - close[i]) / (0.01 * close[i]))  # within 1% -> near 1
    else:
        bo = 1.0 - max(0.0, (close[i] - ll) / (0.01 * close[i]))
    bo = max(0.0, min(1.0, bo))

    # Candle quality
    rng = max(1e-12, high[i] - low[i])
    near_high = (close[i] - low[i]) / rng
    near_low  = (high[i] - close[i]) / rng
    cndl = near_high if st_buy else near_low
    cndl = max(0.0, min(1.0, cndl))

    # MTF 4h trend
    if SRC == "okx":
        df4 = okx_klines(sym, bar="4H", limit=300)
    else:
        df4 = bybit_klines(sym, minutes=240, limit=300)
    c4 = df4["close"].values if not df4.empty else np.array([])
    mtf = False
    if len(c4) >= 200:
        ema50 = ema(c4, 50)
        ema200 = ema(c4, 200)
        m4, s4, h4 = macd(c4, 12, 26, 9)
        j = len(c4) - 1
        if st_buy:
            mtf = (c4[j] > ema50[j] > ema200[j]) and (h4[j] > 0)
        else:
            mtf = (c4[j] < ema50[j] < ema200[j]) and (h4[j] < 0)

    # Orderbook imbalance
    try:
        ob_imb = okx_depth(sym, 20) if SRC == "okx" else bybit_depth(sym, 50)
    except Exception:
        ob_imb = None
    if ob_imb is None:
        ob_sig = 0.5
    else:
        ob_sig = ob_imb if st_buy else -ob_imb
        ob_sig = max(-1.0, min(1.0, ob_sig))
        ob_sig = 0.5 + 0.5 * ob_sig  # 0..1

    # Futures confluence (SWAP/linear)
    oi = None
    fund = None
    try:
        oi, fund = (okx_swap_metrics(sym) if SRC == "okx" else bybit_linear_metrics(sym.replace("-", "")))
    except Exception:
        pass

    # Use open_time of last closed as bar id (OKX/Bybit both have open_time_dt)
    bar_close_ms = int(df.iloc[i]["open_time_dt"].value / 1e6) if "open_time_dt" in df else 0

    return {
        "symbol": sym,
        "side": "BUY" if st_buy else "SELL",
        "price": float(close[i]),
        "rvol": float(rv) if rv == rv else None,
        "adx":  float(adx30[i]) if adx30[i] == adx30[i] else None,
        "bo":   bo,
        "cndl": cndl,
        "mtf":  bool(mtf),
        "ob":   float(ob_sig) if ob_sig is not None else 0.5,
        "qvol": float(qmap.get(sym, 0.0)),
        "oi":   oi,
        "fund": fund,
        "bar_close_ms": bar_close_ms,
    }
