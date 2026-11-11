#!/usr/bin/env python3
"""
متابع الأسهم الأمريكية + محفظة وهمية  (بيانات Yahoo Finance)

التثبيت:  pip install streamlit yfinance pandas numpy requests
التشغيل:  streamlit run stock_tracker.py

ملاحظات:
- البيانات من Yahoo وقد تتأخر؛ هذا البرنامج للتجربة والتعلم وليس توصية استثمارية.
- المحفظة تُحفظ في ملف portfolio.json بجانب البرنامج.
- التداول الوهمي التلقائي يعمل فقط والتطبيق مفتوح (مع تفعيل التحديث التلقائي).
  عند فتحه من جديد يعيد فحص الشموع منذ الدخول فلا يضيع هدف/وقف تحقق أثناء غيابك
  (طالما البيانات ضمن آخر 5 أيام).
"""
import io
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

NY = "America/New_York"
URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]
CHUNK = 400
PF_FILE = "portfolio.json"
PRE, REG, POST = "قبل الافتتاح", "الجلسة الرسمية", "بعد الإغلاق"

AR = {
    "ticker": "الرمز", "session": "الجلسة", "price": "السعر", "prev_close": "إغلاق أمس",
    "chg": "التغير%", "ext_chg": "الممتد%", "gap": "الفجوة%", "vol": "الحجم",
    "rvol": "RVOL", "dollar_vol": "سيولة $", "surge": "تسارع الحجم", "atr_pct": "ATR%",
    "above_vwap": "فوق VWAP", "flow": "الاتجاه", "entry": "دخول فوق", "stop": "وقف",
    "target": "هدف", "support": "الدعم", "resist": "المقاومة", "score": "نقاط",
}
FMT = {"السعر": "{:.3f}", "إغلاق أمس": "{:.3f}", "التغير%": "{:+.1f}", "الممتد%": "{:+.1f}",
       "الفجوة%": "{:+.1f}", "الحجم": "{:,.0f}", "RVOL": "{:.1f}", "سيولة $": "{:,.0f}",
       "تسارع الحجم": "{:.1f}x", "ATR%": "{:.1f}", "دخول فوق": "{:.3f}", "وقف": "{:.3f}",
       "هدف": "{:.3f}", "الدعم": "{:.3f}", "المقاومة": "{:.3f}", "نقاط": "{:.0f}"}


# ───────────────────────── جلب البيانات ─────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_tickers():
    syms = []
    for u in URLS:
        try:
            txt = requests.get(u, timeout=30).text
            df = pd.read_csv(io.StringIO(txt), sep="|", dtype=str)
        except Exception:
            continue
        col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]
        if "ETF" in df.columns:
            df = df[df["ETF"] == "N"]
        syms += df[col].dropna().astype(str).tolist()
    return sorted({s for s in syms if s.isalpha() and len(s) <= 5})


def download(tickers, cb=None, **kw):
    """يرجع dict: ticker -> DataFrame"""
    out = {}
    for i in range(0, len(tickers), CHUNK):
        if cb:
            cb(i, len(tickers))
        chunk = tickers[i:i + CHUNK]
        try:
            data = yf.download(chunk, group_by="ticker", threads=True,
                               progress=False, auto_adjust=False, **kw)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        top = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in top:
                d = data[t].dropna(subset=["Close"])
                if not d.empty:
                    d["Volume"] = d["Volume"].fillna(0)
                    out[t] = d
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_universe(min_p, max_p, min_avg_vol, max_n, _cb=None):
    """فلترة يومية: السعر / السيولة (نطاق السعر موسّع لأن الجلسات الممتدة تحرّك السعر)."""
    daily = download(get_tickers(), cb=_cb, period="2mo", interval="1d")
    rows = []
    for t, d in daily.items():
        if len(d) < 15:
            continue
        last = d["Close"].iloc[-1]
        if not (min_p * 0.7 <= last <= max_p * 1.4):
            continue
        avg_vol = d["Volume"].iloc[-20:].mean()
        if avg_vol < min_avg_vol:
            continue
        atr = ((d["High"] - d["Low"]) / d["Close"]).iloc[-14:].mean() * 100
        rows.append(dict(ticker=t, avg_vol=avg_vol, atr_pct=atr,
                         support=d["Low"].iloc[-10:].min(), resist=d["High"].iloc[-10:].max()))
    u = pd.DataFrame(rows)
    if u.empty:
        return u
    u["rank"] = u["avg_vol"] * u["atr_pct"]
    return u.sort_values("rank", ascending=False).head(max_n).drop(columns="rank").reset_index(drop=True)


def session_of_minute(m):
    return PRE if m < 570 else REG if m < 960 else POST


def analyze(d):
    """تحليل شموع 5 دقائق (مع الجلسات الممتدة) لسهم واحد."""
    if len(d) < 8:
        return None
    idx = d.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert(NY)
    d = d.copy()
    d.index = idx
    dates = np.asarray(idx.date)
    mins = np.asarray(idx.hour) * 60 + np.asarray(idx.minute)
    reg = (mins >= 570) & (mins < 960)
    today_m = dates == dates[-1]
    prev_dates = sorted(set(dates[~today_m]))
    if not prev_dates:
        return None
    prev_reg = d[(dates == prev_dates[-1]) & reg]
    if prev_reg.empty:
        return None
    prev_close = prev_reg["Close"].iloc[-1]
    today = d[today_m]
    today_reg = d[today_m & reg]
    price = today["Close"].iloc[-1]
    session = session_of_minute(mins[-1])
    reg_close = today_reg["Close"].iloc[-1] if not today_reg.empty else prev_close
    chg = (price / prev_close - 1) * 100
    ext_chg = 0.0 if session == REG else (price / reg_close - 1) * 100
    gap = (today_reg["Open"].iloc[0] / prev_close - 1) * 100 if not today_reg.empty else chg
    vol = today["Volume"].sum()
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    vwap = (tp * today["Volume"]).sum() / max(vol, 1)
    v = d["Volume"].to_numpy()
    base = v[:-6].mean()
    surge = v[-6:].mean() / base if base > 0 else 0.0
    return dict(price=price, prev_close=prev_close, chg=chg, ext_chg=ext_chg, gap=gap, vol=vol,
                vwap=vwap, above_vwap=bool(price > vwap), high_t=today["High"].max(),
                surge=surge, session=session, ts=d.index[-1].tz_convert("UTC"))


@st.cache_data(ttl=90, show_spinner=False)
def snapshot(tickers: tuple):
    data = download(list(tickers), period="5d", interval="5m", prepost=True)
    rows = []
    for t, d in data.items():
        r = analyze(d)
        if r:
            r["ticker"] = t
            rows.append(r)
    return pd.DataFrame(rows)


def enrich(snap, uni, tp_pct):
    df = snap.merge(uni, on="ticker")
    df["rvol"] = df["vol"] / df["avg_vol"].replace(0, np.nan)
    df["rvol"] = df["rvol"].fillna(0)
    df["dollar_vol"] = df["price"] * df["vol"]
    h = df["high_t"].where(df["high_t"] > df["price"], np.inf)
    r = df["resist"].where(df["resist"] > df["price"], np.inf)
    trig = np.minimum(h, r)
    trig = trig.where(np.isfinite(trig), df["price"])
    df["entry"] = np.maximum(trig * 1.005, df["price"])
    df["target"] = df["entry"] * (1 + tp_pct / 100)
    df["score"] = (np.minimum(df["rvol"], 5) * 20 + np.minimum(df["atr_pct"], 20) * 2
                   + np.where(df["above_vwap"], 10, 0) + np.where(df["chg"] > 0, 10, 0)
                   + np.minimum(df["surge"], 5) * 4)
    df["flow"] = np.where((df["chg"] > 0) & df["above_vwap"], "تدفق شراء 🟢",
                          np.where(df["chg"] < 0, "ضغط بيع 🔴", "محايد"))
    return df


# ───────────────────────── العرض ─────────────────────────
def show(df, cols, stop_pct=None):
    d = df.copy()
    if stop_pct is not None:
        d["stop"] = d["entry"] * (1 - stop_pct / 100)
    d["above_vwap"] = d["above_vwap"].map({True: "نعم", False: "لا"})
    d = d[[c for c in cols if c in d.columns]].rename(columns=AR)
    sty = d.style.format({k: v for k, v in FMT.items() if k in d.columns}, na_rep="-")
    colored = [c for c in ("التغير%", "الممتد%", "الفجوة%") if c in d.columns]
    if colored:
        f = getattr(sty, "map", None) or sty.applymap
        sty = f(lambda v: "color:#16a34a;font-weight:600" if v > 0
                else ("color:#dc2626;font-weight:600" if v < 0 else ""), subset=colored)
    st.dataframe(sty, use_container_width=True, hide_index=True)


def market_label():
    now = pd.Timestamp.now(tz=NY)
    if now.weekday() >= 5:
        return "السوق مغلق (عطلة نهاية الأسبوع)", now
    m = now.hour * 60 + now.minute
    if m < 240 or m >= 1200:
        return "السوق مغلق", now
    return session_of_minute(m), now


# ───────────────────────── المحفظة الوهمية ─────────────────────────
def load_pf():
    if os.path.exists(PF_FILE):
        try:
            with open(PF_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_pf(pf):
    with open(PF_FILE, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=1, default=str)


def new_pf(cash):
    return {"initial": cash, "cash": cash, "positions": [], "history": [],
            "created": pd.Timestamp.now(tz="UTC").isoformat()}


def equity(pf):
    return pf["cash"] + sum(p["shares"] * p.get("last", p["entry"]) for p in pf["positions"])


def open_position(pf, r, cfg):
    """r: صف السهم (price, session, score, rvol, atr_pct, chg)."""
    entry = float(r["price"]) * (1 + cfg["slip"] / 100)
    alloc = min(pf["cash"], equity(pf) * cfg["size_pct"] / 100)
    shares = int(alloc // entry)
    if shares < 1:
        return False
    pf["cash"] -= shares * entry
    pf["positions"].append(dict(
        ticker=r["ticker"], shares=shares, entry=entry, cost=shares * entry, last=entry,
        target=entry * (1 + cfg["tp"] / 100), stop=entry * (1 - cfg["sl"] / 100), high=entry,
        opened=pd.Timestamp.now(tz="UTC").isoformat(), session=r["session"],
        score=float(r["score"]), rvol=float(r["rvol"]), atr=float(r["atr_pct"]), chg=float(r["chg"])))
    return True


def close_position(pf, p, price, reason, ts):
    proceeds = p["shares"] * price
    pf["cash"] += proceeds
    pf["history"].append(dict(
        ticker=p["ticker"], shares=p["shares"], entry=p["entry"], exit=price,
        pnl=proceeds - p["cost"], pnl_pct=(price / p["entry"] - 1) * 100, reason=reason,
        opened=p["opened"], closed=pd.Timestamp(ts).tz_convert("UTC").isoformat(),
        session=p["session"], score=p["score"], rvol=p["rvol"], atr=p["atr"], chg=p["chg"]))
    pf["positions"].remove(p)


def monitor(pf, cfg):
    """يفحص شموع 5 دقائق بعد وقت الدخول: وقف الخسارة أولاً (تحفّظاً) ثم الهدف."""
    if not pf["positions"]:
        return
    data = download([p["ticker"] for p in pf["positions"]], period="5d", interval="5m", prepost=True)
    for p in list(pf["positions"]):
        d = data.get(p["ticker"])
        if d is None:
            continue
        idx = d.index.tz_localize("UTC") if d.index.tz is None else d.index
        d = d.copy()
        d.index = idx
        p["last"] = float(d["Close"].iloc[-1])
        for ts, r in d[d.index > pd.Timestamp(p["opened"])].iterrows():
            if r["Open"] <= p["stop"]:
                close_position(pf, p, r["Open"] * (1 - cfg["slip"] / 100), "وقف خسارة (فجوة)", ts)
                break
            if r["Low"] <= p["stop"]:
                close_position(pf, p, p["stop"] * (1 - cfg["slip"] / 100), "وقف خسارة", ts)
                break
            if r["Open"] >= p["target"]:
                close_position(pf, p, r["Open"], "هدف (فجوة)", ts)
                break
            if r["High"] >= p["target"]:
                close_position(pf, p, p["target"], "هدف الربح", ts)
                break
            if cfg["trail"]:  # وقف متحرك: بعد نصف الهدف يلحق القمة
                p["high"] = max(p["high"], float(r["High"]))
                if p["high"] >= p["entry"] * (1 + cfg["tp"] / 200):
                    p["stop"] = max(p["stop"], p["high"] * (1 - cfg["sl"] / 100))


def auto_buy(pf, df, cfg):
    held = {p["ticker"] for p in pf["positions"]}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)
    recent = {h["ticker"] for h in pf["history"] if pd.Timestamp(h["closed"]) > cutoff}
    c = df[df["above_vwap"] & (df["chg"] > 0) & (df["chg"] <= cfg["max_chase"])
           & (df["rvol"] >= cfg["min_rvol"]) & (df["score"] >= cfg["min_score"])
           & df["session"].isin(cfg["sessions"]) & ~df["ticker"].isin(held | recent)]
    for _, r in c.sort_values("score", ascending=False).iterrows():
        if len(pf["positions"]) >= cfg["max_pos"]:
            break
        open_position(pf, r, cfg)


# ───────────────────────── التصميم (بطاقات) ─────────────────────────
CSS = """<style>
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1400px}
button[data-baseweb="tab"]{font-size:1rem;font-weight:700;padding:.6rem 1.1rem}
.stMarkdown p,.stCaption p,h1,h2,h3,h4,h5{direction:rtl;text-align:right}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;direction:rtl}
.card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:12px 14px;
      background:var(--secondary-background-color,rgba(128,128,128,.06))}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tk{font-size:1.25rem;font-weight:800;letter-spacing:.5px;direction:ltr;display:inline-block}
.px{font-size:1.15rem;font-weight:700;direction:ltr}
.badge{padding:2px 10px;border-radius:999px;font-weight:700;font-size:.95rem;direction:ltr;display:inline-block}
.up{background:rgba(22,163,74,.16);color:#16a34a}
.dn{background:rgba(220,38,38,.16);color:#dc2626}
.fl{background:rgba(128,128,128,.18)}
.chip{font-size:.72rem;padding:1px 8px;border-radius:999px;border:1px solid rgba(128,128,128,.4);opacity:.9}
.meta{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:8px;font-size:.82rem;opacity:.9}
.lv{display:flex;justify-content:space-between;margin-top:8px;font-size:.82rem;
    border-top:1px dashed rgba(128,128,128,.35);padding-top:6px;direction:ltr}
</style>"""


def kfmt(x):
    x = float(x)
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.1f}{u}"
    return f"{x:.0f}"


def cards(df, extra=None, sl=None):
    if df.empty:
        st.info("لا توجد بيانات مطابقة حالياً.")
        return
    html = ['<div class="grid">']
    for _, r in df.iterrows():
        cls = "up" if r["chg"] > 0 else "dn" if r["chg"] < 0 else "fl"
        ext = f'ممتد {r["ext_chg"]:+.1f}%' if r["session"] != REG else ""
        meta = (f'<span>RVOL <b>{r["rvol"]:.1f}</b></span><span>حجم <b>{kfmt(r["vol"])}</b></span>'
                f'<span>ATR <b>{r["atr_pct"]:.0f}%</b></span>'
                f'<span>VWAP <b>{"▲" if r["above_vwap"] else "▼"}</b></span>')
        if extra in ("flow", "plan"):
            meta += f'<span>تسارع <b>{r["surge"]:.1f}x</b></span><span>{r["flow"]}</span>'
        lv = ""
        if extra == "plan":
            stop = r["entry"] * (1 - sl / 100)
            lv = (f'<div class="lv"><span>دخول <b>{r["entry"]:.3f}</b></span>'
                  f'<span>وقف <b style="color:#dc2626">{stop:.3f}</b></span>'
                  f'<span>هدف <b style="color:#16a34a">{r["target"]:.3f}</b></span></div>')
        html.append(
            f'<div class="card"><div class="row"><div><span class="tk">{r["ticker"]}</span> '
            f'<span class="chip">{r["session"]}</span></div><div class="px">${r["price"]:.3f}</div></div>'
            f'<div class="row" style="margin-top:6px"><span class="badge {cls}">{r["chg"]:+.1f}%</span>'
            f'<span style="font-size:.8rem;opacity:.7">{ext}</span></div>'
            f'<div class="meta">{meta}</div>{lv}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render(df, cols, view, extra=None, sl=None):
    if view == "جدول":
        show(df, cols, stop_pct=sl if extra == "plan" else None)
    else:
        cards(df, extra, sl)


# ───────────────────────── الإعدادات (نموذج: لا يعيد التشغيل إلا عند «تطبيق») ─────────────────────────
DEFAULTS = dict(min_p=0.5, max_p=20.0, min_avg_vol=300_000, min_today_vol=50_000, min_atr=7.0,
                min_rvol_liq=2.0, max_n=1000, secs=120, tp=10.0, sl=5.0, trail=False, size_pct=20,
                max_pos=5, slip=0.3, auto_trade=False, sessions=[PRE, REG], min_score=100,
                min_rvol=1.5, max_chase=30)
SET_FILE = "settings.json"


def load_settings():
    s = dict(DEFAULTS)
    if os.path.exists(SET_FILE):
        try:
            with open(SET_FILE, encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception:
            pass
    return s


def render_settings():
    S = st.session_state.S
    st.caption("غيّر ما تريد ثم اضغط «تطبيق وحفظ» — لن يحدث أي تحديث قبل الضغط.")
    with st.form("settings_form"):
        st.markdown("##### 🔎 الفلاتر")
        a, b, c = st.columns(3)
        min_p = a.number_input("أدنى سعر $", 0.1, 1000.0, float(S["min_p"]), 0.1)
        max_p = b.number_input("أعلى سعر $", 0.2, 1000.0, float(S["max_p"]), 0.5)
        min_avg_vol = c.number_input("أدنى متوسط حجم يومي", 0, 50_000_000, int(S["min_avg_vol"]), 50_000)
        a, b, c = st.columns(3)
        min_today_vol = a.number_input("أدنى حجم اليوم", 0, 50_000_000, int(S["min_today_vol"]), 10_000)
        min_atr = b.slider("أدنى تذبذب ATR%", 1.0, 30.0, float(S["min_atr"]), 0.5)
        min_rvol_liq = c.slider("RVOL لتبويب السيولة", 1.0, 10.0, float(S["min_rvol_liq"]), 0.5)
        a, b = st.columns(2)
        max_n = a.slider("أقصى عدد رموز للمراقبة", 200, 3000, int(S["max_n"]), 100)
        secs = b.slider("التحديث التلقائي كل (ثانية)", 60, 600, int(S["secs"]), 30)

        st.markdown("##### 💼 المحفظة الوهمية")
        a, b, c = st.columns(3)
        tp = a.number_input("هدف الربح %", 1.0, 100.0, float(S["tp"]), 0.5)
        sl = b.number_input("وقف الخسارة %", 0.5, 50.0, float(S["sl"]), 0.5)
        slip = c.number_input("انزلاق سعري %", 0.0, 3.0, float(S["slip"]), 0.1)
        a, b = st.columns(2)
        size_pct = a.slider("حجم الصفقة % من المحفظة", 5, 100, int(S["size_pct"]), 5)
        max_pos = b.slider("أقصى عدد صفقات مفتوحة", 1, 20, int(S["max_pos"]))
        a, b, c = st.columns(3)
        min_score = a.slider("أدنى نقاط للدخول", 0, 200, int(S["min_score"]), 5)
        min_rvol = b.slider("أدنى RVOL للدخول", 0.5, 10.0, float(S["min_rvol"]), 0.5)
        max_chase = c.slider("أقصى ارتفاع عند الدخول %", 5, 200, int(S["max_chase"]), 5)
        sessions = st.multiselect("الدخول مسموح في", [PRE, REG, POST],
                                  default=[x for x in S["sessions"] if x in (PRE, REG, POST)])
        a, b = st.columns(2)
        trail = a.checkbox("وقف متحرك (بعد نصف الهدف)", value=bool(S["trail"]))
        auto_trade = b.checkbox("تداول وهمي تلقائي", value=bool(S["auto_trade"]))
        ok = st.form_submit_button("✅ تطبيق وحفظ", type="primary", use_container_width=True)
    if ok:
        new = dict(min_p=min_p, max_p=max_p, min_avg_vol=min_avg_vol, min_today_vol=min_today_vol,
                   min_atr=min_atr, min_rvol_liq=min_rvol_liq, max_n=max_n, secs=secs, tp=tp, sl=sl,
                   trail=trail, size_pct=size_pct, max_pos=max_pos, slip=slip, auto_trade=auto_trade,
                   sessions=sessions, min_score=min_score, min_rvol=min_rvol, max_chase=max_chase)
        st.session_state.S = new
        try:
            with open(SET_FILE, "w", encoding="utf-8") as f:
                json.dump(new, f, ensure_ascii=False)
        except Exception:
            pass
        snapshot.clear()
        st.rerun()


# ───────────────────────── الواجهة ─────────────────────────
st.set_page_config(page_title="متابع الأسهم الأمريكية", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
if "S" not in st.session_state:
    st.session_state.S = load_settings()
S = st.session_state.S
cfg = dict(tp=S["tp"], sl=S["sl"], trail=S["trail"], size_pct=S["size_pct"], max_pos=S["max_pos"],
           slip=S["slip"], sessions=S["sessions"], min_score=S["min_score"],
           min_rvol=S["min_rvol"], max_chase=S["max_chase"])
tp, sl = S["tp"], S["sl"]

label, now_ny = market_label()
h1, h2 = st.columns([3, 2])
with h1:
    st.markdown("## 📈 متابع الأسهم الأمريكية")
    st.caption(f"الحالة: **{label}** — نيويورك {now_ny:%Y-%m-%d %H:%M} — بيانات Yahoo قد تتأخر وليست توصية.")
with h2:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    view = c1.radio("العرض", ["بطاقات", "جدول"], horizontal=True, label_visibility="collapsed")
    c2.toggle("تلقائي", key="auto_refresh")
    if c3.button("🔄 تحديث", use_container_width=True):
        snapshot.clear()

prog = st.progress(0.0, text="تجهيز قائمة الأسهم...")


def _cb(done, total):
    prog.progress(min(done / max(total, 1), 1.0), text=f"تحميل قائمة الأسهم (أول مرة فقط، عدة دقائق)... {done}/{total}")


uni = build_universe(S["min_p"], S["max_p"], S["min_avg_vol"], S["max_n"], _cb=_cb)
prog.empty()
if uni.empty:
    st.warning("لا توجد أسهم مطابقة. خفّف الفلاتر من الإعدادات.")
    with st.expander("⚙️ الإعدادات", expanded=True):
        render_settings()
    st.stop()

with st.spinner(f"جلب الأسعار اللحظية لـ {len(uni)} سهم..."):
    snap = snapshot(tuple(uni["ticker"]))
if snap.empty:
    st.error("لم تصل بيانات لحظية (السوق مغلق أو Yahoo حجب الطلبات مؤقتاً). حاول لاحقاً.")
    with st.expander("⚙️ الإعدادات", expanded=True):
        render_settings()
    st.stop()

df = enrich(snap, uni, tp)
fresh = (pd.Timestamp.now(tz="UTC") - df["ts"].max()).total_seconds() < 1800
base = df[df["price"].between(S["min_p"], S["max_p"]) & (df["vol"] >= S["min_today_vol"])]

pf = load_pf()
if pf:
    monitor(pf, cfg)
    if S["auto_trade"] and fresh:
        auto_buy(pf, base, cfg)
    save_pf(pf)

t1, t6, t2, t3, t4, t5 = st.tabs(["🚀 الحركة", "🎯 القناص", "💧 السيولة", "⚡ المضاربة", "💼 المحفظة", "⚙️ الإعدادات"])

with t1:
    by = st.radio("الترتيب حسب", ["التغير عن إغلاق أمس", "الجلسة الممتدة فقط"], horizontal=True)
    pool, key = (base, "chg") if by.startswith("التغير") else (base[base["session"] != REG], "ext_chg")
    cols = ["ticker", "session", "price", "prev_close", "chg", "ext_chg", "gap", "vol", "rvol", "above_vwap"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🟢 أعلى 10 ارتفاعاً")
        render(pool.sort_values(key, ascending=False).head(10), cols, view)
    with c2:
        st.markdown("#### 🔴 أعلى 10 انخفاضاً")
        render(pool.sort_values(key).head(10), cols, view)

with t6:
    st.caption("أسهم قريبة من اختراق أعلى سعر اليوم، فوق VWAP، بسيولة متسارعة. ترشيحات فنية للتجربة وليست ضماناً للربح.")
    sn = base.copy()
    sn["dist_high"] = (sn["high_t"] / sn["price"] - 1) * 100
    sn = sn[sn["above_vwap"] & (sn["chg"] > 1) & (sn["chg"] <= S["max_chase"])
            & (sn["rvol"] >= 2) & (sn["surge"] >= 1.5) & (sn["dist_high"] <= 1.5)].copy()
    if sn.empty:
        st.info("لا توجد فرص مطابقة الآن. الشروط: فوق VWAP، RVOL ≥ 2، تسارع ≥ 1.5x، وعلى بعد ≤ 1.5% من أعلى سعر اليوم.")
    else:
        sn["snipe"] = (np.minimum(sn["rvol"], 5) * 15 + np.minimum(sn["surge"], 5) * 8
                       + np.maximum(0, 20 - sn["dist_high"] * 8) + np.minimum(sn["atr_pct"], 20) * 1.5)
        sn = sn.sort_values("snipe", ascending=False).head(10)
        sn["flow"] = np.where(sn["dist_high"] <= 0.2, "🔥 اختراق الآن", "👀 قريب من الاختراق")
        render(sn, ["ticker", "session", "price", "chg", "rvol", "surge", "atr_pct", "flow",
                    "entry", "stop", "target"], view, extra="plan", sl=sl)
        pf_s = load_pf()
        if pf_s is None:
            st.caption("أنشئ محفظة وهمية من تبويب المحفظة لتجربة الدخول.")
        else:
            c1, c2 = st.columns([2, 1])
            pick_s = c1.selectbox("اقتناص بالمحفظة الوهمية", list(sn["ticker"]), key="snipe_pick")
            if c2.button("🎯 شراء وهمي", key="snipe_buy", use_container_width=True):
                if open_position(pf_s, sn[sn["ticker"] == pick_s].iloc[0], cfg):
                    save_pf(pf_s)
                    st.rerun()
                else:
                    st.error("الرصيد لا يكفي.")

with t2:
    st.caption("RVOL = حجم اليوم ÷ المتوسط اليومي. التسارع = حجم آخر 30 دقيقة ÷ المعتاد.")
    liq = base[(base["rvol"] >= S["min_rvol_liq"]) | (base["surge"] >= 3)].sort_values("rvol", ascending=False).head(24)
    render(liq, ["ticker", "session", "price", "chg", "ext_chg", "vol", "rvol", "surge", "dollar_vol", "above_vwap", "flow"],
           view, extra="flow")

with t3:
    st.caption("أسهم متذبذبة وسيولتها كافية. «دخول» = اختراق أعلى سعر اليوم أو المقاومة.")
    vol_df = base[base["atr_pct"] >= S["min_atr"]].sort_values("score", ascending=False).head(24)
    render(vol_df, ["ticker", "session", "price", "chg", "rvol", "atr_pct", "above_vwap",
                    "entry", "stop", "target", "support", "resist", "score"], view, extra="plan", sl=sl)

with t4:
    if pf is None:
        st.markdown("#### إنشاء محفظة وهمية")
        amt = st.number_input("مبلغ المحفظة ($)", 100.0, 10_000_000.0, 10_000.0, 500.0)
        if st.button("إنشاء المحفظة", type="primary"):
            save_pf(new_pf(amt))
            st.rerun()
    else:
        eq = equity(pf)
        hist = pd.DataFrame(pf["history"])
        realized = hist["pnl"].sum() if not hist.empty else 0.0
        m = st.columns(4)
        m[0].metric("قيمة المحفظة", f"${eq:,.2f}", f"{(eq / pf['initial'] - 1) * 100:+.2f}%")
        m[1].metric("النقد المتاح", f"${pf['cash']:,.2f}")
        m[2].metric("ربح/خسارة محققة", f"${realized:,.2f}")
        m[3].metric("صفقات مفتوحة", len(pf["positions"]))
        if S["auto_trade"] and not fresh:
            st.warning("البيانات قديمة (السوق مغلق غالباً) — الدخول التلقائي متوقف.")

        st.markdown("#### الصفقات المفتوحة")
        if pf["positions"]:
            op = pd.DataFrame(pf["positions"])
            op["pnl_pct"] = (op["last"] / op["entry"] - 1) * 100
            op["pnl"] = (op["last"] - op["entry"]) * op["shares"]
            st.dataframe(op[["ticker", "shares", "entry", "last", "pnl_pct", "pnl", "target", "stop", "session"]].rename(
                columns={"ticker": "الرمز", "shares": "الكمية", "entry": "الدخول", "last": "آخر سعر",
                         "pnl_pct": "ربح%", "pnl": "ربح $", "target": "الهدف", "stop": "الوقف",
                         "session": "جلسة الدخول"}).round(3), use_container_width=True, hide_index=True)
        else:
            st.caption("لا توجد صفقات مفتوحة.")

        with st.expander("شراء/بيع يدوي"):
            pick = st.selectbox("اختر سهماً", sorted(base["ticker"]))
            if st.button("شراء بالسعر الحالي"):
                if open_position(pf, base[base["ticker"] == pick].iloc[0], cfg):
                    save_pf(pf)
                    st.rerun()
                else:
                    st.error("الرصيد لا يكفي.")
            held = [p["ticker"] for p in pf["positions"]]
            if held:
                sell = st.selectbox("إغلاق صفقة", held)
                if st.button("بيع الآن"):
                    p = next(x for x in pf["positions"] if x["ticker"] == sell)
                    close_position(pf, p, p["last"] * (1 - S["slip"] / 100), "إغلاق يدوي", pd.Timestamp.now(tz="UTC"))
                    save_pf(pf)
                    st.rerun()

        st.markdown("#### 📊 فاعلية المؤشرات")
        if hist.empty:
            st.caption("لا صفقات مغلقة بعد.")
        else:
            hist["win"] = hist["pnl"] > 0
            k = st.columns(3)
            k[0].metric("نسبة النجاح", f"{hist['win'].mean() * 100:.0f}%")
            k[1].metric("متوسط الصفقة", f"{hist['pnl_pct'].mean():+.2f}%")
            k[2].metric("عدد الصفقات", len(hist))
            hist["شريحة RVOL"] = pd.cut(hist["rvol"], [0, 1.5, 3, 5, np.inf], labels=["<1.5", "1.5-3", "3-5", ">5"])
            for name, col in (("حسب الجلسة", "session"), ("حسب سبب الخروج", "reason"), ("حسب RVOL", "شريحة RVOL")):
                g = hist.groupby(col, observed=True).agg(صفقات=("pnl", "size"), نجاح=("win", "mean"),
                                                         متوسط_ربح=("pnl_pct", "mean"), إجمالي=("pnl", "sum"))
                g["نجاح"] = (g["نجاح"] * 100).round(0)
                st.markdown(f"**{name}**")
                st.dataframe(g.round(2), use_container_width=True)
            st.markdown("#### سجل الصفقات")
            st.dataframe(hist.drop(columns=["win", "شريحة RVOL"]).round(3).iloc[::-1],
                         use_container_width=True, hide_index=True)

        if st.button("🗑️ حذف المحفظة وبدء جديدة"):
            os.remove(PF_FILE)
            st.rerun()

with t5:
    render_settings()

if st.session_state.get("auto_refresh"):
    time.sleep(S["secs"])
    st.rerun()
