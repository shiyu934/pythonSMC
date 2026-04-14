import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import ccxt  # <--- 資工系處理及時行情的標配

# --- 1. 核心配置 ---
st.set_page_config(page_title="BTC 15D 實時戰術中心", layout="wide", initial_sidebar_state="expanded")

# 深色科技 UI 補丁
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 12px; border-radius: 8px; border: 1px solid #444; }
    [data-testid="stSidebar"] { background-color: #161a24; }
    </style>
    """, unsafe_allow_html=True)


# --- 2. 交易所 Socket 直連 (解決「不及時」問題) ---
@st.cache_resource  # 確保連線只會初始化一次，不會隨刷新斷線
def get_binance_ticker():
    return ccxt.binance({'enableRateLimit': True})


def fetch_live_price():
    try:
        exchange = get_binance_ticker()
        ticker = exchange.fetch_ticker('BTC/USDT')
        return float(ticker['last'])
    except:
        return 0.0


# --- 3. 資料庫讀取 (用於 PnL 與 K 線) ---
def get_db_data(table="experiences", limit=500):
    try:
        conn = sqlite3.connect("wisdom_book.db", timeout=10, check_same_thread=False)
        # 開啟 WAL 模式，讓讀取不被 main.py 的寫入卡住
        conn.execute("PRAGMA journal_mode=WAL;")
        sort_col = "timestamp" if table == "experiences" else "ts"
        query = f"SELECT * FROM {table} ORDER BY {sort_col} DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()


# --- 4. 側邊欄：實時行情燈號 ---
st.sidebar.title("🧭 戰略監控中心")

# 直接從交易所 Socket 拿價格
socket_price = fetch_live_price()

# 從資料庫拿 SMC 戰略數據 (VAH/VAL)
df_all = get_db_data("experiences", 500)
df_kline = get_db_data("ohlcv", 100)

if socket_price > 0:
    st.sidebar.metric("即時 BTC 市價 (Socket)", f"${socket_price:,.1f}")
else:
    st.sidebar.warning("⚠️ 交易所連線中...")

if not df_all.empty:
    last_rec = df_all.iloc[0]
    vah, val = float(last_rec.get('vah', 0)), float(last_rec.get('val', 0))
    fvg = last_rec.get('fvg', 0)

    # 🚨 紅綠燈邏輯：改用絕對即時的 socket_price 來判定
    st.sidebar.subheader("🚨 實時戰略燈號")
    l, t, c = "⚪", "市場中性", "#262730"
    if fvg > 0:
        l, t, c = "🔵", "FVG 向上磁吸", "#0083b0"
    elif fvg < 0:
        l, t, c = "🟠", "FVG 向下磁吸", "#e67e22"
    elif socket_price >= vah and vah > 0:
        l, t, c = "🔴", "壓力區 (VAH)", "#ef5350"
    elif socket_price <= val and val > 0:
        l, t, c = "🟢", "支撐區 (VAL)", "#26a69a"

    st.sidebar.markdown(f"""
        <div style="background-color:{c}; padding:15px; border-radius:12px; text-align:center;">
            <h1 style="margin:0;">{l}</h1><b style="color:white; font-size:16px;">{t}</b>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.write(f"🟡 POC: {last_rec.get('poc_price', 0):.1f}")
    st.sidebar.write(f"🟦 VAH/VAL: {vah:.1f} / {val:.1f}")

st.sidebar.write(f"系統同步: {datetime.now().strftime('%H:%M:%S')}")

# --- 5. 戰士排行榜 (總報酬) ---
st.title("🚀 BTC 15-Dimensional Army Dashboard")
if not df_all.empty:
    leaderboard = []
    for i in range(16):
        a_df = df_all[df_all['agent_id'] == str(i)]
        tot = a_df['pnl'].iloc[0] if not a_df.empty else 0.0
        leaderboard.append({'Agent': f"A{i}", 'PnL': tot})

    df_rank = pd.DataFrame(leaderboard).sort_values(by='PnL', ascending=False)
    st.plotly_chart(px.bar(df_rank, x='Agent', y='PnL', color='PnL',
                           color_continuous_scale='RdYlGn', template="plotly_dark", height=220),
                    use_container_width=True)

# --- 6. K 線與 SMC 戰略圖 ---
if not df_kline.empty:
    df_kline['dt'] = pd.to_datetime(df_kline['ts'], unit='ms')
    df_plot = df_kline.sort_values('ts')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df_plot['dt'], open=df_plot['open'], high=df_plot['high'],
                                 low=df_plot['low'], close=df_plot['close'], name="BTC"), row=1, col=1)

    # 畫出價值區區間
    if not df_all.empty:
        l = df_all.iloc[0]
        fig.add_hrect(y0=float(l.get('val', 0)), y1=float(l.get('vah', 0)),
                      fillcolor="rgba(41, 98, 255, 0.1)", line_width=0, row=1, col=1)

    vol_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df_plot['open'], df_plot['close'])]
    fig.add_trace(go.Bar(x=df_plot['dt'], y=df_plot['volume'], marker_color=vol_colors), row=2, col=1)
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 7. 16 位戰士實時面板 (數據分家修正) ---
st.divider()
if not df_all.empty:
    st.subheader("🛡️ 軍團戰士實時報酬 (雙維度對齊)")
    status_list = []
    for i in range(16):
        latest = df_all[df_all['agent_id'] == str(i)].head(1)
        if not latest.empty:
            row = latest.iloc[0]
            action = int(row['action'])
            # reward = 當前單子, pnl = 生涯總報酬
            cur_pnl = row.get('cur_pnl', 0.0)  # 【修正】直接讀 cur_pnl，不用 reward 替代

            status_list.append({
                "編號": f"Agent {i}",
                "動作": action,
                "進場價": row.get('entry', 0.0),   # 【修正】DB 欄位是 entry，不是 entry_price
                "最新價": socket_price if socket_price > 0 else 0.0,
                "當前單子 (%)": cur_pnl,
                "生涯總計 (%)": row.get('pnl', 0.0),
                "更新時間": str(row['timestamp']).split(" ")[-1]
            })

    status_df = pd.DataFrame(status_list)
    action_map = {0: '💤 HOLD', 1: '🟢 LONG', 2: '🔴 SHORT'}
    status_df['動作'] = status_df['動作'].map(action_map)


    def color_pnl(val):
        color = '#26a69a' if val > 0.001 else '#ef5350' if val < -0.001 else 'white'
        return f'color: {color}; font-weight: bold'


    st.dataframe(
        status_df.style.map(color_pnl, subset=['當前單子 (%)', '生涯總計 (%)'])
            .format({'進場價': '{:.1f}', '最新價': '{:.1f}', '當前單子 (%)': '{:.2f}%', '生涯總計 (%)': '{:.2f}%'}),
        use_container_width=True, hide_index=True
    )

# 強制 1 秒刷新 (及時感來源)
time.sleep(1)
st.rerun()