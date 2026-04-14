import sqlite3
import pandas as pd
import mplfinance as mpf
import numpy as np

# --- 戰報點名配置 ---
DB_PATH = "wisdom_book.db"
TARGET_AGENT = "9"  # <--- 在這裡輸入你想看的那位戰士 (例如 Agent_9, Agent_17)
K_LINE_COUNT = 100  # 顯示最近幾根 K 線


def plot_specific_agent_warroom():
    conn = sqlite3.connect(DB_PATH)

    # 1. 抓取 K 線 (保持全局一致)
    ohlc_df = pd.read_sql_query(
        f"SELECT datetime, open, high, low, close, volume FROM ohlcv ORDER BY ts DESC LIMIT {K_LINE_COUNT}", conn)

    # 2. 抓取「特定 Agent」的交易紀錄
    # 必須按時間升序排，平倉邏輯 (Shift) 才不會亂掉
    trade_df = pd.read_sql_query(f"""
        SELECT timestamp, action, entry, cur_pnl, vah, val 
        FROM experiences 
        WHERE agent_id = ? 
        ORDER BY timestamp ASC
    """, conn, params=(TARGET_AGENT,))
    conn.close()

    if trade_df.empty:
        print(f"[!] 找不到 {TARGET_AGENT} 的交易數據，請檢查 ID 是否輸入正確。")
        return
    if ohlc_df.empty: return

    # --- 數據預處理 ---
    ohlc_df['datetime'] = pd.to_datetime(ohlc_df['datetime'])
    ohlc_df.set_index('datetime', inplace=True)
    ohlc_df = ohlc_df.sort_index()
    trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])

    # --- 關鍵：抓取該 Agent 的平倉瞬間 ---
    trade_df['prev_action'] = trade_df['action'].shift(1).fillna(0)
    # 平倉點：現在是 0 且剛才不是 0
    trade_df['is_exit'] = (trade_df['action'] == 0) & (trade_df['prev_action'].isin([1, 2]))
    # 開倉點：現在不是 0 且剛才是 0
    trade_df['is_entry'] = (trade_df['action'].isin([1, 2])) & (trade_df['prev_action'] == 0)

    # --- 建立對齊 K 線的訊號層 ---
    long_entries = np.full(len(ohlc_df), np.nan)
    short_entries = np.full(len(ohlc_df), np.nan)
    exit_markers = np.full(len(ohlc_df), np.nan)
    vah_area = np.full(len(ohlc_df), np.nan)
    val_area = np.full(len(ohlc_df), np.nan)

    for _, row in trade_df.iterrows():
        # 尋找與 K 線時間最接近的索引
        idx_search = ohlc_df.index.get_indexer([row['timestamp']], method='nearest')[0]
        # 確保時間誤差在 15 分鐘（一根 K 線）內
        if idx_search < 0: continue

        # 標註進場 (Entry)
        if row['is_entry']:
            if row['action'] == 1:  # Long
                long_entries[idx_search] = ohlc_df.iloc[idx_search]['low'] * 0.997
            elif row['action'] == 2:  # Short
                short_entries[idx_search] = ohlc_df.iloc[idx_search]['high'] * 1.003

        # 標註出場 (Exit)
        if row['is_exit']:
            exit_markers[idx_search] = ohlc_df.iloc[idx_search]['close']

        # 聖盾範圍同步
        vah_area[idx_search] = row['vah']
        val_area[idx_search] = row['val']

    # --- 繪圖渲染 ---
    apds = [
        mpf.make_addplot(long_entries, type='scatter', markersize=200, marker='^', color='#00ff00'),
        mpf.make_addplot(short_entries, type='scatter', markersize=200, marker='v', color='#ff0000'),
        mpf.make_addplot(exit_markers, type='scatter', markersize=120, marker='o', color='white', alpha=0.9),
        mpf.make_addplot(vah_area, color='cyan', width=0.5, alpha=0.3),
        mpf.make_addplot(val_area, color='magenta', width=0.5, alpha=0.3),
    ]

    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(base_mpl_style='dark_background', marketcolors=mc, gridstyle='--', y_on_right=True)

    print(f"[*] 正在生成 {TARGET_AGENT} 的 29D 視覺化戰報...")
    mpf.plot(ohlc_df, type='candle', style=s, addplot=apds,
             title=f"\n{TARGET_AGENT} Strategy Replay\nTriangle=Entry, White Circle=Exit",
             volume=True, figsize=(16, 10),
             fill_between=dict(y1=vah_area, y2=val_area, alpha=0.1, color='cyan'))


if __name__ == "__main__":
    plot_specific_agent_warroom()