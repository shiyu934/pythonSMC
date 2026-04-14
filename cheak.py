import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- 配置區：完全對齊聖盾邏輯 (29 維) ---
DB_PATH = "wisdom_book.db"

# 嚴格對齊 F_RSI (0) 到 F_OBI (28) 的索引順序
FEATURE_NAMES = [
    "rsi", "atr_norm", "p_pos", "fvg_dist", "is_sweep",  # 0-4
    "hour_sin", "hour_cos", "amd_phase", "asia_pos", "pdh_dist",  # 5-9
    "pdl_dist", "bos", "choch", "ob_bull", "ob_bear",  # 10-14
    "p_chg", "poc_dist", "eq_hl", "is_ny", "is_tue_wed",  # 15-19
    "h4_trend", "h4_pos",  # 20-21
    "delta_sum", "delta_ratio", "trade_intensity",  # 22-24
    "absorption", "hurst", "vpin", "obi"  # 25-28 (核心物理層)
]


def extract_last_features(state_str):
    """
    資工系自動對齊：根據 FEATURE_NAMES 長度 (29) 動態抓取特徵
    """
    target_dim = len(FEATURE_NAMES)
    try:
        s = json.loads(state_str)
        # 處理 TCN 的序列數據 [Batch, Seq, Dim]
        if isinstance(s, list) and len(s) > 0 and isinstance(s[0], list):
            flat_state = s[-1]
        else:
            # 扁平模式
            flat_state = s if isinstance(s, list) else []

        # 自動截斷或補齊，確保矩陣形狀完美 (N, 29)
        if len(flat_state) < target_dim:
            return [float(x) for x in flat_state] + [0.0] * (target_dim - len(flat_state))
        return [float(x) for x in flat_state[:target_dim]]
    except Exception:
        return [0.0] * target_dim


def analyze_all():
    conn = sqlite3.connect(DB_PATH)
    print(f"[*] 正在讀取 {DB_PATH} 戰報 (29D 聖盾模式)...")

    # 讀取最近 15000 筆，確保涵蓋 29 維更新後的數據
    df = pd.read_sql_query("SELECT * FROM experiences ORDER BY id DESC LIMIT 15000", conn)
    conn.close()

    if df.empty:
        print("[!] 資料庫是空的，請先讓戰士跑幾分鐘。")
        return

    # 1. 腦袋性能對決
    print("[1] 正在計算腦袋性能...")
    df['brain_type'] = 'TCN_29D'
    performance = df.groupby('brain_type').agg({
        'cur_pnl': ['mean', 'std', 'sum'],
        'reward': 'mean',
        'id': 'count'
    })
    print("\n--- 腦袋性能對比 ---")
    print(performance)

    # 2. SMC 物理邊界驗證 (VAL 附近)
    val_trades = df[df['action'] != 0].copy()
    if not val_trades.empty:
        # 計算進場價與 VAL 的偏離度
        val_trades['dist_to_val'] = (val_trades['entry'] - val_trades['val']) / (val_trades['val'] + 1e-9)
        near_val = val_trades[val_trades['dist_to_val'].abs() < 0.001]
        if not near_val.empty:
            print(f"\n[2] VAL 邊界效應：靠近時平均盈虧 {near_val['cur_pnl'].mean():.4f}%")

    # 3. 特徵重要性分析
    print(f"\n[3] 正在解剖 {len(FEATURE_NAMES)} 維特徵權重...")
    X_list = df['state'].apply(extract_last_features).tolist()
    X = np.array(X_list)
    y = df['reward'].values

    # 特徵數量校驗
    if X.shape[1] != len(FEATURE_NAMES):
        print(f"[!] 維度不匹配：數據有 {X.shape[1]} 維，但對齊目標為 {len(FEATURE_NAMES)} 維")
        feature_labels = [f"F_{i}" for i in range(X.shape[1])]
    else:
        feature_labels = FEATURE_NAMES

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': feature_labels,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # --- 繪圖區 (修正 FutureWarning 並優化 29D 顯示) ---
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 14))  # 稍微拉高高度，避免 29 個特徵太擠

    # 子圖 1：特徵重要性 (解決 palette 警告)
    plt.subplot(2, 1, 1)
    sns.barplot(
        x='importance',
        y='feature',
        data=importance_df.head(29),
        hue='feature',  # 修正處 1：明確指定 hue 為 feature
        palette='viridis',
        legend=False  # 修正處 2：關閉圖例，保持畫面乾淨
    )
    plt.title("29D SMC Feature Importance (TCN Decision Drivers)", fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("SMC & Physical Features", fontsize=12)

    # 子圖 2：24 小時 PnL 分佈
    plt.subplot(2, 1, 2)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    sns.boxplot(
        x='hour',
        y='cur_pnl',
        data=df[df['action'] != 0],
        hue='hour',  # 同樣加上 hue 避免未來警告
        palette='magma',
        legend=False
    )
    plt.axhline(0, color='cyan', linestyle='--', alpha=0.5)
    plt.title("PnL Distribution by Hour (Killzone Performance)", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_all()