"""
system_diagnosis.py — SMC TCN 交易系統全面診斷工具

執行：python system_diagnosis.py
      python system_diagnosis.py --db ./wisdom_book.db --models ./models
      python system_diagnosis.py --agent 4   # 只分析特定 agent

輸出：
  1. DB 健康報告（數據品質、分布）
  2. 每個 Agent 的績效分析
  3. 特徵有效性分析（哪些特徵真的有用）
  4. 聖盾通過率分析
  5. 時段勝率熱圖
  6. 問題清單與優先修復建議
"""
import matplotlib
import matplotlib.pyplot as plt


# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 指定微軟正黑體
plt.rcParams['axes.unicode_minus'] = False              # 解決座標軸負號顯示為方塊的問題
plt.rcParams['font.sans-serif'] = ['Segoe UI Emoji', 'Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 如果你是在 Linux/Ubuntu 伺服器跑，請改用 'Noto Sans CJK TC' 或 'Heiti TC'
import sys
import os
import json
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────
try:
    from config import STATE_DIM, STACK_SIZE, TOTAL_DIM, NUM_AGENTS
    FEAT_NAMES = [
        "rsi","atr_norm","p_pos","fvg_dist","is_sweep",
        "hour_sin","hour_cos","amd_phase","asia_pos","pdh_dist",
        "pdl_dist","bos","choch","ob_bull","ob_bear","p_chg",
        "poc_dist","eq_hl","is_ny","is_tue_wed","h4_trend","h4_pos",
        "delta_sum","delta_ratio","trade_intensity",
        "absorption","hurst","vpin","obi"
    ]
except ImportError:
    STATE_DIM, STACK_SIZE, TOTAL_DIM, NUM_AGENTS = 29, 12, 348, 32
    FEAT_NAMES = [f"f{i}" for i in range(STATE_DIM)]

COLORS = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "green":    "#3fb950",
    "red":      "#f85149",
    "yellow":   "#d29922",
    "blue":     "#58a6ff",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["panel"],
    "axes.edgecolor":    COLORS["border"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["subtext"],
    "ytick.color":       COLORS["subtext"],
    "text.color":        COLORS["text"],
    "grid.color":        COLORS["border"],
    "grid.alpha":        0.5,
})


# ══════════════════════════════════════════════════════════════════════
# 1. 資料載入
# ══════════════════════════════════════════════════════════════════════
def load_data(db_path: str, limit: int = 200000) -> pd.DataFrame:
    print(f"[*] 連接資料庫：{db_path}")
    conn = sqlite3.connect(db_path)

    # 自動偵測欄位
    cols_info = pd.read_sql("PRAGMA table_info(experiences)", conn)
    all_cols  = cols_info["name"].tolist()

    base_cols = ["id","agent_id","timestamp","action","reward",
                 "pnl","cur_pnl","entry","vah","val",
                 "is_sweep","is_choch","is_bos","level_desc"]
    phys_cols = [c for c in ["delta_sum","delta_ratio","trade_intensity",
                              "absorption","hurst","vpin","obi"]
                 if c in all_cols]
    state_col = ["state"] if "state" in all_cols else []

    sel = base_cols + phys_cols + state_col
    sel = [c for c in sel if c in all_cols]

    df = pd.read_sql(
        f"SELECT {','.join(sel)} FROM experiences ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()

    df["agent_id"] = df["agent_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["action_name"] = df["action"].map({0:"HOLD", 1:"LONG", 2:"SHORT"})
    print(f"[OK] 載入 {len(df):,} 筆資料，欄位：{sel}")
    return df, phys_cols


# ══════════════════════════════════════════════════════════════════════
# 2. 解析 state 向量
# ══════════════════════════════════════════════════════════════════════
def parse_states(df: pd.DataFrame, sample_n: int = 5000) -> np.ndarray:
    if "state" not in df.columns:
        return None
    print(f"[*] 解析 state 向量（抽樣 {sample_n} 筆）...")
    sample = df[df["state"].notna()].sample(min(sample_n, len(df)), random_state=42)
    states = []
    for s in sample["state"]:
        try:
            arr = np.array(json.loads(s), dtype=np.float32).flatten()
            if len(arr) == TOTAL_DIM:
                # 取最後一幀（最新特徵）
                states.append(arr[-STATE_DIM:])
        except Exception:
            continue
    if not states:
        return None
    arr = np.array(states)
    print(f"[OK] 解析成功 {len(arr)} 筆 state（{arr.shape}）")
    return arr


# ══════════════════════════════════════════════════════════════════════
# 3. 各項分析
# ══════════════════════════════════════════════════════════════════════
def analyze_db_health(df: pd.DataFrame) -> dict:
    total   = len(df)
    actions = df["action"].value_counts()
    agents  = df["agent_id"].nunique()

    # 幽靈持倉
    ghosts = df[(df["action"] == 0) & (df["cur_pnl"].abs() > 0.01)]

    # 重複 entry（開倉後未平倉又開倉）
    trading = df[df["action"] != 0]
    dup_entry = trading[trading["entry"] == 0]

    # PnL 分布
    pnl_mean = df["cur_pnl"].mean()
    pnl_std  = df["cur_pnl"].std()
    pnl_pos  = (df["cur_pnl"] > 0).sum()
    pnl_neg  = (df["cur_pnl"] < 0).sum()

    return {
        "total":        total,
        "agents":       agents,
        "hold_pct":     actions.get(0, 0) / total * 100,
        "long_pct":     actions.get(1, 0) / total * 100,
        "short_pct":    actions.get(2, 0) / total * 100,
        "ghost_count":  len(ghosts),
        "dup_entry":    len(dup_entry),
        "pnl_mean":     pnl_mean,
        "pnl_std":      pnl_std,
        "win_rate":     pnl_pos / (pnl_pos + pnl_neg + 1e-9) * 100,
        "reward_mean":  df["reward"].mean(),
        "reward_std":   df["reward"].std(),
    }


def analyze_per_agent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def _agent_stats(g):
        total = len(g)
        actions = g["action"].values
        cur_pnl = g["cur_pnl"].values

        holds_cnt = (actions == 0).sum()
        longs_cnt = (actions == 1).sum()
        shorts_cnt = (actions == 2).sum()
        trades_cnt = longs_cnt + shorts_cnt

        final_pnl = g["pnl"].values[-1] if total > 0 else 0

        valid_pnl = cur_pnl[~np.isnan(cur_pnl)]
        if len(valid_pnl) > 0:
            wins = (valid_pnl > 0).sum()
            win_rate = wins / (len(valid_pnl) + 1e-9) * 100
        else:
            win_rate = 0.0

        if total > 1:
            changes = actions[1:] != actions[:-1]
            change_idx = np.flatnonzero(changes) + 1

            starts = np.empty(len(change_idx) + 1, dtype=int)
            starts[0] = 0
            starts[1:] = change_idx

            ends = np.empty(len(change_idx) + 1, dtype=int)
            ends[:-1] = change_idx
            ends[-1] = total

            lengths = ends - starts
            block_actions = actions[starts]

            keep = (block_actions != 0) & (lengths > 1)
            if len(keep) > 0:
                keep[-1] = False

            valid_lengths = lengths[keep]
            avg_hold = np.mean(valid_lengths) if len(valid_lengths) > 0 else 0
        else:
            avg_hold = 0

        return {
            "agent": g["agent_id"].values[0],
            "total_records": total,
            "final_pnl": round(final_pnl, 2),
            "win_rate": round(win_rate, 1),
            "avg_hold_bars": round(avg_hold, 1),
            "trade_count": trades_cnt,
            "hold_pct": round(holds_cnt / total * 100, 1),
            "long_pct": round(longs_cnt / total * 100, 1),
            "short_pct": round(shorts_cnt / total * 100, 1),
            "reward_mean": round(g["reward"].mean(), 3),
            "pnl_std": round(np.nanstd(cur_pnl, ddof=1) if len(valid_pnl)>1 else 0.0, 2)
        }

    df_sorted = df.sort_values("id")

    records = [_agent_stats(g) for aid, g in df_sorted.groupby("agent_id", sort=False)]

    res = pd.DataFrame(records)
    cols = ["agent", "total_records", "final_pnl", "win_rate", "avg_hold_bars", "trade_count", "hold_pct", "long_pct", "short_pct", "reward_mean", "pnl_std"]
    return res[cols].sort_values("agent")


def analyze_feature_importance(states: np.ndarray, df_sample: pd.DataFrame) -> pd.DataFrame:
    """用隨機森林分析每個特徵對 reward 的預測力"""
    if states is None or len(states) < 100:
        return pd.DataFrame()

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        n = min(len(states), len(df_sample))
        X = states[:n]
        y = df_sample["reward"].values[:n]

        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(X) < 50:
            return pd.DataFrame()

        sc = StandardScaler()
        X  = sc.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=100, max_depth=6,
                                   n_jobs=-1, random_state=42)
        rf.fit(X, y)

        names = FEAT_NAMES[:STATE_DIM] if len(FEAT_NAMES) >= STATE_DIM else [f"f{i}" for i in range(STATE_DIM)]
        return pd.DataFrame({
            "feature":    names,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
    except ImportError:
        print("[!] sklearn 未安裝，跳過特徵重要性分析")
        return pd.DataFrame()


def analyze_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """每小時的勝率和開倉次數"""
    records = []
    for hour, g in df.groupby("hour"):
        trades = g[g["action"] != 0]
        if len(trades) == 0:
            continue
        wins = (trades["cur_pnl"] > 0).sum()
        records.append({
            "hour":       hour,
            "trades":     len(trades),
            "win_rate":   wins / len(trades) * 100,
            "avg_pnl":    trades["cur_pnl"].mean(),
            "long_count": (trades["action"] == 1).sum(),
            "short_count":(trades["action"] == 2).sum(),
        })
    return pd.DataFrame(records).sort_values("hour")


def analyze_phys_features(df: pd.DataFrame, phys_cols: list) -> dict:
    """物理特徵的分布和相關性"""
    if not phys_cols:
        return {}
    result = {}
    for col in phys_cols:
        vals = df[col].dropna()
        zero_pct = (vals == 0).sum() / len(vals) * 100
        result[col] = {
            "zero_pct": round(zero_pct, 1),
            "mean":     round(vals.mean(), 3),
            "std":      round(vals.std(), 3),
            "min":      round(vals.min(), 3),
            "max":      round(vals.max(), 3),
        }
    return result


def analyze_action_sequence(df: pd.DataFrame) -> dict:
    """分析 action 序列的合理性"""
    # 開倉後立刻平倉（持倉 1 根就結束）的比例
    snap_close = 0
    total_trades = 0
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("id")
        acts = g["action"].values

        if len(acts) < 2:
            continue

        is_zero = (acts == 0)
        is_nonzero = ~is_zero

        # acts[i-1] != 0 and acts[i] == 0
        closes = is_nonzero[:-1] & is_zero[1:]
        total_trades += closes.sum()

        if len(acts) >= 3:
            # acts[i-2] == 0 and acts[i-1] != 0 and acts[i] == 0
            snap_closes = is_zero[:-2] & is_nonzero[1:-1] & is_zero[2:]
            snap_close += snap_closes.sum()

    return {
        "snap_close_pct": float(snap_close / (total_trades + 1e-9) * 100),
        "total_trades":   int(total_trades),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. 問題診斷與建議
# ══════════════════════════════════════════════════════════════════════
def generate_diagnosis(health, agent_df, feat_df, hourly_df, phys_info, seq_info) -> list:
    """
    診斷 15D 軍團健康狀況
    回傳格式: (severity_level, category, description, fix_suggestion)
    severity_level: 'CRITICAL', 'WARNING', 'SUCCESS' (避開 Emoji 防止畫圖報錯)
    """
    issues = []

    # ── 1. 系統生存診斷 (核心 PnL 邏輯) ───────────────────────────
    if health["win_rate"] < 45:
        # 考慮到 15D 高頻交易，手續費損耗極大，勝率低於 45% 基本上是在做公益
        issues.append(('CRITICAL', "獲利能力",
                       f"勝率 {health['win_rate']:.1f}% 太低",
                       "手續費正在吃掉本金。必須提高 OBI 門檻，只在牆體厚度 > 0.6 時開單"))

    if health["ghost_count"] > 50:
        issues.append(('CRITICAL', "系統同步",
                       f"發現 {health['ghost_count']} 筆幽靈持倉",
                       "Action=0 但 PnL 仍在跳動。檢查 risk_manager.py 的 position 歸零邏輯"))

    # ── 2. 行為模式診斷 (解決「躺平」或「過動」) ────────────────────
    # 觀望比例過高 = 躺平
    if health["hold_pct"] > 85:
        issues.append(('WARNING', "戰鬥意願",
                       f"觀望率 {health['hold_pct']:.1f}% (躺平中)",
                       "AI 發現『不動就不會賠』。請檢查 Reward Function 是否給予過重的損益懲罰"))

    # 觀望比例過低 = 多動症 (手續費殺手)
    elif health["hold_pct"] < 30:
        issues.append(('WARNING', "交易頻率",
                       f"觀望率 {health['hold_pct']:.1f}% (過度交易)",
                       "Agent 正在瘋狂刷手續費。請強制加入 LOCK_SECONDS 冷卻時間"))

    # ── 3. Agent 個別審計 (篩選學霸與學渣) ─────────────────────────
    if not agent_df.empty:
        # 抓出賠超過本金 20% 的敗家子
        losers = agent_df[agent_df["final_pnl"] < -20]["agent"].tolist()
        if losers:
            issues.append(('WARNING', "Agent 汰換",
                           f"Agent {losers} 表現極差",
                           "權重已黑化。建議執行『奪舍』：用 Agent 19 的權重直接覆蓋它們"))

        # 檢測「快閃交易」 (逃避行為)
        snappers = agent_df[agent_df["avg_hold_bars"] < 1.5]["agent"].tolist()
        if snappers:
            issues.append(('WARNING', "持倉時間",
                           f"Agent {snappers} 持倉太短",
                           "這是在刷手續費，完全沒吃到趨勢。檢查 VPIN 是否太敏感導致受驚平倉"))

    # ── 4. 29 維特徵有效性 (解決「權重偏差」) ─────────────────────
    if not feat_df.empty:
        # 針對你說的 VPIN 權重過高問題
        if "vpin" in feat_df["feature"].values:
            vpin_imp = feat_df[feat_df["feature"] == "vpin"]["importance"].values[0]
            if vpin_imp > 0.35:
                issues.append(('CRITICAL', "特徵中毒",
                               f"VPIN 權重 {vpin_imp:.3f} 過高",
                               "大腦過度依賴成交量毒性。請將 VPIN 加入 Dropout 或強行削減其輸入權重"))

        # SMC 特徵失效檢查
        smc_feats = ["bos", "choch", "h4_trend", "val", "vah"]
        dead_smc = feat_df[(feat_df["feature"].isin(smc_feats)) & (feat_df["importance"] < 0.02)]["feature"].tolist()
        if dead_smc:
            issues.append(('CRITICAL', "SMC 聖盾失效",
                           f"核心特徵 {dead_smc} 被無視",
                           "模型看不懂結構。請將特徵改為『距離 VAL/VAH 的距離』而非 0/1 訊號"))

    # ── 5. 時段性優勢 ──────────────────────────────────────────────
    if not hourly_df.empty:
        best_h = hourly_df.loc[hourly_df["win_rate"].idxmax()]
        if best_h['win_rate'] > 55:
            issues.append(('SUCCESS', "黃金時段",
                           f"UTC {int(best_h['hour'])} 點表現優異 (勝率 {best_h['win_rate']:.1f}%)",
                           "可在該時段自動放寬 OBI 門檻以增加獲利"))

    # 排序：CRITICAL 最優先，其次 WARNING，最後 SUCCESS
    order = {"CRITICAL": 0, "WARNING": 1, "SUCCESS": 2}
    return sorted(issues, key=lambda x: order.get(x[0], 3))

# ══════════════════════════════════════════════════════════════════════
# 5. 視覺化
# ══════════════════════════════════════════════════════════════════════
def plot_dashboard(health, agent_df, feat_df, hourly_df, phys_info,
                   issues, output_path: str):
    fig = plt.figure(figsize=(22, 28), facecolor=COLORS["bg"])
    fig.suptitle("SMC TCN 交易系統全面診斷報告",
                 fontsize=20, fontweight="bold", color=COLORS["text"],
                 y=0.98)

    gs = gridspec.GridSpec(5, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.04,
                           left=0.06, right=0.97)

    # ── Row 0：系統健康摘要 ────────────────────────────────────────
    ax_health = fig.add_subplot(gs[0, :])
    ax_health.set_facecolor(COLORS["panel"])
    ax_health.axis("off")

    metrics = [
        ("總記錄筆數",  f"{health['total']:,}",       COLORS["blue"]),
        ("Agent 數量",  f"{health['agents']}",         COLORS["blue"]),
        ("整體勝率",    f"{health['win_rate']:.1f}%",
         COLORS["green"] if health["win_rate"] > 50 else COLORS["red"]),
        ("觀望比例",    f"{health['hold_pct']:.1f}%",
         COLORS["yellow"]),
        ("平均 Reward", f"{health['reward_mean']:.3f}",
         COLORS["green"] if health["reward_mean"] > 0 else COLORS["red"]),
        ("幽靈持倉數",  f"{health['ghost_count']}",
         COLORS["green"] if health["ghost_count"] < 10 else COLORS["red"]),
    ]
    for i, (label, val, color) in enumerate(metrics):
        x = 0.02 + i * 0.165
        ax_health.text(x, 0.75, label, transform=ax_health.transAxes,
                       fontsize=9, color=COLORS["subtext"], ha="left")
        ax_health.text(x, 0.25, val, transform=ax_health.transAxes,
                       fontsize=16, fontweight="bold", color=color, ha="left")
    ax_health.set_title("系統健康摘要", color=COLORS["subtext"],
                         fontsize=10, pad=8, loc="left")

    # ── Row 1 左：Agent PnL 排行 ───────────────────────────────────
    ax_pnl = fig.add_subplot(gs[1, 0])
    if len(agent_df) > 0:
        ag = agent_df.sort_values("final_pnl")
        colors = [COLORS["green"] if v >= 0 else COLORS["red"]
                  for v in ag["final_pnl"]]
        bars = ax_pnl.barh(ag["agent"].astype(str), ag["final_pnl"],
                            color=colors, alpha=0.85)
        ax_pnl.axvline(0, color=COLORS["border"], lw=1.5)
        ax_pnl.set_xlabel("累計 PnL (%)")
        ax_pnl.set_title("Agent 累計 PnL 排行", fontsize=10)
        ax_pnl.grid(axis="x", alpha=0.3)

    # ── Row 1 中：Action 分布 ──────────────────────────────────────
    ax_act = fig.add_subplot(gs[1, 1])
    act_vals = [health["hold_pct"], health["long_pct"], health["short_pct"]]
    act_labs = ["HOLD 觀望", "LONG 做多", "SHORT 做空"]
    act_cols = [COLORS["subtext"], COLORS["green"], COLORS["red"]]
    wedges, texts, autotexts = ax_act.pie(
        act_vals, labels=act_labs, colors=act_cols,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": COLORS["text"], "fontsize": 8}
    )
    for at in autotexts:
        at.set_color(COLORS["bg"])
        at.set_fontweight("bold")
    ax_act.set_title("整體 Action 分布", fontsize=10)

    # ── Row 1 右：Agent 勝率分布 ───────────────────────────────────
    ax_wr = fig.add_subplot(gs[1, 2])
    if len(agent_df) > 0:
        wr = agent_df["win_rate"].values
        ax_wr.hist(wr, bins=15, color=COLORS["purple"], alpha=0.8, edgecolor=COLORS["bg"])
        ax_wr.axvline(50, color=COLORS["yellow"], lw=2, linestyle="--", label="50% 盈虧平衡")
        ax_wr.axvline(wr.mean(), color=COLORS["blue"], lw=2, label=f"平均 {wr.mean():.1f}%")
        ax_wr.set_xlabel("勝率 (%)")
        ax_wr.set_ylabel("Agent 數量")
        ax_wr.set_title("Agent 勝率分布", fontsize=10)
        ax_wr.legend(fontsize=7)
        ax_wr.grid(alpha=0.3)

    # ── Row 2 左：特徵重要性 ───────────────────────────────────────
    ax_feat = fig.add_subplot(gs[2, :2])
    if len(feat_df) > 0:
        top = feat_df.head(20)
        colors_f = []
        smc_core = {"bos","choch","is_sweep","ob_bull","ob_bear","h4_trend","h4_pos"}
        phys_core = {"delta_sum","delta_ratio","trade_intensity","absorption","vpin","obi","hurst"}
        for f in top["feature"]:
            if f in smc_core:
                colors_f.append(COLORS["blue"])
            elif f in phys_core:
                colors_f.append(COLORS["purple"])
            else:
                colors_f.append(COLORS["green"])
        ax_feat.barh(top["feature"][::-1], top["importance"][::-1],
                     color=colors_f[::-1], alpha=0.85)
        ax_feat.set_xlabel("重要性")
        ax_feat.set_title("特徵重要性 TOP 20（藍=SMC 紫=物理 綠=其他）", fontsize=10)
        ax_feat.grid(axis="x", alpha=0.3)

    # ── Row 2 右：物理特徵零值比例 ────────────────────────────────
    ax_phys = fig.add_subplot(gs[2, 2])
    if phys_info:
        names = list(phys_info.keys())
        zeros = [phys_info[n]["zero_pct"] for n in names]
        colors_p = [COLORS["red"] if z > 80 else
                    COLORS["yellow"] if z > 50 else COLORS["green"]
                    for z in zeros]
        ax_phys.barh(names, zeros, color=colors_p, alpha=0.85)
        ax_phys.axvline(80, color=COLORS["red"], lw=1.5,
                        linestyle="--", label="80% 警戒線")
        ax_phys.set_xlabel("零值比例 (%)")
        ax_phys.set_title("物理特徵零值率\n（高=連線不穩）", fontsize=10)
        ax_phys.legend(fontsize=7)
        ax_phys.grid(axis="x", alpha=0.3)

    # ── Row 3 左：時段勝率 ────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[3, :2])
    if len(hourly_df) > 0:
        hours = hourly_df["hour"].values
        wr    = hourly_df["win_rate"].values
        cnt   = hourly_df["trades"].values
        colors_h = [COLORS["green"] if w >= 50 else COLORS["red"] for w in wr]
        bars = ax_hr.bar(hours, wr, color=colors_h, alpha=0.8, width=0.8)
        ax_hr.axhline(50, color=COLORS["yellow"], lw=2, linestyle="--", label="50%")
        # 在柱子上顯示開倉次數
        for h, w, c in zip(hours, wr, cnt):
            ax_hr.text(h, w + 1, str(c), ha="center", fontsize=6,
                       color=COLORS["subtext"])
        ax_hr.set_xlabel("UTC 小時")
        ax_hr.set_ylabel("勝率 (%)")
        ax_hr.set_title("各時段勝率（數字=開倉次數，UTC 時間）", fontsize=10)
        ax_hr.set_ylim(0, 110)
        ax_hr.legend(fontsize=8)
        ax_hr.set_xticks(range(0, 24))
        ax_hr.grid(axis="y", alpha=0.3)

    # ── Row 3 右：Agent 持倉週期 ──────────────────────────────────
    ax_hold = fig.add_subplot(gs[3, 2])
    if len(agent_df) > 0:
        holds = agent_df["avg_hold_bars"].values
        ax_hold.hist(holds, bins=10, color=COLORS["blue"], alpha=0.8, edgecolor=COLORS["bg"])
        ax_hold.axvline(holds.mean(), color=COLORS["yellow"], lw=2,
                        label=f"平均 {holds.mean():.1f} 根")
        ax_hold.set_xlabel("平均持倉 K 棒數")
        ax_hold.set_ylabel("Agent 數量")
        ax_hold.set_title("Agent 平均持倉週期", fontsize=10)
        ax_hold.legend(fontsize=7)
        ax_hold.grid(alpha=0.3)

    # ── Row 4：問題診斷清單 (優化版：去 Emoji + 自動對齊) ───────────────────
    ax_issues = fig.add_subplot(gs[4, :])
    ax_issues.axis("off")

    # 設定中文字體 (確保標題能顯示)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    ax_issues.set_title("DEEP AUDIT | 系統診斷與修復建議",
                        fontsize=11,
                        color=COLORS["yellow"],
                        pad=10,
                        loc="left",
                        fontweight="bold")

    # 定義等級對應的顯示字串與顏色
    LEVEL_CFG = {
        "CRITICAL": {"label": " CRIT ", "color": COLORS["red"]},
        "WARNING": {"label": " WARN ", "color": COLORS["yellow"]},
        "SUCCESS": {"label": " PASS ", "color": COLORS["green"]}
    }

    y_pos = 0.95
    # 只顯示前 8 條，避免底部溢出
    for i, (level, cat, desc, fix) in enumerate(issues[:8]):
        cfg = LEVEL_CFG.get(level, {"label": " INFO ", "color": COLORS["subtext"]})

        # 1. 繪製等級標籤 (用有底色的文字框模擬 Emoji 效果)
        ax_issues.text(0.01, y_pos, cfg["label"],
                       transform=ax_issues.transAxes,
                       fontsize=8, color="white", fontweight="bold",
                       bbox=dict(facecolor=cfg["color"], edgecolor='none', boxstyle='round,pad=0.2'))

        # 2. 繪製類別 (例如 [數據品質])
        ax_issues.text(0.06, y_pos, f"[{cat}]",
                       transform=ax_issues.transAxes,
                       fontsize=9, color=cfg["color"], fontweight="bold")

        # 3. 繪製問題描述 (Desc)
        ax_issues.text(0.16, y_pos, desc,
                       transform=ax_issues.transAxes,
                       fontsize=8.5, color=COLORS["text"])

        # 4. 繪製修復建議 (Fix) - 稍微下移一點
        ax_issues.text(0.16, y_pos - 0.06, f"→ 建議修復: {fix}",
                       transform=ax_issues.transAxes,
                       fontsize=7.5, color=COLORS["subtext"], style="italic")

        # 更新下一個項目的位置 (減少間距讓排版更緊湊)
        y_pos -= 0.12
        if y_pos < 0.05:
            break

    # 存檔 (確保包含背景色)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[OK] 戰情報告生成成功：{output_path}")


# ══════════════════════════════════════════════════════════════════════
# 6. 文字報告
# ══════════════════════════════════════════════════════════════════════
def print_report(health, agent_df, feat_df, hourly_df, phys_info,
                 seq_info, issues):
    sep = "═" * 70

    print(f"\n{sep}")
    print("  SMC TCN 交易系統全面診斷報告")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    print("\n【系統健康摘要】")
    print(f"  總記錄：{health['total']:,} 筆 | Agent：{health['agents']} 個")
    print(f"  勝率：{health['win_rate']:.1f}% | 觀望：{health['hold_pct']:.1f}%"
          f" | 做多：{health['long_pct']:.1f}% | 做空：{health['short_pct']:.1f}%")
    print(f"  平均 Reward：{health['reward_mean']:.4f} ± {health['reward_std']:.4f}")
    print(f"  幽靈持倉：{health['ghost_count']} 筆")

    if len(agent_df) > 0:
        print("\n【Agent 績效排行（前 10）】")
        top = agent_df.nlargest(10, "final_pnl")
        print(f"  {'Agent':>6} {'PnL%':>8} {'勝率%':>7} {'交易數':>7} "
              f"{'平均持倉':>8} {'Reward':>8}")
        print("  " + "-" * 55)
        for _, r in top.iterrows():
            pnl_col = "+" if r["final_pnl"] >= 0 else ""
            print(f"  {str(r['agent']):>6} {pnl_col}{r['final_pnl']:>7.2f}%"
                  f" {r['win_rate']:>6.1f}% {int(r['trade_count']):>7}"
                  f" {r['avg_hold_bars']:>7.1f}根 {r['reward_mean']:>8.3f}")

    if len(feat_df) > 0:
        print("\n【特徵重要性 TOP 10】")
        for _, r in feat_df.head(10).iterrows():
            bar = "█" * int(r["importance"] * 200)
            print(f"  {r['feature']:>18}: {r['importance']:.4f} {bar}")

        print("\n【特徵重要性墊底（需關注）】")
        smc = {"bos","choch","is_sweep","ob_bull","ob_bear","h4_trend"}
        bottom = feat_df.tail(10)
        for _, r in bottom.iterrows():
            flag = " ← SMC核心！" if r["feature"] in smc else ""
            print(f"  {r['feature']:>18}: {r['importance']:.4f}{flag}")

    if phys_info:
        print("\n【物理特徵健康狀況】")
        for col, info in phys_info.items():
            status = ("❌ 幾乎無效" if info["zero_pct"] > 80 else
                      "⚠️ 偏低"    if info["zero_pct"] > 50 else "✅ 正常")
            print(f"  {col:>18}: 零值{info['zero_pct']:>5.1f}% | "
                  f"均值{info['mean']:>7.3f} | 標準差{info['std']:>6.3f} {status}")

    if len(hourly_df) > 0:
        print("\n【時段分析（UTC）】")
        print(f"  {'小時':>4} {'勝率%':>7} {'開倉數':>7} {'平均PnL':>9}")
        print("  " + "-" * 32)
        for _, r in hourly_df.iterrows():
            flag = " ★" if r["win_rate"] >= 55 else (" ✗" if r["win_rate"] < 40 else "")
            print(f"  {int(r['hour']):>4}  {r['win_rate']:>6.1f}%"
                  f" {int(r['trades']):>7} {r['avg_pnl']:>9.3f}{flag}")

    print(f"\n{'─'*70}")
    print("【問題診斷清單】")
    for sev, cat, desc, fix in issues:
        print(f"\n  {sev} [{cat}]")
        print(f"     問題：{desc}")
        print(f"     修法：{fix}")

    print(f"\n{sep}")
    print(f"  共發現 {len([i for i in issues if i[0]=='🔴'])} 個嚴重問題、"
          f"{len([i for i in issues if i[0]=='🟡'])} 個警告、"
          f"{len([i for i in issues if i[0]=='🟢'])} 個優化建議")
    print(sep)


# ══════════════════════════════════════════════════════════════════════
# 7. 主程式
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SMC TCN 系統診斷")
    parser.add_argument("--db",     default="./wisdom_book.db")
    parser.add_argument("--models", default="./models")
    parser.add_argument("--agent",  type=int, default=None)
    parser.add_argument("--limit",  type=int, default=200000)
    parser.add_argument("--out",    default="./diagnosis_report.png")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[ERR] 找不到資料庫：{args.db}")
        sys.exit(1)

    # 載入資料
    df, phys_cols = load_data(args.db, args.limit)

    if args.agent is not None:
        df = df[df["agent_id"] == str(args.agent)]
        print(f"[*] 只分析 Agent {args.agent}，共 {len(df)} 筆")

    # 各項分析
    print("[*] 分析系統健康狀況...")
    health   = analyze_db_health(df)

    print("[*] 分析各 Agent 績效...")
    agent_df = analyze_per_agent(df)

    print("[*] 解析 state 向量...")
    states   = parse_states(df)
    df_feat  = df[df["state"].notna()].sample(
        min(5000, len(df[df["state"].notna()])), random_state=42
    ) if "state" in df.columns else df

    print("[*] 分析特徵重要性（需要 sklearn）...")
    feat_df  = analyze_feature_importance(states, df_feat)

    print("[*] 分析時段勝率...")
    hourly_df = analyze_hourly(df)

    print("[*] 分析物理特徵...")
    phys_info = analyze_phys_features(df, phys_cols)

    print("[*] 分析 Action 序列...")
    seq_info  = analyze_action_sequence(df)

    # 生成診斷報告
    print("[*] 生成診斷建議...")
    issues = generate_diagnosis(health, agent_df, feat_df,
                                hourly_df, phys_info, seq_info)

    # 輸出
    print_report(health, agent_df, feat_df, hourly_df,
                 phys_info, seq_info, issues)

    print(f"\n[*] 產生視覺化圖表...")
    plot_dashboard(health, agent_df, feat_df, hourly_df,
                   phys_info, issues, args.out)

    print(f"\n[完成] 報告圖表：{args.out}")


if __name__ == "__main__":
    main()