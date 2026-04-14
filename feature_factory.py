import pandas as pd
import numpy as np
from collections import deque
from config import STATE_DIM, STACK_SIZE


def _pivot_high(values, lb=5):
    pivots = []
    for i in range(lb, len(values) - lb):
        win = list(values[i-lb:i]) + list(values[i+1:i+lb+1])
        if values[i] >= max(win):
            pivots.append((i, float(values[i])))
    return pivots

def _pivot_low(values, lb=5):
    pivots = []
    for i in range(lb, len(values) - lb):
        win = list(values[i-lb:i]) + list(values[i+1:i+lb+1])
        if values[i] <= min(win):
            pivots.append((i, float(values[i])))
    return pivots


class StateStacker:
    def __init__(self, state_dim=STATE_DIM, stack_size=STACK_SIZE):
        self.state_dim, self.stack_size = state_dim, stack_size
        self.reset()

    def reset(self):
        self.buffer = deque(
            [np.zeros(self.state_dim) for _ in range(self.stack_size)],
            maxlen=self.stack_size
        )

    def update_and_get(self, new_features):
        self.buffer.append(np.array(new_features, dtype=np.float32))
        return np.concatenate(list(self.buffer), axis=0)


class FeatureFactory:
    """
    22 維特徵（對齊 config.py 的 F_* 索引）
    所有時間均使用 UTC，對齊 Binance 時間戳

    【核心改動】把稀疏二元特徵改成連續值：
    - bos       : 0/±1 → 突破幅度（%）
    - choch     : 0/±1 → 反轉幅度（前段創低深度 × 突破距離）
    - is_sweep  : 0/±1 → 掃蕩深度（%）
    - is_ny     : 0/1  → 距 NY 開盤的小時距離（連續）
    - is_tue_wed: 0/1  → 改成 weekday 數值（0~6）
    - eq_hl     : 0/±1 → 等高等低的相似程度（越接近 0 越相似）
    - atr_norm  : ×100 → ×50，避免單一特徵佔比過大
    """

    @staticmethod
    def build_state(raw_list):
        try:
            df = pd.DataFrame(raw_list, columns=["ts", "open", "high", "low", "close", "volume"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("ts").reset_index(drop=True)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["close", "high", "low", "open"])
            df = df[df["close"] > 0]
            df = df.reset_index(drop=True)

            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df["volume"] = df["volume"].fillna(0)

            if len(df) < 32:
                return pd.Series(np.zeros(STATE_DIM))

            last_c = df["close"].iloc[-1]
            last_h = df["high"].iloc[-1]
            last_l = df["low"].iloc[-1]
            curr_hour    = df["dt"].iloc[-1].hour
            curr_weekday = df["dt"].iloc[-1].weekday()
            today_utc    = df["dt"].iloc[-1].date()   # 提前定義，AMD 和 PDH/PDL 都需要

            # ── [0] RSI（Wilder 平滑）────────────────────────────────
            delta    = df["close"].diff()
            gain     = delta.where(delta > 0, 0.0)
            loss     = delta.where(delta < 0, 0.0).abs()
            avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
            avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
            rsi = (100 - 100 / (1 + avg_gain / (avg_loss + 1e-9))) / 100.0

            # ── [1] ATR 正規化（×50，避免單一特徵佔比過大）──────────
            # 原本 ×100 導致 atr_norm 佔比 40%，壓縮到 ×50 讓其他特徵有機會
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs()
            ], axis=1).max(axis=1)
            atr_norm = tr.tail(14).mean() / (last_c + 1e-9)

            # ── [2] PD Ratio (Volume Profile VAH/VAL) ────────────────
            vp_df = df.tail(96).copy()   # 用一整天 96 根，Volume Profile 更準
            p_min, p_max = vp_df["low"].min(), vp_df["high"].max()
            vah, val = last_c, last_c

            if p_max > p_min * 1.0005 and p_min > 0:
                try:
                    bins = np.linspace(p_min, p_max, 20)
                    vp = vp_df.groupby(
                        pd.cut((vp_df["high"] + vp_df["low"]) / 2, bins, include_lowest=True),
                        observed=False
                    )["volume"].sum().fillna(0)

                    if vp.sum() > 0:
                        poc_i = int(vp.argmax())
                        tv = vp.sum() * 0.7
                        cv = float(vp.iloc[poc_i])
                        lo = hi = poc_i
                        for _ in range(len(vp)):
                            if cv >= tv: break
                            vlo = float(vp.iloc[lo-1]) if lo > 0 else 0.0
                            vhi = float(vp.iloc[hi+1]) if hi < len(vp)-1 else 0.0
                            if vlo >= vhi:
                                cv += vlo; lo = max(0, lo-1)
                            else:
                                cv += vhi; hi = min(len(vp)-1, hi+1)
                        vah_raw = bins[hi+1] if hi+1 < len(bins) else p_max
                        val_raw = bins[lo]
                        if np.isfinite(vah_raw) and np.isfinite(val_raw) and vah_raw > val_raw:
                            vah, val = float(vah_raw), float(val_raw)
                except Exception:
                    pass

            p_pos    = (last_c - val) / (vah - val + 1e-9) if abs(vah - val) > 1e-9 else 0.5
            poc_dist = (last_c / ((vah + val) / 2 + 1e-9)) - 1

            # ── [3] FVG 距離（持續追蹤版）────────────────────────────
            # 舊版：只看最後一根，FVG 剛形成才有值，下一根歸零
            # 新版：搜尋最近 30 根，找最近的 FVG，輸出現價距 FVG 中點的距離
            #   正值 = 現價在 FVG 上方（上行 FVG，看多）
            #   負值 = 現價在 FVG 下方（下行 FVG，看空）
            #   絕對值越小 = 越接近 FVG（回填機率越高）
            fvg_val = 0.0
            fvg_search = min(30, len(df) - 2)
            for fi in range(2, fvg_search + 2):
                if fi >= len(df):
                    break
                h3 = float(df["high"].iloc[-fi])
                l3 = float(df["low"].iloc[-fi])
                h1 = float(df["high"].iloc[-fi-2]) if fi+2 <= len(df) else h3
                l1 = float(df["low"].iloc[-fi-2])  if fi+2 <= len(df) else l3
                if l3 > h1:
                    # 上行 FVG：K1 的高點 < K3 的低點
                    fvg_mid = (l3 + h1) / 2
                    fvg_val = (last_c - fvg_mid) / (fvg_mid + 1e-9) * 100
                    break
                elif h3 < l1:
                    # 下行 FVG：K1 的低點 > K3 的高點
                    fvg_mid = (h3 + l1) / 2
                    fvg_val = (last_c - fvg_mid) / (fvg_mid + 1e-9) * 100
                    break

            # ── [4] Sweep 深度（連續值）──────────────────────────────
            # 原本：0/±1 二元值，AI 無法感知掃蕩強弱
            # 改成：掃蕩深度百分比，值越大代表掃得越深
            lookback = 48
            prev_high = df["high"].iloc[-lookback:-3].max() if len(df) > lookback else df["high"].iloc[:-3].max()
            prev_low  = df["low"].iloc[-lookback:-3].min()  if len(df) > lookback else df["low"].iloc[:-3].min()
            recent_low_3  = df["low"].iloc[-3:].min()
            recent_high_3 = df["high"].iloc[-3:].max()

            sweep_depth = 0.0
            if recent_low_3 < prev_low:
                # 多方掃蕩：只要破前低就算，不要求已收回
                # 已收回（last_c > prev_low）給更高分數
                depth = float((prev_low - recent_low_3) / (prev_low + 1e-9) * 100)
                if last_c > prev_low:
                    sweep_depth = depth           # 完整 sweep（已收回）
                else:
                    sweep_depth = depth * 0.4     # 部分 sweep（還在低位）
            elif recent_high_3 > prev_high:
                depth = float((recent_high_3 - prev_high) / (prev_high + 1e-9) * 100)
                if last_c < prev_high:
                    sweep_depth = -depth          # 完整空方 sweep
                else:
                    sweep_depth = -depth * 0.4    # 部分空方 sweep

            # ── [5][6] 時間循環編碼 ───────────────────────────────────
            hour_sin = np.sin(2 * np.pi * curr_hour / 24)
            hour_cos = np.cos(2 * np.pi * curr_hour / 24)

            # ── [7] AMD 階段 + 時段內位置（移植自 TFlab AMD 指標）────
            # 原本：4 個離散值（0/0.33/0.67/1.0），資訊損失大
            # 改成：三個時段 × 價格在時段內的相對位置
            #
            # 時段定義（UTC，對應 NY 時間）：
            #   Accumulation : 00:00-06:00 UTC（亞盤積累，養肥羊）
            #   Manipulation : 06:00-12:00 UTC（倫敦操控，掃流動性）
            #   Distribution : 12:00-18:00 UTC（紐約分配，真實方向）
            #   Transition   : 18:00-24:00 UTC（過渡，等待下一輪）

            # 判斷當前時段
            if   0  <= curr_hour <  6:
                amd_session = 0   # Accumulation
                sess_start_h = 0
                sess_end_h   = 6
            elif 6  <= curr_hour < 12:
                amd_session = 1   # Manipulation
                sess_start_h = 6
                sess_end_h   = 12
            elif 12 <= curr_hour < 18:
                amd_session = 2   # Distribution
                sess_start_h = 12
                sess_end_h   = 18
            else:
                amd_session = 3   # Transition
                sess_start_h = 18
                sess_end_h   = 24

            # 計算當前時段的高低點
            sess_df = df[
                (df["dt"].dt.hour >= sess_start_h) &
                (df["dt"].dt.hour <  sess_end_h)   &
                (df["dt"].dt.date == today_utc)
            ]
            if not sess_df.empty:
                sess_h = sess_df["high"].max()
                sess_l = sess_df["low"].min()
            else:
                # 時段剛開始，用最近 4 根估算
                sess_h = df["high"].tail(4).max()
                sess_l = df["low"].tail(4).min()

            # 時段進度（0=剛開始 1=快結束）
            sess_progress = (curr_hour - sess_start_h) / max(sess_end_h - sess_start_h, 1)

            # 現價在時段內的相對位置（0=底部 1=頂部）
            sess_pos = float((last_c - sess_l) / (sess_h - sess_l + 1e-9))
            sess_pos = float(np.clip(sess_pos, -0.5, 1.5))

            # amd_phase：編碼時段 + 時段內位置
            # 值域：0~3 代表時段，小數代表時段內位置
            # 例：1.7 = Manipulation 時段，位置在 70% 高度
            amd_phase = float(amd_session) + float(np.clip(sess_progress, 0, 0.99))

            # ── [8] 亞盤座標 ──────────────────────────────────────────
            asia_df = df[
                (df["dt"].dt.date == today_utc) &
                (df["dt"].dt.hour >= 0) &
                (df["dt"].dt.hour <  6)
            ]
            if not asia_df.empty:
                asia_h = asia_df["high"].max()
                asia_l = asia_df["low"].min()
            else:
                asia_h = df["high"].tail(24).max()
                asia_l = df["low"].tail(24).min()
            asia_pos = (last_c - asia_l) / (asia_h - asia_l + 1e-9)

            # ── [9][10] PDH / PDL ─────────────────────────────────────
            df["date_utc"] = df["dt"].dt.date
            yesterday_df  = df[df["date_utc"] < today_utc]
            if not yesterday_df.empty:
                pdh = yesterday_df.groupby("date_utc")["high"].max().iloc[-1]
                pdl = yesterday_df.groupby("date_utc")["low"].min().iloc[-1]
            else:
                pdh = df["high"].tail(96).max()
                pdl = df["low"].tail(96).min()
            pdh_dist = (last_c / (pdh + 1e-9)) - 1
            pdl_dist = (last_c / (pdl + 1e-9)) - 1

            # ── [11] BOS 突破幅度（Pivot 版）────────────────────────
            # 用真正的擺動高低點計算 BOS，對應 Pine Script ta.pivothigh
            _ph_vals = _pivot_high(df["high"].values, lb=5)
            _pl_vals = _pivot_low(df["low"].values,   lb=5)
            valid_ph = [v for i, v in _ph_vals if i < len(df) - 5]
            valid_pl = [v for i, v in _pl_vals if i < len(df) - 5]

            struct_high = float(valid_ph[-1]) if valid_ph else df["high"].iloc[-60:-1].max()
            struct_low  = float(valid_pl[-1]) if valid_pl else df["low"].iloc[-60:-1].min()

            if last_c > struct_high:
                bos = float((last_c - struct_high) / (struct_high + 1e-9) * 100)
            elif last_c < struct_low:
                bos = float((last_c - struct_low) / (struct_low + 1e-9) * 100)
            else:
                dist_to_high = (struct_high - last_c) / (struct_high + 1e-9) * 100
                dist_to_low  = (last_c - struct_low)  / (struct_low  + 1e-9) * 100
                bos = float(np.clip(dist_to_low - dist_to_high, -2.0, 2.0))

            # ── [12] CHoCH 反轉幅度（連續值）────────────────────────
            # 原本：0/±1，AI 只知道有沒有反轉
            # 改成：反轉信念強度 = 前段序列創低/高的幅度 × 突破距離
            # ── [12] CHoCH 反轉幅度（Pivot 版）──────────────────────
            choch = 0.0
            if len(_pl_vals) >= 2 and len(_ph_vals) >= 2:
                last_pl = _pl_vals[-1][1]; prev_pl = _pl_vals[-2][1]
                last_ph = _ph_vals[-1][1]; prev_ph = _ph_vals[-2][1]

                if last_pl < prev_pl and last_c > struct_high:
                    seq_depth  = (prev_pl - last_pl) / (prev_pl + 1e-9) * 100
                    break_dist = (last_c - struct_high) / (struct_high + 1e-9) * 100
                    choch = float(np.clip(seq_depth + break_dist, 0, 5))
                elif last_ph > prev_ph and last_c < struct_low:
                    seq_depth  = (last_ph - prev_ph) / (prev_ph + 1e-9) * 100
                    break_dist = (struct_low - last_c) / (struct_low + 1e-9) * 100
                    choch = float(np.clip(-(seq_depth + break_dist), -5, 0))

            # ── [13][14] Order Block 距離特徵（放寬版）─────────────
            # 問題：舊版要求「空頭K棒 + 下一根立刻突破」條件太嚴
            # 診斷：ob_bull/ob_bear 在 DB 裡 100% 是 0
            #
            # 修法：兩步驟
            #   1. 先找最近的空頭K棒（多方OB候選）/ 多頭K棒（空方OB候選）
            #   2. 確認該K棒之後的某根（最多 10 根內）有突破，就認定為 OB
            #   3. 輸出當前價格距 OB 中點的百分比距離
            #
            # 這樣 OB 不再是瞬間值，而是持續追蹤最近的 OB 位置
            ob_bull = ob_bear = 0.0
            search_range = min(80, len(df) - 12)

            # 提取 NumPy 陣列以加速運算
            if search_range >= 2:
                c_arr = df["close"].values[-search_range:]
                o_arr = df["open"].values[-search_range:]
                h_arr = df["high"].values[-search_range:]
                l_arr = df["low"].values[-search_range:]

                bull_found = False
                bear_found = False

                for j in range(2, search_range):
                    if bull_found and bear_found:
                        break

                    idx = search_range - j
                    c_j = c_arr[idx]
                    o_j = o_arr[idx]

                    # 多方 OB：找最近的空頭K棒，且之後 10 根內有突破其最高點
                    if not bull_found and c_j < o_j:
                        h_j = h_arr[idx]
                        limit = min(10, j)
                        if np.any(c_arr[idx+1:idx+limit] > h_j):
                            ob_mid = (o_j + c_j) / 2
                            dist = (last_c - ob_mid) / (ob_mid + 1e-9) * 100
                            ob_bull = float(np.clip(dist, -10, 10))
                            bull_found = True

                    # 空方 OB：找最近的多頭K棒，且之後 10 根內有跌破其最低點
                    if not bear_found and c_j > o_j:
                        l_j = l_arr[idx]
                        limit = min(10, j)
                        if np.any(c_arr[idx+1:idx+limit] < l_j):
                            ob_mid = (o_j + c_j) / 2
                            dist = (last_c - ob_mid) / (ob_mid + 1e-9) * 100
                            ob_bear = float(np.clip(dist, -10, 10))
                            bear_found = True

            # ── [15] 價格變化率 ────────────────────────────────────────
            p_chg = df["close"].pct_change().iloc[-1] * 10

            # ── [17] Equal Highs/Lows 相似程度（連續值）─────────────
            # 原本：0/±1，AI 只知道有沒有等高等低
            # 改成：兩個高點差距百分比（越接近 0 越相似，越危險）
            # ── [17] Equal Highs/Lows（移植自 AlgoAlpha EQH/EQL）────
            # AlgoAlpha 核心邏輯：
            #   1. 動態門檻：用 EMA(500) 平滑的波動率當基準，不用固定百分比
            #   2. RSI 過濾：EQH 必須在超買區，EQL 必須在超賣區
            #   3. 失效偵測：被收盤突破後不再計算該層級
            #
            # 對應 Pine Script：
            #   varian   = avg(|high-high[1]|, |low-low[1]|)
            #   smoothvar = ema(varian, 500)
            #   eqh = |high-high[1]| < smoothvar * tolerance
            eq_hl = 0.0
            if len(df) >= 30:
                highs = df["high"].values
                lows  = df["low"].values
                closes = df["close"].values

                # 動態波動率基準（對應 smoothvar）
                h_diffs = np.abs(np.diff(highs))
                l_diffs = np.abs(np.diff(lows))
                varian  = (h_diffs + l_diffs) / 2
                # EMA 平滑（用最近 30 根估算，實際應 500 根，這裡取可用最大值）
                smooth_len = min(len(varian), 100)
                weights = np.exp(np.linspace(-1, 0, smooth_len))
                weights /= weights.sum()
                smoothvar = float(np.convolve(varian, weights, mode='valid')[-1])

                tolerance = 0.05   # 對應 AlgoAlpha 預設值

                # RSI 過濾（對應 state_thresh=5）
                rsi_val = float(rsi)   # 已在上方計算，0~1 範圍
                rsi_overbought = rsi_val > 0.55   # RSI > 55%
                rsi_oversold   = rsi_val < 0.45   # RSI < 45%

                # EQH：最近兩根高點差距 < 動態門檻 + RSI 超買
                if len(highs) >= 2:
                    h_diff = abs(highs[-1] - highs[-2])
                    if h_diff < smoothvar * tolerance and rsi_overbought:
                        # 強度：差距越小越強（1.0 = 完全相等）
                        eq_hl = float(np.clip(1.0 - h_diff / (smoothvar * tolerance + 1e-9), 0, 1.0))
                        # 失效檢查：如果收盤突破等高點，強度衰減
                        eqh_level = max(highs[-1], highs[-2])
                        if closes[-1] > eqh_level:
                            eq_hl *= 0.3   # 已被突破，訊號衰減

                # EQL：最近兩根低點差距 < 動態門檻 + RSI 超賣
                if len(lows) >= 2 and eq_hl == 0.0:
                    l_diff = abs(lows[-1] - lows[-2])
                    if l_diff < smoothvar * tolerance and rsi_oversold:
                        eq_hl = float(-np.clip(1.0 - l_diff / (smoothvar * tolerance + 1e-9), 0, 1.0))
                        eql_level = min(lows[-1], lows[-2])
                        if closes[-1] < eql_level:
                            eq_hl *= 0.3

            # ── [18] Killzone 距離（NY + London 雙重考量）────────────
            # 原本只考慮 NY，漏掉 London（06:00 UTC）
            # 改成：取 NY 和 London 中距離更近的那個
            # 值域：在 Killzone 內=高值，遠離=低值
            dist_to_ny     = abs(curr_hour - 12)
            dist_to_ny     = min(dist_to_ny, 24 - dist_to_ny)
            dist_to_london = abs(curr_hour - 6)
            dist_to_london = min(dist_to_london, 24 - dist_to_london)
            # 取距離最近的 Killzone
            min_kz_dist = min(dist_to_ny, dist_to_london)
            ny_proximity = float(1.0 / (1.0 + min_kz_dist))

            # ── [19] 週間效應（循環編碼）────────────────────────────
            # 原本線性編碼：週一=0, 週日=1，但週一和週日其實相鄰（循環）
            # 改成：sin/cos 循環編碼，讓 AI 理解週間是循環的
            # 但 22 維已滿，只用 sin（cos 另一半資訊量較低）
            # 週二(1)和週三(2)是 ICT 最重要的反轉日，sin 值最高
            weekday_val = float(np.sin(2 * np.pi * curr_weekday / 7))

            # ── Hurst 指數（用最近 60 根 K 棒的收盤價計算）────────
            # H > 0.5：趨勢延續，可以追突破
            # H < 0.5：均值回歸，突破可能是假的
            # H = 0.5：隨機漫步
            hurst_val = 0.5   # 預設中性
            if len(df) >= 30:
                try:
                    prices = df["close"].tail(60).values
                    # prices 必須有足夠變動才能算 Hurst
                    price_std = np.std(prices)
                    if price_std < 1e-6:
                        hurst_val = 0.5   # 價格完全不動，跳過
                    else:
                        lags = range(2, min(20, len(prices) // 2))
                        tau  = [float(np.std(
                            np.subtract(prices[lag:], prices[:-lag])
                        )) for lag in lags]
                        tau = [t for t in tau if t > 1e-9]
                        if len(tau) >= 5:
                            log_lags = np.log(list(range(2, 2 + len(tau))))
                            log_tau  = np.log(tau)
                            # 過濾 inf/nan
                            mask = np.isfinite(log_lags) & np.isfinite(log_tau)
                            if mask.sum() >= 5:
                                reg = np.polyfit(log_lags[mask], log_tau[mask], 1)
                                hurst_val = float(np.clip(reg[0], 0.01, 0.99))
                except Exception:
                    hurst_val = 0.5

            # ── [20][21] H4 佔位（由 main.py 注入）──────────────────
            # 注意：這裡先設預設值，main.py 會用實際的 H4 值覆蓋
            h4_trend = 0.0
            h4_pos   = 0.5

            # 15分K 自身趨勢強化：
            # 用最近 20 根的線性回歸斜率補充 H4 趨勢信號
            # 避免 H4 資料延遲時趨勢特徵全是 0
            if len(df) >= 20:
                try:
                    recent_closes = df["close"].tail(20).values
                    x = np.arange(len(recent_closes))
                    slope_15m = float(np.polyfit(x, recent_closes, 1)[0])
                    # 正規化：斜率除以平均價格，×100 讓值域合理
                    avg_price = float(np.mean(recent_closes))
                    trend_15m = float(np.clip(slope_15m / (avg_price + 1e-9) * 100 * 20, -3, 3))
                    # 如果 h4_trend 還是 0（還沒被注入），先用 15m 趨勢填充
                    h4_trend = trend_15m
                except Exception:
                    pass

            # ── 組裝（順序必須對齊 config.py 的 F_* 索引）────────────
            res = {
                "rsi":        float(np.clip(rsi, 0, 1)),
                "atr_norm":   float(np.clip(atr_norm * 15, 0, 1.5)), # 從 ×30 改 ×15，讓其他特徵有更多空間
                "p_pos":      float(np.clip(p_pos, -2, 3)),
                "fvg_dist":   float(np.clip(fvg_val * 100, -5, 5)),
                "is_sweep":   float(np.clip(sweep_depth, -3, 3)),     # 連續掃蕩深度
                "hour_sin":   float(hour_sin),
                "hour_cos":   float(hour_cos),
                "amd_phase":  float(np.clip(amd_phase, 0, 3.99)),  # 0~3.99 代表 AMD 四個時段
                "asia_pos":   float(np.clip(asia_pos, -1, 2)),
                "pdh_dist":   float(np.clip(pdh_dist * 100, -10, 10)),
                "pdl_dist":   float(np.clip(pdl_dist * 100, -10, 10)),
                "bos":        float(np.clip(bos, -5, 5)),             # 連續突破幅度
                "choch":      float(np.clip(choch, -5, 5)),           # 連續反轉幅度
                # ob_bull/bear = 當前價格相對最近 OB 中點的百分比距離
                # 接近 0 = 正在回踩 OB（關鍵進場訊號）
                # 乘以 30（槓桿）讓數值範圍擴大到 TCN 可感知的尺度
                "ob_bull":    float(np.clip(ob_bull * 30, -3, 3)),
                "ob_bear":    float(np.clip(ob_bear * 30, -3, 3)),
                "p_chg":      float(np.clip(p_chg, -5, 5)),
                "poc_dist":   float(np.clip(poc_dist * 100, -10, 10)),
                "eq_hl":      float(np.clip(eq_hl, -1, 1)),           # 連續相似程度
                "is_ny":      float(ny_proximity),                    # 連續 NY 距離
                "is_tue_wed": float(weekday_val),                     # 連續週間效應
                "h4_trend":     float(h4_trend),
                "h4_pos":       float(h4_pos),
                # Volume Delta 物理特徵（佔位，由 main.py 注入）
                "delta_sum":    0.0,
                "delta_ratio":  0.0,
                "trade_intensity": 0.0,
                # 市場微觀結構特徵
                "absorption":   0.0,   # 佔位，由 main.py 注入
                "hurst":        float(np.clip(hurst_val, 0.0, 1.0)),
                "vpin":         0.0,   # 佔位，由 main.py 注入
                "obi":          0.0,   # 佔位，由 main.py 注入（Order Book Imbalance）
            }

            result = pd.Series(res)
            result = result.replace([np.inf, -np.inf], 0).fillna(0)
            return result

        except Exception as e:
            return pd.Series(np.zeros(STATE_DIM))