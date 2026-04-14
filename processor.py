import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from config import STATE_DIM, STACK_SIZE, H4_LOOKBACK
from feature_factory import FeatureFactory, StateStacker


def find_pivot_high(values, lb=5):
    """
    找真正的擺動高點（左右各 lb 根都比它低）
    比 .max() 更準確，對應 Pine Script 的 ta.pivothigh
    """
    pivots = []
    for i in range(lb, len(values) - lb):
        window = list(values[i - lb:i]) + list(values[i + 1:i + lb + 1])
        if values[i] >= max(window):  # 用 >= 允許平台高點
            pivots.append((i, float(values[i])))
    return pivots


def find_pivot_low(values, lb=5):
    """找真正的擺動低點"""
    pivots = []
    for i in range(lb, len(values) - lb):
        window = list(values[i - lb:i]) + list(values[i + 1:i + lb + 1])
        if values[i] <= min(window):
            pivots.append((i, float(values[i])))
    return pivots


class DataProcessor:
    def __init__(self, num_agents: int, state_dim: int = STATE_DIM, stack_size: int = STACK_SIZE):
        self.state_dim = state_dim
        self.stack_size = stack_size
        self.num_agents = num_agents
        self.stackers = [StateStacker(state_dim, stack_size) for _ in range(num_agents)]

    def prepare_state(self, agent_idx: int, raw_candles: List, use_lstm: bool = False,
                      shared_features=None) -> Tuple[np.ndarray, Optional[pd.Series]]:
        try:
            features = shared_features.copy() if shared_features is not None else FeatureFactory.build_state(
                raw_candles)

            if features is None or (hasattr(features, "empty") and features.empty):
                raise ValueError(f"Agent {agent_idx} 特徵提取為空")

            # build_state 回傳 pd.Series，直接取 .values
            latest_feat = features.values[:self.state_dim].astype("float32")
            latest_feat = np.nan_to_num(latest_feat, nan=0.0)

            if len(latest_feat) < self.state_dim:
                latest_feat = np.pad(latest_feat, (0, self.state_dim - len(latest_feat)))

            state_stacked = self.stackers[agent_idx].update_and_get(latest_feat)

            if use_lstm:
                return state_stacked.reshape(self.stack_size, self.state_dim), features
            return state_stacked, features

        except Exception as e:
            fallback = np.zeros(self.state_dim * self.stack_size, dtype="float32")
            return (fallback.reshape(self.stack_size, self.state_dim), None) if use_lstm else (fallback, None)

    def calculate_global_smc(self, raw_candles: List) -> Dict:
        """
        SMC 計算核心 (對齊書上定義)
        - BOS  : 雙向偵測
        - CHoCH: 雙向偵測，需有反向結構確認
        - Sweep: 雙向偵測 (上方/下方流動性掃蕩)
        - Killzone: 使用 UTC 時間 (London 06-10, NY 12-15)
        """
        try:
            # 取 64 根 15分K = 16 小時歷史
            lookback = 64
            target_data = raw_candles[-lookback:] if len(raw_candles) > lookback else raw_candles

            df = pd.DataFrame(target_data, columns=["ts", "o", "h", "l", "c", "v"])
            if len(df) < 20:
                return self._fallback_smc()

            for col in ["o", "h", "l", "c", "v"]:
                df[col] = df[col].astype(float)
            price_now = df["c"].iloc[-1]

            # ----------------------------------------------------------------
            # 1. Volume Profile -> VAH / VAL
            # ----------------------------------------------------------------
            p_min, p_max = df["l"].min(), df["h"].max()
            bins = np.linspace(p_min, p_max, 31)
            df["tp"] = (df["h"] + df["l"] + df["c"]) / 3
            v_profile = df.groupby(
                pd.cut(df["tp"], bins=bins, include_lowest=True), observed=False
            )["v"].sum()

            vah, val = self._get_va_bounds_pine_style(v_profile, price_now)
            absolute_low = df["l"].min()

            # ----------------------------------------------------------------
            # 2. BOS 雙向偵測
            #    書上: 突破前一個擺動高點 = Bullish BOS (多方延續)
            #          跌破前一個擺動低點 = Bearish BOS (空方延續)
            # ----------------------------------------------------------------
            # BOS：用 Pivot 找真正的擺動高低點（對應 Pine Script ta.pivothigh）
            # 比 .max() 更準確，避免把盤整區間的每個高點都算成結構
            pivot_highs = find_pivot_high(df["h"].values, lb=5)
            pivot_lows = find_pivot_low(df["l"].values, lb=5)

            # 取最近的擺動高低點作為結構（排除最後 5 根，避免 Pivot 還沒確認）
            valid_ph = [v for i, v in pivot_highs if i < len(df) - 5]
            valid_pl = [v for i, v in pivot_lows if i < len(df) - 5]

            struct_high = float(valid_ph[-1]) if valid_ph else df["h"].iloc[-60:-1].max()
            struct_low = float(valid_pl[-1]) if valid_pl else df["l"].iloc[-60:-1].min()

            # BOS 突破幅度（連續值）
            if price_now > struct_high:
                # 真正的多方突破：輸出突破幅度（正值）
                is_bos = float((price_now - struct_high) / (struct_high + 1e-9) * 100)
            elif price_now < struct_low:
                # 真正的空方突破：輸出突破幅度（負值）
                is_bos = float((price_now - struct_low) / (struct_low + 1e-9) * 100)
            else:
                # 在結構區間內：bos = 0，不輸出假信號
                # 舊版在這裡輸出「距高低點的距離差」，讓 bos 永遠有值
                # 導致 TCN 把 bos 當成「價格位置」指標而不是「突破」指標
                # p_pos 和 poc_dist 已經負責位置資訊，bos 只負責突破
                is_bos = 0.0
            is_bos = float(np.clip(is_bos, -5, 5))

            # ----------------------------------------------------------------
            # 3. Sweep 雙向偵測
            #    書上: 破前低後收回 = Bullish Sweep (掃下方流動性，預期反彈)
            #          破前高後收回 = Bearish Sweep (掃上方流動性，預期回跌)
            # ----------------------------------------------------------------
            # 【對齊 feature_factory】Sweep 連續掃蕩深度
            prev_high = df["h"].iloc[-48:-3].max() if len(df) > 48 else df["h"].iloc[:-3].max()
            prev_low = df["l"].iloc[-48:-3].min() if len(df) > 48 else df["l"].iloc[:-3].min()
            recent_low_3 = df["l"].iloc[-3:].min()
            recent_high_3 = df["h"].iloc[-3:].max()

            is_sweep = 0.0
            if recent_low_3 < prev_low and price_now > prev_low:
                is_sweep = float((prev_low - recent_low_3) / (prev_low + 1e-9) * 100)
            elif recent_high_3 > prev_high and price_now < prev_high:
                is_sweep = float(-(recent_high_3 - prev_high) / (prev_high + 1e-9) * 100)
            is_sweep = float(np.clip(is_sweep, -3, 3))

            # ----------------------------------------------------------------
            # 4. CHoCH 雙向偵測
            #    書上: 需有結構反轉確認，不是單純的 BOS
            #    多方 CHoCH: 跌破前低後價格又突破前高 (空頭結構被打破)
            #    空方 CHoCH: 突破前高後價格又跌破前低 (多頭結構被打破)
            # ----------------------------------------------------------------
            is_choch = self._detect_choch(df, struct_high, struct_low, vah, val)

            # ----------------------------------------------------------------
            # 5. Order Block + Breaker Block（移植自 LuxAlgo OB & BB 指標）
            #
            # LuxAlgo 邏輯：
            #   OB = Swing 突破前，搜尋區間內最強的反向 K 棒（實體）
            #   BB = OB 被跌破後反轉，變成 Breaker Block
            #   BB 回踩 = 最強進場點（書上說 BB 比 OB 更強）
            #
            # 完整狀態機：
            #   等待 Swing High 突破 → 往回找多方 OB
            #   OB 被跌破 → 變成 Breaker Block
            #   Breaker Block 被回踩 → 最強做空點
            # ----------------------------------------------------------------

            # 用 Pivot 找最近的 Swing High/Low（對應 LuxAlgo 的 swings 函數）
            bull_pivots = find_pivot_high(df["h"].values, lb=5)
            bear_pivots = find_pivot_low(df["l"].values, lb=5)

            # 最近有效的 Swing（排除最後 5 根，Pivot 需要確認時間）
            last_swing_high_idx = None
            last_swing_high_val = None
            last_swing_low_idx = None
            last_swing_low_val = None

            for idx, val_p in reversed(bull_pivots):
                if idx < len(df) - 5:
                    last_swing_high_idx = idx
                    last_swing_high_val = val_p
                    break
            for idx, val_p in reversed(bear_pivots):
                if idx < len(df) - 5:
                    last_swing_low_idx = idx
                    last_swing_low_val = val_p
                    break

            # 初始化 OB/BB 狀態
            ob_bull_price = 0.0  # 多方 OB 實體頂
            ob_bull_bottom = 0.0  # 多方 OB 實體底
            ob_bear_price = 0.0  # 空方 OB 實體底
            ob_bear_top = 0.0  # 空方 OB 實體頂
            ob_bull_retrace = 0.0  # 回踩狀態（正=回踩中 負=失效）
            ob_bear_retrace = 0.0
            is_bull_breaker = False  # 是否是 Breaker Block（更強訊號）
            is_bear_breaker = False

            # ── 多方 OB 搜尋（BOS 上方突破後往回找）────────────────────
            # 對應 LuxAlgo：close > top.y 後，從現在到 top.x 之間找最低點
            if price_now > struct_high and last_swing_high_idx is not None:
                search_start = len(df) - last_swing_high_idx
                search_end = min(search_start + 50, len(df) - 1)

                best_ob_top = 0.0
                best_ob_bot = 0.0
                min_low_seen = float("inf")

                for j in range(search_start, search_end):
                    if j >= len(df):
                        break
                    c_j = df["c"].iloc[-j] if j > 0 else df["c"].iloc[-1]
                    o_j = df["o"].iloc[-j] if j > 0 else df["o"].iloc[-1]
                    h_j = df["h"].iloc[-j] if j > 0 else df["h"].iloc[-1]
                    l_j = df["l"].iloc[-j] if j > 0 else df["l"].iloc[-1]

                    body_top = max(c_j, o_j)
                    body_bot = min(c_j, o_j)

                    # LuxAlgo：在搜尋區間內找最低的那根空頭 K 棒
                    if c_j < o_j and body_bot < min_low_seen:
                        min_low_seen = body_bot
                        best_ob_top = body_top
                        best_ob_bot = body_bot

                if best_ob_top > 0:
                    ob_bull_price = best_ob_top
                    ob_bull_bottom = best_ob_bot
                    has_fvg = False
                    if search_start + 1 < len(df):
                        h_prev = df["h"].iloc[-(search_start + 1)]
                        if best_ob_bot > h_prev:
                            has_fvg = True
                    ob_strength = 1.5 if has_fvg else 1.0

                    # 判斷是否在 OB 區間內（回踩中）
                    if ob_bull_bottom <= price_now <= ob_bull_price:
                        ob_range = ob_bull_price - ob_bull_bottom + 1e-9
                        ob_bull_retrace = float(
                            (ob_bull_price - price_now) / ob_range * ob_strength
                        )
                    elif price_now < ob_bull_bottom:
                        # 跌破 OB → 變成 Breaker Block（做空訊號！）
                        ob_bull_retrace = -1.0
                        is_bear_breaker = True
                        # BB 的做空區間就是原來的多方 OB 區間
                        if ob_bear_price == 0.0:
                            ob_bear_price = ob_bull_bottom
                            ob_bear_top = ob_bull_price
                            if ob_bear_price <= price_now <= ob_bear_top:
                                ob_range = ob_bear_top - ob_bear_price + 1e-9
                                ob_bear_retrace = float(
                                    (price_now - ob_bear_price) / ob_range * 2.0
                                )  # BB 比 OB 強，強度 ×2

            # ── 空方 OB 搜尋（BOS 下方突破後往回找）────────────────────
            if price_now < struct_low and last_swing_low_idx is not None and ob_bear_price == 0.0:
                search_start = len(df) - last_swing_low_idx
                search_end = min(search_start + 50, len(df) - 1)

                best_ob_top = 0.0
                best_ob_bot = 0.0
                max_high_seen = 0.0

                for j in range(search_start, search_end):
                    if j >= len(df):
                        break
                    c_j = df["c"].iloc[-j] if j > 0 else df["c"].iloc[-1]
                    o_j = df["o"].iloc[-j] if j > 0 else df["o"].iloc[-1]
                    h_j = df["h"].iloc[-j] if j > 0 else df["h"].iloc[-1]

                    body_top = max(c_j, o_j)
                    body_bot = min(c_j, o_j)

                    # 找最高的那根多頭 K 棒
                    if c_j > o_j and body_top > max_high_seen:
                        max_high_seen = body_top
                        best_ob_top = body_top
                        best_ob_bot = body_bot

                if best_ob_top > 0:
                    ob_bear_top = best_ob_top
                    ob_bear_price = best_ob_bot
                    has_fvg = False
                    if search_start + 1 < len(df):
                        l_prev = df["l"].iloc[-(search_start + 1)]
                        if best_ob_top < l_prev:
                            has_fvg = True
                    ob_strength = 1.5 if has_fvg else 1.0

                    if ob_bear_price <= price_now <= ob_bear_top:
                        ob_range = ob_bear_top - ob_bear_price + 1e-9
                        ob_bear_retrace = float(
                            (price_now - ob_bear_price) / ob_range * ob_strength
                        )
                    elif price_now > ob_bear_top:
                        # 突破 OB 頂 → Breaker Block（做多訊號！）
                        ob_bear_retrace = -1.0
                        is_bull_breaker = True
                        if ob_bull_price == 0.0:
                            ob_bull_price = ob_bear_top
                            ob_bull_bottom = ob_bear_price
                            if ob_bull_bottom <= price_now <= ob_bull_price:
                                ob_range = ob_bull_price - ob_bull_bottom + 1e-9
                                ob_bull_retrace = float(
                                    (ob_bull_price - price_now) / ob_range * 2.0
                                )

            # ----------------------------------------------------------------
            # 6. Killzone (UTC 時間，書上明確定義)
            #    London: 06:00-10:00 UTC
            #    New York: 12:00-15:00 UTC
            # ----------------------------------------------------------------
            curr_hour_utc = pd.to_datetime(df["ts"].iloc[-1], unit="ms", utc=True).hour
            is_london = 1 if 6 <= curr_hour_utc < 10 else 0
            is_ny = 1 if 12 <= curr_hour_utc < 15 else 0
            is_kz = 1 if (is_london or is_ny) else 0

            # ----------------------------------------------------------------
            # 7. PD Ratio (Premium/Discount)
            # ----------------------------------------------------------------
            p_pos = float((price_now - val) / (vah - val + 1e-9))

            res = self._navigate_levels(price_now, vah, val, absolute_low, is_choch)
            res.update({
                "is_smc_valid": 1,
                "vah": float(vah),
                "val": float(val),
                "absolute_low": float(absolute_low),
                "is_bos": float(is_bos),  # 連續突破幅度（%）
                "is_sweep": float(is_sweep),  # 連續掃蕩深度（%）
                "is_ny": int(is_ny),
                "is_london": int(is_london),
                "is_kz": int(is_kz),
                "is_choch": float(is_choch),  # 連續反轉幅度
                "p_pos": p_pos,
                "ob_bull": ob_bull_price,
                "ob_bull_bottom": ob_bull_bottom,
                "ob_bull_retrace": ob_bull_retrace,  # 正=回踩OB 負=OB失效
                "ob_bear": ob_bear_price,
                "ob_bear_top": ob_bear_top,
                "ob_bear_retrace": ob_bear_retrace,  # 正=回踩OB 負=OB失效
                "is_bull_breaker": int(is_bull_breaker),  # 空方OB變多方BB（更強）
                "is_bear_breaker": int(is_bear_breaker),  # 多方OB變空方BB（更強）
                "struct_high": float(struct_high),
                "struct_low": float(struct_low),
            })
            return res

        except Exception as e:
            print(f"[DataProcessor Error] {e}")
            return self._fallback_smc()

    def _detect_choch(self, df, struct_high, struct_low, vah, val) -> float:
        """
        CHoCH 連續反轉幅度

        問題：舊版需要「創新低 AND 突破前高」同時成立，機率太低
        修法：三層強度輸出
          強 CHoCH（全條件）：創新低序列 + 突破前高 → 高分
          弱 CHoCH（單條件）：只有突破前高/前低 → 低分
          這樣 CHoCH 特徵不再永遠是 0
        """
        if len(df) < 20:
            return 0.0
        price_now = df["c"].iloc[-1]

        pivot_highs = find_pivot_high(df["h"].values, lb=3)
        pivot_lows = find_pivot_low(df["l"].values, lb=3)

        # ── 多方 CHoCH ───────────────────────────────────────────────
        if len(pivot_lows) >= 2:
            last_pl = pivot_lows[-1][1]
            prev_pl = pivot_lows[-2][1]
            break_dist = (price_now - struct_high) / (struct_high + 1e-9) * 100

            if last_pl < prev_pl and price_now > struct_high:
                # 強 CHoCH：創新低序列 + 突破前高
                seq_depth = (prev_pl - last_pl) / (prev_pl + 1e-9) * 100
                return float(np.clip(seq_depth + break_dist, 0, 5))
            elif price_now > struct_high and break_dist > 0.01:
                # 弱 CHoCH：只有突破前高（沒有創新低序列）
                return float(np.clip(break_dist * 0.5, 0, 2))

        # ── 空方 CHoCH ───────────────────────────────────────────────
        if len(pivot_highs) >= 2:
            last_ph = pivot_highs[-1][1]
            prev_ph = pivot_highs[-2][1]
            break_dist = (struct_low - price_now) / (struct_low + 1e-9) * 100

            if last_ph > prev_ph and price_now < struct_low:
                # 強 CHoCH：創新高序列 + 跌破前低
                seq_depth = (last_ph - prev_ph) / (prev_ph + 1e-9) * 100
                return float(np.clip(-(seq_depth + break_dist), -5, 0))
            elif price_now < struct_low and break_dist > 0.01:
                # 弱 CHoCH：只有跌破前低
                return float(np.clip(-break_dist * 0.5, -2, 0))

        return 0.0

    def _get_va_bounds_pine_style(self, v_profile: pd.Series, price_now: float) -> Tuple[float, float]:
        """Pine Script 對齊: 雙格擴張 + 70% 價值捕捉"""
        if v_profile.empty:
            return 0.0, 0.0
        try:
            total_v = v_profile.sum()
            poc_idx = v_profile.argmax()
            va_target = total_v * 0.7
            acc_v = v_profile.iloc[poc_idx]
            lo = hi = poc_idx

            while acc_v < va_target:
                above = sum(v_profile.iloc[hi + i] for i in range(1, 3) if hi + i < len(v_profile))
                below = sum(v_profile.iloc[lo - i] for i in range(1, 3) if lo - i >= 0)
                if above == 0 and below == 0:
                    break
                if above >= below:
                    hi = min(hi + 2, len(v_profile) - 1)
                    acc_v += above
                else:
                    lo = max(lo - 2, 0)
                    acc_v += below
                if lo == 0 and hi == len(v_profile) - 1:
                    break

            vah = float(v_profile.index[hi].right)
            val = float(v_profile.index[lo].left)
            # 【修正】移除 Non-Broken，VAH/VAL 固定不跟著現價移動
            # 突破時 p_pos > 1 或 < 0，AI 才能感知突破訊號
            return vah, val
        except Exception:
            return 0.0, 0.0

    def _navigate_levels(self, price, vah, val, abs_low, choch) -> Dict:
        """市場位置描述"""
        if price > vah:
            desc, target = "🔥 [2F] VAH 突破", price * 1.01
        elif price < val:
            desc, target = "💀 [B1] VAL 跌破", abs_low
        else:
            desc, target = "💤 [1F] 區間積累", vah
        return {
            "level_desc": desc,
            "next_target": float(target),
            "dist_to_target": float((target - price) / (price + 1e-9))
        }

    def calculate_h4_context(self, h4_candles: List) -> Dict:
        """
        H4 大框架分析（書上：大時區定方向）
        改用 EMA 斜率判斷趨勢，數值連續變化讓 AI 更容易學習
        原本用區間位置判斷，大部分時間都是 0.5/-0.5，AI 學不到差異
        """
        try:
            if not h4_candles or len(h4_candles) < 20:
                return {"h4_trend": 0.0, "h4_pos": 0.5}

            df = pd.DataFrame(h4_candles, columns=["ts", "o", "h", "l", "c", "v"])
            for col in ["o", "h", "l", "c", "v"]:
                df[col] = df[col].astype(float)

            price = df["c"].iloc[-1]

            # EMA 斜率判斷趨勢：fast > slow = 多方，fast < slow = 空方
            # 數值連續（不是離散的 +1/-1），AI 能感知趨勢強弱
            ema_fast = df["c"].ewm(span=5, adjust=False).mean()
            ema_slow = df["c"].ewm(span=20, adjust=False).mean()
            # h4_trend：多層趨勢信號合成，確保有足夠梯度
            #
            # 層1：EMA 方向（快線 vs 慢線）
            ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-9) * 1000

            # 層2：EMA 斜率（最近 5 根的變化方向）
            ema_slope = float((ema_fast.iloc[-1] - ema_fast.iloc[-5]) / (ema_fast.iloc[-5] + 1e-9) * 1000) if len(
                ema_fast) >= 5 else 0.0

            # 層3：價格相對 EMA 的位置（在 EMA 上方是多頭）
            price_vs_ema = float((price - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-9) * 100)

            # 合成：三層信號加權平均
            h4_trend = float(np.clip(
                ema_diff * 0.5 + ema_slope * 0.3 + price_vs_ema * 0.2,
                -3.0, 3.0
            ))

            # H4 PD Ratio：現價在 H4 最近 20 根的哪個位置
            h4_high = df["h"].iloc[-20:].max()
            h4_low = df["l"].iloc[-20:].min()
            h4_pos = float(np.clip(
                (price - h4_low) / (h4_high - h4_low + 1e-9),
                0.0, 2.0
            ))

            return {"h4_trend": h4_trend, "h4_pos": h4_pos}

        except Exception:
            return {"h4_trend": 0.0, "h4_pos": 0.5}

    def _fallback_smc(self) -> Dict:
        return {
            "level_desc": "Syncing...", "next_target": 0.0,
            "is_choch": 0, "is_bos": 0, "dist_to_target": 0.0,
            "is_smc_valid": 0, "vah": 0.0, "val": 0.0,
            "absolute_low": 0.0, "is_sweep": 0, "is_ny": 0,
            "is_london": 0, "is_kz": 0, "p_pos": 0.5,
            "ob_bull": 0.0, "ob_bull_bottom": 0.0, "ob_bull_retrace": 0.0,
            "ob_bear": 0.0, "ob_bear_top": 0.0, "ob_bear_retrace": 0.0,
            "is_bull_breaker": 0, "is_bear_breaker": 0,
            "struct_high": 0.0, "struct_low": 0.0,
        }