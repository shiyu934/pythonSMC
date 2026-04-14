"""
trading_env.py — 獎勵函數【全面重構版】

診斷問題：
  - 平均 Reward -1.037：系統處於負學習狀態
  - 68.7% 觀望：agent 學會躺平逃避懲罰
  - 勝率 31.8%：開倉幾乎全輸

根本原因：
  舊版 reward 設計讓「開倉」幾乎必然得到負分
  因為手續費懲罰 + 逆向 SMC 懲罰疊加，agent 學到「不開倉最安全」

重構原則：
  1. 獎勵信號要清晰：正確開倉明確正分，錯誤開倉明確負分
  2. 觀望懲罰要存在：在好機會觀望也是錯的
  3. SMC 加乘改為條件式：只在真正盈利時才加乘，不在虧損時疊加懲罰
  4. Reward 範圍壓縮：避免極端值讓梯度爆炸
"""

import numpy as np
from config import LEVERAGE, FEE_RATE


class TradingArena:
    def __init__(self, leverage=LEVERAGE, daily_target=15.0):
        self.leverage  = leverage
        self.target    = daily_target
        self.fee_rate  = FEE_RATE
        self.liq_limit = -1.0 / leverage

    def calculate_reward(self, entry_price, current_price, action, smc_info=None):
        """
        重構版 Reward 函數

        設計邏輯：
          持倉中：回傳「品質分」（方向對不對 + SMC 加成）
          不直接用盈虧金額，避免短期波動淹沒學習信號

        Reward 範圍目標：-5 ~ +5
        """
        try:
            action = int(action)
        except Exception:
            return 0.0

        s = smc_info if isinstance(smc_info, dict) else {}

        # ── 提取 SMC 特徵 ─────────────────────────────────────────────
        sweep      = float(s.get("is_sweep",   0.0))
        bos        = float(s.get("is_bos",     0.0))
        choch      = float(s.get("is_choch",   0.0))
        pd_ratio   = float(s.get("p_pos",      0.5))
        is_ny      = float(s.get("is_ny",      0.0))
        h4_trend   = float(s.get("h4_trend",   0.0))
        delta_r    = float(s.get("delta_ratio",0.0))
        intensity  = float(s.get("trade_intensity", 0.0))
        peak_pnl   = float(s.get("peak_pnl",   0.0))
        obi        = float(s.get("obi",         0.0))

        # ══════════════════════════════════════════════════════════════
        # 0. 觀望 reward
        # ══════════════════════════════════════════════════════════════
        if action == 0:
            # 高品質 KZ + 活躍市場觀望 → 錯過機會，給懲罰
            if is_ny > 0.5 and intensity > 0.3:
                return -0.1
            # 遠離 KZ 觀望 → 正確保守，給小獎勵
            elif is_ny < 0.2:
                return 0.05
            # 一般時段中性
            return 0.0

        # ══════════════════════════════════════════════════════════════
        # 1. 計算實際盈虧
        #
        # 手續費設計原則：
        #   開倉+平倉總費用 = 0.2%（未槓桿）= 6%（槓桿後 × 30）
        #   持倉中的 reward 要讓 agent 感受到「需要漲超過 6% 才真正獲利」
        #   否則 agent 會以為小幅波動就能賺錢
        #
        #   實作：從 pnl_pct 扣除完整手續費 (fee_rate × 2)
        #   這樣 base > 0 的條件是：實際漲幅 > 0.2%（即槓桿後 > 6%）
        # ══════════════════════════════════════════════════════════════
        if action == 1:
            pnl_pct = (current_price - entry_price) / (entry_price + 1e-9)
        else:
            pnl_pct = (entry_price - current_price) / (entry_price + 1e-9)

        # 扣除完整手續費（開倉 + 平倉）讓 base 正確反映盈虧平衡點
        total_fee = self.fee_rate * 2   # 0.2% 未槓桿，槓桿後 6%
        pnl_pct -= total_fee

        # 爆倉檢查
        if pnl_pct <= self.liq_limit:
            return -10.0

        # 基礎盈虧分
        # pnl_pct > 0 才代表真正超過手續費門檻
        # 乘以 leverage × 20 讓範圍在 -5 ~ +5
        base = float(np.clip(pnl_pct * self.leverage * 20.0, -5.0, 5.0))

        # ══════════════════════════════════════════════════════════════
        # 2. SMC 方向一致性獎懲
        # 診斷：做多 40%+ 做空 7%，需要對做空額外加成
        # 讓 TCN 學到做空也是可行策略
        # ══════════════════════════════════════════════════════════════
        smc_bonus = 0.0

        # 做空稀缺獎勵：做空且盈利時給額外加成
        # 這樣 DB 裡做空盈利樣本的 reward 更高，TCN 更願意學做空
        short_bonus = 0.3 if action == 2 and base > 0 else 0.0

        # H4 方向一致
        if action == 1 and h4_trend > 0.3:
            smc_bonus += 0.3
        elif action == 2 and h4_trend < -0.3:
            smc_bonus += 0.3 + short_bonus   # 做空加成
        elif action == 1 and h4_trend < -0.3:
            smc_bonus -= 0.5
        elif action == 2 and h4_trend > 0.3:
            smc_bonus -= 0.5

        # BOS 結構一致（降低加成，避免繼續壟斷特徵重要性）
        if action == 1 and bos > 0.3:
            smc_bonus += 0.15   # 從 0.3 降到 0.15
        elif action == 2 and bos < -0.3:
            smc_bonus += 0.15 + short_bonus

        # Sweep 流動性一致
        if action == 1 and sweep > 0.3:
            smc_bonus += 0.2
        elif action == 2 and sweep < -0.3:
            smc_bonus += 0.2

        # Delta 物理驗證
        if action == 1 and delta_r > 0.1:
            smc_bonus += 0.2
        elif action == 2 and delta_r < -0.1:
            smc_bonus += 0.2
        elif action == 1 and delta_r < -0.3:
            smc_bonus -= 0.3
        elif action == 2 and delta_r > 0.3:
            smc_bonus -= 0.3

        # KZ 時段加成
        if is_ny > 0.5:
            smc_bonus += 0.2

        # PD Ratio 位置合理
        if action == 1 and 0.1 < pd_ratio < 0.6:
            smc_bonus += 0.1
        elif action == 2 and 0.4 < pd_ratio < 0.9:
            smc_bonus += 0.1

        # ══════════════════════════════════════════════════════════════
        # 3. 趨勢持倉獎勵 + 追蹤止盈引導
        # ══════════════════════════════════════════════════════════════
        tp_bonus = 0.0

        # 趨勢持倉獎勵：順勢持倉且盈利時，給額外加成
        # 這樣 agent 學到「跟著趨勢不要急著平倉」
        if base > 0:
            if action == 1 and h4_trend > 0.3:
                # 做多且 H4 看多：趨勢持倉加成
                tp_bonus += min(0.5, h4_trend * 0.3)
            elif action == 2 and h4_trend < -0.3:
                # 做空且 H4 看空：趨勢持倉加成
                tp_bonus += min(0.5, abs(h4_trend) * 0.3)

        # 逆勢持倉懲罰：方向和 H4 相反時持倉越久越虧
        if base < 0:
            if action == 1 and h4_trend < -0.5:
                tp_bonus -= 0.3   # 逆趨勢做多且虧損
            elif action == 2 and h4_trend > 0.5:
                tp_bonus -= 0.3   # 逆趨勢做空且虧損

        # 追蹤止盈引導
        if peak_pnl > 0 and base > 0:
            if peak_pnl - base < 1.5:
                tp_bonus += 0.2   # 接近峰值，鼓勵繼續持倉
            elif peak_pnl - base >= 2.5:
                tp_bonus -= 0.3   # 從峰值大幅回落，提示該止盈

        # ══════════════════════════════════════════════════════════════
        # 4. 組合最終 Reward
        # base 決定方向性的基礎，smc_bonus 決定品質加分
        # 分開計算避免 base 虧損時 smc_bonus 反而讓虧損更多
        # ══════════════════════════════════════════════════════════════
        total = base + smc_bonus + tp_bonus

        # 盈利時 SMC 品質越好加乘越多
        if base > 0 and smc_bonus > 0:
            total = base * (1.0 + smc_bonus * 0.5) + tp_bonus

        return float(np.clip(round(total, 4), -8.0, 8.0))