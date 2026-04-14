"""
agent_consciousness.py — Agent 自我意識系統

三個核心模組：

1. MarketRegimeDetector（市場狀態感知）
   識別當前市場是：趨勢、盤整、高波動
   不同狀態用不同策略

2. ConfidenceEvaluator（自信心評估）
   agent 對自己的開倉決策打分
   分數低於門檻時拒絕開倉（知道自己不確定時不賭）

3. SelfReflection（自我反省）
   每次平倉後回顧這筆交易
   更新自己在不同市場狀態下的歷史勝率
   讓 agent 知道自己的強項和弱項
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple
import time


# ══════════════════════════════════════════════════════════════════════
# 1. 市場狀態感知
# ══════════════════════════════════════════════════════════════════════
class MarketRegimeDetector:
    """
    識別當前市場狀態：
      TREND_UP   : 明確上升趨勢（做多為主）
      TREND_DOWN : 明確下降趨勢（做空為主）
      RANGING    : 盤整（等待突破）
      VOLATILE   : 高波動（謹慎）
    """

    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGING    = "RANGING"
    VOLATILE   = "VOLATILE"

    def detect(self, feat: dict) -> Tuple[str, float]:
        """
        回傳 (市場狀態, 強度 0~1)

        趨勢識別邏輯（三票決定）：
          票1：H4 EMA 斜率方向
          票2：Hurst 指數（>0.5 趨勢延續）
          票3：BOS 方向（結構突破確認趨勢）
        至少 2/3 投票一致才算趨勢
        """
        h4t   = float(feat.get("h4_trend",   0.0))
        hurst = float(feat.get("hurst",       0.5))
        atr   = float(feat.get("atr_norm",    0.0))
        bos   = float(feat.get("bos",         0.0))
        p_chg = abs(float(feat.get("p_chg",   0.0)))
        pdh   = float(feat.get("pdh_dist",    0.0))
        pdl   = float(feat.get("pdl_dist",    0.0))

        # 高波動優先判斷
        if atr > 1.0 or p_chg > 3.0:
            strength = min(1.0, (atr + p_chg / 3) / 2)
            return self.VOLATILE, round(strength, 3)

        # 三票制趨勢判斷
        bull_votes = 0
        bear_votes = 0

        # 票1：H4 方向
        if h4t > 0.2:   bull_votes += 1
        elif h4t < -0.2: bear_votes += 1

        # 票2：Hurst + 方向
        if hurst > 0.52 and h4t > 0:  bull_votes += 1
        elif hurst > 0.52 and h4t < 0: bear_votes += 1

        # 票3：BOS 方向（結構確認）
        if bos > 0.2:   bull_votes += 1
        elif bos < -0.2: bear_votes += 1

        # 票4：前日高低點位置（在前日高點上方=多頭）
        if pdh < 0 and pdl < 0:   bull_votes += 1   # 突破前日高點
        elif pdh > 0 and pdl > 0:  bear_votes += 1   # 跌破前日低點

        # 判定
        if bull_votes >= 3:
            strength = min(1.0, bull_votes / 4 * abs(h4t) + 0.3)
            return self.TREND_UP, round(strength, 3)

        if bear_votes >= 3:
            strength = min(1.0, bear_votes / 4 * abs(h4t) + 0.3)
            return self.TREND_DOWN, round(strength, 3)

        if bull_votes == 2:
            return self.TREND_UP, 0.4
        if bear_votes == 2:
            return self.TREND_DOWN, 0.4

        # 盤整
        ranging_strength = min(1.0, (1 - hurst) * 2 + 0.2)
        return self.RANGING, round(ranging_strength, 3)


# ══════════════════════════════════════════════════════════════════════
# 2. 自信心評估
# ══════════════════════════════════════════════════════════════════════
class ConfidenceEvaluator:
    """
    agent 對自己的開倉決策評分

    評分依據：
    1. Q 值差距：最高 Q 值和第二高的差距（差距大 = 更確定）
    2. 歷史在此市場狀態的勝率（過去做得好 = 更有信心）
    3. 特徵信號強度（信號越強越有信心）

    低於 min_confidence 時拒絕開倉
    """

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        # 歷史勝率記憶（按市場狀態分類）
        self._regime_history: Dict[str, deque] = {
            "TREND_UP":   deque(maxlen=50),
            "TREND_DOWN": deque(maxlen=50),
            "RANGING":    deque(maxlen=50),
            "VOLATILE":   deque(maxlen=50),
        }

    def evaluate(
        self,
        q_values:     np.ndarray,
        action:       int,
        feat:         dict,
        regime:       str,
        regime_strength: float,
    ) -> Tuple[float, bool, str]:
        """
        回傳 (信心分數 0~1, 是否允許開倉, 原因說明)
        """
        if action == 0:
            return 1.0, True, "觀望不需信心評估"

        # ── 分量 1：Q 值確定性（0~0.4）─────────────────────────────
        q_sorted    = np.sort(q_values.flatten())[::-1]
        q_gap       = float(q_sorted[0] - q_sorted[1]) if len(q_sorted) > 1 else 0
        q_conf      = min(0.4, q_gap * 0.4)   # Q 差距越大越確定

        # ── 分量 2：歷史勝率（0~0.3）────────────────────────────────
        hist = self._regime_history.get(regime, deque())
        if len(hist) >= 10:
            hist_wr   = sum(hist) / len(hist)
            hist_conf = hist_wr * 0.3
        else:
            hist_conf = 0.15   # 樣本不足時給中性分數

        # ── 分量 3：特徵信號強度（0~0.3）────────────────────────────
        h4t   = float(feat.get("h4_trend",   0.0))
        bos   = float(feat.get("bos",         0.0))
        is_ny = float(feat.get("is_ny",       0.0))
        delta = float(feat.get("delta_ratio", 0.0))

        signal_strength = 0.0
        if action == 1:
            signal_strength += min(0.1, max(0, h4t) * 0.1)
            signal_strength += min(0.1, max(0, bos)  * 0.1)
            signal_strength += min(0.05, is_ny * 0.05)
            signal_strength += min(0.05, max(0, delta) * 0.05)
        else:
            signal_strength += min(0.1, max(0, -h4t) * 0.1)
            signal_strength += min(0.1, max(0, -bos)  * 0.1)
            signal_strength += min(0.05, is_ny * 0.05)
            signal_strength += min(0.05, max(0, -delta) * 0.05)

        # ── 市場狀態適配加成/扣分 ────────────────────────────────────
        regime_bonus = 0.0
        if regime == "VOLATILE":
            regime_bonus = -0.1   # 高波動扣分，謹慎
        elif regime in ("TREND_UP", "TREND_DOWN"):
            if (regime == "TREND_UP"   and action == 1) or \
               (regime == "TREND_DOWN" and action == 2):
                regime_bonus = regime_strength * 0.1   # 順勢加分

        total = q_conf + hist_conf + signal_strength + regime_bonus
        total = float(np.clip(total, 0.0, 1.0))

        allowed = total >= self.min_confidence
        reason  = (
            f"Q差={q_gap:.2f}({q_conf:.2f})"
            f" 歷史={hist_conf:.2f}"
            f" 信號={signal_strength:.2f}"
            f" 市場={regime_bonus:+.2f}"
            f" → 總分={total:.2f}"
            f" {'✅' if allowed else '❌'}"
        )

        return total, allowed, reason

    def record_result(self, regime: str, is_win: bool):
        """平倉後記錄結果，更新歷史勝率"""
        if regime in self._regime_history:
            self._regime_history[regime].append(1.0 if is_win else 0.0)

    def get_regime_winrates(self) -> Dict[str, float]:
        """取得各市場狀態的歷史勝率"""
        result = {}
        for regime, hist in self._regime_history.items():
            if len(hist) >= 5:
                result[regime] = round(sum(hist) / len(hist) * 100, 1)
            else:
                result[regime] = None   # 樣本不足
        return result


# ══════════════════════════════════════════════════════════════════════
# 3. 自我反省
# ══════════════════════════════════════════════════════════════════════
class SelfReflection:
    """
    每次平倉後回顧這筆交易，更新自我認知

    記錄：
    - 開倉時的市場狀態
    - 開倉時的信心分數
    - 最終 PnL
    - 是否符合 SMC 條件

    讓 agent 能說出：
    「我在趨勢市場的勝率是 68%，盤整市場只有 31%」
    「當我信心分數 > 0.7 時勝率 74%，< 0.4 時只有 28%」
    """

    def __init__(self, agent_id: str):
        self.agent_id  = agent_id
        self.trades    = deque(maxlen=200)   # 最近 200 筆
        self.total_pnl = 0.0

    def record_open(
        self,
        action:          int,
        entry_price:     float,
        regime:          str,
        regime_strength: float,
        confidence:      float,
        feat:            dict,
    ) -> str:
        """開倉時記錄，回傳這筆交易的 ID"""
        trade_id = f"{self.agent_id}_{int(time.time()*1000)}"
        self.trades.append({
            "id":           trade_id,
            "action":       action,
            "entry":        entry_price,
            "regime":       regime,
            "strength":     regime_strength,
            "confidence":   confidence,
            "h4_trend":     float(feat.get("h4_trend",   0.0)),
            "bos":          float(feat.get("bos",         0.0)),
            "is_ny":        float(feat.get("is_ny",       0.0)),
            "open_time":    time.time(),
            "close_pnl":    None,   # 平倉後填入
            "is_win":       None,
        })
        return trade_id

    def record_close(self, final_pnl: float, min_win_pnl: float = 1.0):
        """平倉時更新最後一筆記錄"""
        if self.trades and self.trades[-1]["close_pnl"] is None:
            self.trades[-1]["close_pnl"] = final_pnl
            self.trades[-1]["is_win"]    = final_pnl >= min_win_pnl
            self.total_pnl += final_pnl

    def get_self_assessment(self) -> Dict:
        """
        自我評估報告：
        我在什麼情況下表現好？什麼情況下表現差？
        """
        closed = [t for t in self.trades if t["close_pnl"] is not None]
        if len(closed) < 5:
            return {"status": "樣本不足"}

        # 按市場狀態統計
        regime_stats = {}
        for t in closed:
            r = t["regime"]
            if r not in regime_stats:
                regime_stats[r] = {"wins": 0, "total": 0, "pnl": []}
            regime_stats[r]["total"] += 1
            if t["is_win"]:
                regime_stats[r]["wins"] += 1
            regime_stats[r]["pnl"].append(t["close_pnl"])

        regime_report = {}
        for r, s in regime_stats.items():
            regime_report[r] = {
                "win_rate": round(s["wins"] / s["total"] * 100, 1),
                "avg_pnl":  round(np.mean(s["pnl"]), 2),
                "count":    s["total"],
            }

        # 按信心分數統計
        high_conf = [t for t in closed if t["confidence"] >= 0.6]
        low_conf  = [t for t in closed if t["confidence"] <  0.4]

        conf_report = {}
        if high_conf:
            conf_report["high_conf_wr"] = round(
                sum(1 for t in high_conf if t["is_win"]) / len(high_conf) * 100, 1)
        if low_conf:
            conf_report["low_conf_wr"] = round(
                sum(1 for t in low_conf if t["is_win"]) / len(low_conf) * 100, 1)

        # 最強和最弱的市場狀態
        best_regime  = max(regime_report.items(),
                           key=lambda x: x[1]["win_rate"]) if regime_report else None
        worst_regime = min(regime_report.items(),
                           key=lambda x: x[1]["win_rate"]) if regime_report else None

        total_closed = len(closed)
        overall_wr   = sum(1 for t in closed if t["is_win"]) / total_closed * 100

        return {
            "status":        "ok",
            "agent_id":      self.agent_id,
            "total_trades":  total_closed,
            "overall_wr":    round(overall_wr, 1),
            "total_pnl":     round(self.total_pnl, 2),
            "regime_stats":  regime_report,
            "conf_stats":    conf_report,
            "best_regime":   best_regime,
            "worst_regime":  worst_regime,
        }

    def print_assessment(self):
        a = self.get_self_assessment()
        if a.get("status") != "ok":
            return

        print(f"\n  [自我評估] Agent {a['agent_id']}")
        print(f"    整體勝率：{a['overall_wr']}%  累計PnL：{a['total_pnl']}%")

        print(f"    各市場狀態：")
        for regime, stat in a.get("regime_stats", {}).items():
            bar = "█" * int(stat["win_rate"] / 10)
            print(f"      {regime:12}: {stat['win_rate']:>5.1f}% "
                  f"({stat['count']}筆) {bar}")

        conf = a.get("conf_stats", {})
        if "high_conf_wr" in conf:
            print(f"    高信心(>0.6)勝率：{conf['high_conf_wr']}%")
        if "low_conf_wr" in conf:
            print(f"    低信心(<0.4)勝率：{conf['low_conf_wr']}%")

        if a.get("best_regime"):
            r, s = a["best_regime"]
            print(f"    強項：{r} ({s['win_rate']}%)")
        if a.get("worst_regime"):
            r, s = a["worst_regime"]
            print(f"    弱項：{r} ({s['win_rate']}%)")


# ══════════════════════════════════════════════════════════════════════
# 整合入口：AgentConsciousness
# ══════════════════════════════════════════════════════════════════════
class AgentConsciousness:
    """
    三個模組的統一入口
    每個 agent 擁有自己的意識實例
    """

    def __init__(self, agent_id: str, min_confidence: float = 0.35):
        self.agent_id   = agent_id
        self.detector   = MarketRegimeDetector()
        self.evaluator  = ConfidenceEvaluator(min_confidence)
        self.reflection = SelfReflection(agent_id)

        self._last_regime    = MarketRegimeDetector.RANGING
        self._last_strength  = 0.5
        self._last_confidence= 0.5

    def pre_action(
        self,
        q_values: np.ndarray,
        action:   int,
        feat:     dict,
        verbose:  bool = False,
    ) -> Tuple[int, float, str]:
        """
        開倉前評估：
        1. 識別市場狀態
        2. 評估自信心
        3. 決定是否允許開倉

        回傳 (最終action, 信心分數, 說明)
        """
        # 市場狀態感知
        regime, strength = self.detector.detect(feat)
        self._last_regime   = regime
        self._last_strength = strength

        if action == 0:
            return 0, 1.0, "觀望"

        # 自信心評估
        conf, allowed, reason = self.evaluator.evaluate(
            q_values, action, feat, regime, strength
        )
        self._last_confidence = conf

        if not allowed:
            if verbose:
                print(f"  [意識] A{self.agent_id} 信心不足，放棄開倉 {reason}")
            return 0, conf, reason

        if verbose:
            print(f"  [意識] A{self.agent_id} 允許開倉 {reason}")

        return action, conf, reason

    def post_trade(self, final_pnl: float, entry_price: float, min_win: float = 1.0):
        """
        平倉後反省：
        1. 記錄交易結果
        2. 更新市場狀態勝率
        """
        is_win = final_pnl >= min_win
        self.reflection.record_close(final_pnl, min_win)
        self.evaluator.record_result(self._last_regime, is_win)

    def open_trade(self, action: int, entry_price: float, feat: dict):
        """開倉時記錄"""
        self.reflection.record_open(
            action, entry_price,
            self._last_regime,
            self._last_strength,
            self._last_confidence,
            feat,
        )

    def get_status(self) -> Dict:
        """取得當前意識狀態摘要"""
        assessment = self.reflection.get_self_assessment()
        regime_wrs = self.evaluator.get_regime_winrates()
        return {
            "agent_id":   self.agent_id,
            "regime":     self._last_regime,
            "strength":   self._last_strength,
            "confidence": self._last_confidence,
            "regime_wrs": regime_wrs,
            "assessment": assessment,
        }