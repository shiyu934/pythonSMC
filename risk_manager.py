import torch
import json
import time
import numpy as np
from config import (
    LEVERAGE, FEE_LEVERAGED,
    FEE_OPEN_LEVERAGED, FEE_CLOSE_LEVERAGED,
    DEAD_THRESHOLD, LOCK_SECONDS, REBORN_COOLDOWN,
    STOP_LOSS_PCT, TRAIL_ACTIVATE, TRAIL_DRAWDOWN
)


class RiskManager:
    def __init__(self, num_agents):
        self.config_path = "config.json"
        self.load_config()
        self.stats = {
            str(i): {
                "realized_pnl":  0.0,
                "entry":         0.0,
                "pos":           0,
                "trades":        0,
                "last_action_ts": 0,
                "reborn_ts":     0,
                "max_drawdown":  0.0,
                "peak_pnl":      0.0,   # 【追蹤止盈】持倉期間的最高浮動獲利
                "trail_active":  False, # 【追蹤止盈】是否已啟動追蹤
            } for i in range(num_agents)
        }
        self.best_weights = None

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except:
            self.config = {"dead_threshold": DEAD_THRESHOLD}

    def update_pnl(self, a_id, action, price_now, arena, smc_info=None):
        """
        回傳值: (brain_score, cur_pnl, total_pnl, actual_action)

        止損止盈機制：
          止損：單筆浮動虧損 <= STOP_LOSS_PCT 強制平倉
          追蹤止盈：獲利達 TRAIL_ACTIVATE 後啟動
                    從最高點回撤 TRAIL_DRAWDOWN 觸發止盈
                    讓 AI 能跟著行情走，拿到最大利潤
        """
        s = self.stats[str(a_id)]
        now = time.time()

        # 1. 轉世冷卻
        if now - s["reborn_ts"] < REBORN_COOLDOWN:
            return 0.0, 0.0, s["realized_pnl"], 0

        # 2. 持倉鎖（防止高頻抖動）
        actual_action = action
        if s["pos"] != 0 and action != s["pos"]:
            if now - s["last_action_ts"] < LOCK_SECONDS:
                actual_action = s["pos"]

        brain_score = 0.0
        pure_money  = 0.0

        # 3. 計算浮動盈虧
        if s["pos"] != 0 and s["entry"] > 0:
            brain_score = arena.calculate_reward(s["entry"], price_now, s["pos"], smc_info)

            # 純價格損益（不含任何手續費）
            if s["pos"] == 1:
                pnl_pct = (price_now - s["entry"]) / s["entry"]
            else:
                pnl_pct = (s["entry"] - price_now) / s["entry"]

            gross_pnl  = pnl_pct * LEVERAGE * 100.0
            # 持倉中：只扣開倉費，讓 AI 感受到持倉成本
            # 平倉時：再扣平倉費，這樣 total_pnl 才不會在平倉瞬間突然跳水
            pure_money = round(gross_pnl - FEE_OPEN_LEVERAGED, 4)

            # ── 【止損】────────────────────────────────────────────────
            if pure_money <= STOP_LOSS_PCT and actual_action == s["pos"]:
                print(f"\n[STOP] Agent {a_id} 止損 "
                      f"(浮動:{pure_money:.2f}% 進場:{s['entry']:.1f} 現價:{price_now:.1f})")
                actual_action = 0

            # ── 【追蹤止盈】────────────────────────────────────────────
            elif actual_action == s["pos"]:
                if pure_money > s["peak_pnl"]:
                    s["peak_pnl"] = pure_money
                if pure_money >= TRAIL_ACTIVATE:   # 現在是 2.0%
                    s["trail_active"] = True
                if s["trail_active"] and (s["peak_pnl"] - pure_money) >= TRAIL_DRAWDOWN:  # 回撤 1.5%
                    print(f"\n[TP] Agent {a_id} 追蹤止盈 "
                          f"(峰值:{s['peak_pnl']:.2f}% 現在:{pure_money:.2f}%)")
                    actual_action = 0

        # 4. 結算（只有真正平倉或換方向才結算）
        if actual_action != s["pos"]:
            if s["pos"] != 0:
                # ── 真正的平倉（從持倉 → 觀望 或 換方向）──────────────
                # pure_money 含開倉費，平倉時再扣平倉費
                realized = round(pure_money - FEE_CLOSE_LEVERAGED, 4)
                before   = s["realized_pnl"]
                s["realized_pnl"] += realized
                s["trades"]        += 1
                print(f"\n[結算] Agent {a_id} "
                      f"浮動:{pure_money:.2f}% "
                      f"平倉費:-{FEE_CLOSE_LEVERAGED:.2f}% "
                      f"本次:{realized:.2f}% "
                      f"累計:{before:.2f}%→{s['realized_pnl']:.2f}%")
            # else: 從觀望 → 開倉，不結算，不扣任何費用

            s["pos"]            = actual_action
            s["entry"]          = float(price_now) if actual_action != 0 else 0.0
            s["last_action_ts"] = now
            s["peak_pnl"]      = 0.0
            s["trail_active"]  = False
            # 開倉時 entry 必須大於 0，否則是異常
            if actual_action != 0 and s["entry"] <= 0:
                print(f"\n[⚠️] Agent {a_id} 開倉但 entry={s['entry']}，強制觀望")
                s["pos"]   = 0
                s["entry"] = 0.0
                actual_action = 0

            if actual_action != 0:
                # 開倉瞬間：cur_pnl 顯示 -開倉費，讓前端正確呈現持倉成本
                open_cur_pnl = round(-FEE_OPEN_LEVERAGED, 4)
                total_pnl    = s["realized_pnl"] + open_cur_pnl
                return brain_score, open_cur_pnl, total_pnl, actual_action
            return brain_score, 0.0, s["realized_pnl"], actual_action

        # 生涯累計 = 已實現 + 未實現浮動（未實現不含平倉費）
        total_pnl = s["realized_pnl"] + pure_money
        return brain_score, pure_money, total_pnl, actual_action

    def check_evolution(self, a_id, agent, total_pnl):
        if total_pnl > 20.0:
            self.best_weights = {
                k: v.cpu().clone()
                for k, v in agent.model.state_dict().items()
            }

        dead_limit = self.config.get("dead_threshold", DEAD_THRESHOLD)
        if total_pnl <= dead_limit:
            print(f"\n[DEAD] Agent {a_id} 陣亡 ({total_pnl:.1f}%)，轉世中...")
            self.stats[str(a_id)] = {
                "realized_pnl": 0.0, "entry": 0.0, "pos": 0, "trades": 0,
                "last_action_ts": 0,  "reborn_ts": time.time(),
                "max_drawdown": 0.0,  "peak_pnl": 0.0, "trail_active": False
            }
            from agent import TradingAgent
            return TradingAgent(
                agent_id=a_id,
                state_dim=agent.state_dim,
                stack_size=agent.stack_size,
                action_dim=agent.action_dim,
                use_lstm=agent.use_lstm
            )
        return None