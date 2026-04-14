# position_manager.py
import os
from datetime import datetime, timedelta
from agent import TradingAgent


class PositionManager:
    def __init__(self, num_agents, mdd_limit=-5.0, dead_threshold=-10.0, cooldown=300):
        self.mdd_limit = mdd_limit
        self.dead_threshold = dead_threshold
        self.cooldown = cooldown

        # 狀態紀錄
        self.stats = {str(i): {"pnl": 0.0, "trades": 0} for i in range(num_agents)}
        self.locks = {
            str(i): {
                "current_action": 0,
                "last_change_time": datetime.now() - timedelta(minutes=10)
            } for i in range(num_agents)
        }
        self.entry_prices = {str(i): 0.0 for i in range(num_agents)}

    def _get_safe_pnl(self, entry, now_price, action, arena):
        """【資工保險栓】：強制所有損益計算最高虧損 98%，防止數據噴出 -380%"""
        if action == 0 or entry == 0:
            return 0.0
        raw_pnl = arena.calculate_reward(entry, now_price, action)
        # max(pnl, -98.0) 代表如果賠了 150%，它會被拉回到 -98%
        return max(raw_pnl, -98.0)

    def check_and_execute(self, a_id, agent, price_now, proposed_action, arena, state_dim):
        a_id = str(a_id)
        lock = self.locks[a_id]
        now = datetime.now()
        time_diff = (now - lock["last_change_time"]).total_seconds()

        # 1. 計算目前的損益 (使用保險栓)
        floating_pnl = self._get_safe_pnl(self.entry_prices[a_id], price_now, lock["current_action"], arena)
        total_pnl = self.stats[a_id]["pnl"] + floating_pnl

        # --- 【優先權 1：數位處決】 ---
        # 如果總分低於死線，立刻清除所有數據並重生
        if total_pnl <= self.dead_threshold:
            print(f"\n[💀 EXECUTION] Agent {a_id} 總分達死線 ({total_pnl:.1f}%)，斬首轉世。")
            self._rebirth(a_id)
            # 回傳新 Agent 實體
            return 0, "🆕NEW", TradingAgent(agent_id=a_id, state_dim=state_dim)

        # --- 【優先權 2：鋼鐵止損】 ---
        # 單筆虧損達 5% 立刻結算並冷靜
        if floating_pnl <= self.mdd_limit and lock["current_action"] != 0:
            self.stats[a_id]["pnl"] += floating_pnl
            self.stats[a_id]["trades"] += 1
            self.entry_prices[a_id] = 0.0
            lock["current_action"] = 0
            lock["last_change_time"] = now
            return 0, "🛑STOP", None

        # --- 【優先權 3：5 分鐘持倉鎖定】 ---
        if proposed_action != lock["current_action"] and time_diff < self.cooldown:
            return lock["current_action"], "⏳LOCK", None

        # --- 【優先權 4：正常交易處理】 ---
        if proposed_action != lock["current_action"]:
            if lock["current_action"] != 0:  # 結算上一單
                # 結算時也要過濾保險栓！
                realized = self._get_safe_pnl(self.entry_prices[a_id], price_now, lock["current_action"], arena)
                self.stats[a_id]["pnl"] += realized
                self.stats[a_id]["trades"] += 1

            self.entry_prices[a_id] = price_now if proposed_action != 0 else 0.0
            lock["current_action"] = proposed_action
            lock["last_change_time"] = now
            return proposed_action, f"⚡ACT-{proposed_action}", None

        return proposed_action, "HOLD" if lock["current_action"] != 0 else "IDLE", None

    def _rebirth(self, a_id):
        """內部轉世邏輯"""
        model_path = f"./models/agent_{a_id}.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
        # 重置所有變數
        self.stats[a_id] = {"pnl": 0.0, "trades": 0}
        self.entry_prices[a_id] = 0.0
        self.locks[a_id]["current_action"] = 0
        self.locks[a_id]["last_change_time"] = datetime.now()