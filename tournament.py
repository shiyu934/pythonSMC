import os
import torch
import shutil
import numpy as np
from datetime import datetime
from typing import List, Dict


class TournamentManager:
    def __init__(self, model_dir="./models", log_dir="./logs"):
        self.model_dir = model_dir
        self.log_dir = log_dir
        # 12 小時內至少 1 筆交易 (SMC 交易頻率低，適中即可)
        self.min_trade_threshold = 1
        if not os.path.exists(model_dir): os.makedirs(model_dir)

    def calculate_fitness(self, pnl: float, drawdown: float, trade_count: int, smc_info: Dict = None) -> float:
        """
        [SMC 實戰進化評分 - 240維度適配]
        pnl: 總獲利 %
        drawdown: 最大回撤 (絕對值)
        smc_info: 包含 is_sweep, is_ny, pd_ratio 等統計資訊
        """
        # 1. 懶散判定：完全不動手的戰士直接處決
        if trade_count < self.min_trade_threshold:
            return -99999.0

        # 2. 核心公式優化：PnL / (DD^1.5 + 0.01)
        # 增加對回撤的敏感度，資工系優化：DD 只要超過 3% 權重就指數級下降
        fitness = pnl / (pow(drawdown, 1.5) + 0.01)

        # 3. 【SMC 基因加權】：這才是精華
        if smc_info:
            # A. 掃蕩獲利比 (Sweep Win Rate)
            # 如果戰士是在 is_sweep=1 時獲利，給予額外 50% 權重
            sweep_bonus = smc_info.get('sweep_pnl_ratio', 1.0)

            # B. 殺戮時刻專注度 (Killzone Focus)
            # 獎勵那些在紐約盤 (is_ny=1) 表現穩定的戰士
            kz_bonus = 1.2 if smc_info.get('ny_win_rate', 0) > 0.6 else 1.0

            fitness *= (sweep_bonus * kz_bonus)

        # 4. 負值處理：回撤越大，負分越重 (鼓勵止損)
        if pnl < 0:
            fitness = pnl * (drawdown * 10 + 1)

        return float(fitness)

    def evolve(self, results: List[Dict]):
        """執行處決與精英權重注入"""
        # 排序：高分到低分
        rankings = sorted(results, key=lambda x: x['score'], reverse=True)
        winner, loser = rankings[0], rankings[-1]

        print(f"[*] --- 鬥獸場結算 ({datetime.now().strftime('%H:%M')}) ---")
        print(f"🏆 冠軍: Agent {winner['agent_id']} | 分數: {winner['score']:.2f} | PnL: {winner['pnl']:.2%}")
        print(f"💀 處決: Agent {loser['agent_id']} | PnL: {loser['pnl']:.2%}")

        w_path = f"{self.model_dir}/agent_{winner['agent_id']}.pth"
        l_path = f"{self.model_dir}/agent_{loser['agent_id']}.pth"

        if os.path.exists(w_path):
            # 複製冠軍基因
            shutil.copy(w_path, l_path)

            # 變異強度：根據冠軍的 PnL 穩定性決定
            # 如果賺很多，變異小一點(守成)；如果大家都沒賺錢，變異大一點(突破)
            m_intensity = 0.015 if winner['pnl'] > 0.1 else 0.06
            self.apply_mutation(l_path, intensity=m_intensity)

    def apply_mutation(self, model_path: str, intensity: float = 0.05):
        """權重微調：資工系權重隨機擾動"""
        try:
            # 關鍵：這裡要處理 240 維度的權重
            sd = torch.load(model_path, map_location='cpu', weights_only=True)
            for k in sd:
                if 'weight' in k:
                    # 加入高斯噪聲，這能幫助模型跳出局部最優解
                    sd[k] += torch.randn_like(sd[k]) * intensity
            torch.save(sd, model_path)
        except Exception as e:
            print(f"[!] 變異失敗: {e}")