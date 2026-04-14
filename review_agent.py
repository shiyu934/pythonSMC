"""
review_agent.py — 覆盤 Agent（教師網路）

設計哲學：
  每個 agent 只看「當下」的 state，開了倉就開了，不知道結果好不好。
  覆盤 Agent 不同：它事後去看「哪些開倉賺錢了，哪些虧錢了」，
  找出賺錢開倉的共同特徵，然後蒸餾給所有 agent。

工作流程：
  1. 每隔一段時間（預設 96 根 K 棒 = 一天）從 DB 撈歷史交易
  2. 分析獲利交易和虧損交易的特徵差異
  3. 產生「教學 batch」：高品質的正樣本 + 對應的負對照
  4. 把這個 batch 分發給所有 agent 學習（知識蒸餾）

這比普通的 experience replay 強的地方：
  - 普通 replay：隨機抽樣，好壞樣本混在一起
  - 覆盤：刻意選出「最值得學習的對比對」，梯度信號更清晰
"""

import sqlite3
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STATE_DIM, STACK_SIZE, TOTAL_DIM,
    F_BOS, F_SWEEP, F_CHOCH, F_H4_TREND,
    F_IS_NY, F_DELTA_RATIO, F_OBI,
    F_OB_BULL, F_OB_BEAR
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReviewAgent:
    """
    覆盤 Agent

    不是一個會開倉的 agent，而是一個「老師」：
    - 分析歷史交易找出規律
    - 產生高品質的教學樣本
    - 蒸餾知識給所有學生 agent
    """

    def __init__(self, db_path: str = "./wisdom_book.db"):
        self.db_path    = db_path
        self.review_count = 0

        # 覆盤統計
        self.stats = {
            "total_reviewed":    0,
            "good_trades":       0,
            "bad_trades":        0,
            "patterns_found":    0,
            "distill_sessions":  0,
        }

    # ══════════════════════════════════════════════════════════════════
    # 核心：覆盤分析
    # ══════════════════════════════════════════════════════════════════
    def run_review(self, lookback: int = 500) -> Dict:
        """
        覆盤最近 lookback 筆交易，找出規律

        回傳：覆盤報告（哪些特徵組合賺錢，哪些組合虧錢）
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # 只看有實際開倉的記錄（action != 0）
            rows = conn.execute("""
                SELECT state, action, reward, next_state, cur_pnl,
                       is_sweep, is_choch, is_bos, delta_sum, delta_ratio,
                       absorption, vpin, obi
                FROM experiences
                WHERE action != 0
                  AND next_state IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
            """, (lookback,)).fetchall()
        finally:
            conn.close()

        if len(rows) < 20:
            return {"status": "not_enough_data", "count": len(rows)}

        # 解析樣本
        good_trades = []   # reward > 0 的交易
        bad_trades  = []   # reward < 0 的交易

        for row in rows:
            state_str, action, reward, ns_str, cur_pnl = row[:5]
            smc_vals = row[5:]

            try:
                state = np.array(json.loads(state_str),
                                 dtype=np.float16).astype(np.float32).flatten()
                ns    = np.array(json.loads(ns_str),
                                 dtype=np.float16).astype(np.float32).flatten()
                if len(state) != TOTAL_DIM or len(ns) != TOTAL_DIM:
                    continue
            except Exception:
                continue

            sample = {
                "state":      state,
                "next_state": ns,
                "action":     int(action),
                "reward":     float(reward),
                "cur_pnl":    float(cur_pnl) if cur_pnl else 0.0,
                "feat":       state[-STATE_DIM:],  # 最新幀特徵
            }

            if reward > 0.5:
                good_trades.append(sample)
            elif reward < -0.5:
                bad_trades.append(sample)

        self.stats["total_reviewed"] += len(rows)
        self.stats["good_trades"]    += len(good_trades)
        self.stats["bad_trades"]     += len(bad_trades)

        # ── 分析好壞交易的特徵差異 ──────────────────────────────────
        report = self._analyze_patterns(good_trades, bad_trades)
        report["good_count"] = len(good_trades)
        report["bad_count"]  = len(bad_trades)
        report["status"]     = "ok"

        return report

    def _analyze_patterns(
        self, good: List[Dict], bad: List[Dict]
    ) -> Dict:
        """分析好壞交易的特徵差異，找出規律"""
        if not good or not bad:
            return {}

        feat_names = [
            "rsi","atr_norm","p_pos","fvg_dist","is_sweep",
            "hour_sin","hour_cos","amd_phase","asia_pos","pdh_dist",
            "pdl_dist","bos","choch","ob_bull","ob_bear","p_chg",
            "poc_dist","eq_hl","is_ny","is_tue_wed","h4_trend","h4_pos",
            "delta_sum","delta_ratio","trade_intensity",
            "absorption","hurst","vpin","obi"
        ]

        good_feats = np.array([s["feat"] for s in good])
        bad_feats  = np.array([s["feat"] for s in bad])

        # 每個特徵在好交易 vs 壞交易的均值差異
        good_mean = good_feats.mean(axis=0)
        bad_mean  = bad_feats.mean(axis=0)
        diff      = good_mean - bad_mean

        # 找差異最大的特徵（代表這些特徵最能區分好壞交易）
        top_idx   = np.argsort(np.abs(diff))[::-1][:8]
        patterns  = {}
        for idx in top_idx:
            if idx < len(feat_names):
                patterns[feat_names[idx]] = {
                    "good_mean": round(float(good_mean[idx]), 4),
                    "bad_mean":  round(float(bad_mean[idx]),  4),
                    "diff":      round(float(diff[idx]),      4),
                }

        self.stats["patterns_found"] += len(patterns)
        return {"patterns": patterns}

    # ══════════════════════════════════════════════════════════════════
    # 核心：產生教學 batch
    # ══════════════════════════════════════════════════════════════════
    def generate_teaching_batch(
        self, batch_size: int = 64, min_reward_good: float = 1.0,
        max_reward_bad: float = -1.0
    ) -> Optional[List[Dict]]:
        """
        產生對比教學 batch

        設計：
        - 正樣本：reward > min_reward_good 的高品質獲利交易
        - 負樣本：reward < max_reward_bad 的典型失敗交易
        - 比例：正 60% + 負 40%
        - 做空樣本強制佔 30%（解決做空比例過低問題）

        這個 batch 比隨機 replay 的信號更清晰：
        agent 能看到「賺錢的交易長什麼樣，虧錢的交易長什麼樣」
        """
        conn = sqlite3.connect(self.db_path)
        try:
            n_good  = int(batch_size * 0.6)
            n_bad   = batch_size - n_good
            n_short = int(batch_size * 0.3)   # 強制 30% 做空樣本

            # 高品質獲利樣本（reward > 1.0）
            good_rows = conn.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE action != 0
                  AND next_state IS NOT NULL
                  AND reward > ?
                ORDER BY reward DESC
                LIMIT ?
            """, (min_reward_good, n_good * 5)).fetchall()

            # 典型失敗樣本（reward < -1.0）
            bad_rows = conn.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE action != 0
                  AND next_state IS NOT NULL
                  AND reward < ?
                ORDER BY reward ASC
                LIMIT ?
            """, (max_reward_bad, n_bad * 5)).fetchall()

            # 強制做空獲利樣本
            short_rows = conn.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE action = 2
                  AND next_state IS NOT NULL
                  AND reward > 0
                ORDER BY reward DESC
                LIMIT ?
            """, (n_short * 5,)).fetchall()

        finally:
            conn.close()

        # 解析
        def _parse(rows, limit):
            result = []
            for r in rows:
                if len(result) >= limit:
                    break
                try:
                    s  = np.array(json.loads(r[0]),
                                  dtype=np.float16).astype(np.float32).flatten()
                    ns = np.array(json.loads(r[3]),
                                  dtype=np.float16).astype(np.float32).flatten()
                    if len(s) == TOTAL_DIM and len(ns) == TOTAL_DIM:
                        result.append({
                            "state":      s,
                            "action":     int(r[1]),
                            "reward":     float(r[2]),
                            "next_state": ns,
                        })
                except Exception:
                    continue
            return result

        good_s  = _parse(good_rows,  n_good)
        bad_s   = _parse(bad_rows,   n_bad)
        short_s = _parse(short_rows, n_short)

        # 把做空樣本替換進 good_s
        if short_s:
            replace_n = min(len(short_s), n_short)
            good_s    = good_s[replace_n:] + short_s[:replace_n]

        batch = good_s + bad_s
        if len(batch) < batch_size // 2:
            return None   # 樣本不足

        import random
        random.shuffle(batch)
        return batch[:batch_size]

    # ══════════════════════════════════════════════════════════════════
    # 核心：知識蒸餾
    # ══════════════════════════════════════════════════════════════════
    def distill_to_agents(self, agents: list, rounds: int = 3) -> int:
        """
        把覆盤知識蒸餾給所有 agent

        每個 agent 用覆盤 batch 學習 rounds 輪
        覆盤 batch 的樣本比 replay 的信號更清晰，
        所以 rounds=3 相當於普通學習的 5-10 輪效果

        回傳：成功學習的 agent 數量
        """
        teaching_batch = self.generate_teaching_batch(batch_size=64)
        if teaching_batch is None:
            return 0

        success = 0
        for agent in agents:
            try:
                for _ in range(rounds):
                    agent.learn_from_wisdom(teaching_batch)
                success += 1
            except Exception as e:
                print(f"[覆盤] Agent {agent.agent_id} 蒸餾失敗：{e}")

        self.stats["distill_sessions"] += 1
        self.review_count += 1
        return success

    def print_report(self, report: Dict):
        """印出覆盤報告"""
        if report.get("status") != "ok":
            print(f"[覆盤] 資料不足，跳過（{report.get('count', 0)} 筆）")
            return

        print(f"\n{'─'*50}")
        print(f"[覆盤] 第 {self.review_count} 次 | "
              f"好交易 {report['good_count']} 筆 | "
              f"壞交易 {report['bad_count']} 筆")

        patterns = report.get("patterns", {})
        if patterns:
            print("  最能區分好壞的特徵：")
            for feat, vals in list(patterns.items())[:5]:
                direction = "↑好" if vals["diff"] > 0 else "↓好"
                print(f"    {feat:>18}: "
                      f"好={vals['good_mean']:>7.3f} "
                      f"壞={vals['bad_mean']:>7.3f} "
                      f"差={vals['diff']:>+7.3f} {direction}")

        print(f"  累計：審查 {self.stats['total_reviewed']:,} 筆 | "
              f"蒸餾 {self.stats['distill_sessions']} 次")
        print(f"{'─'*50}")

    def get_stats(self) -> Dict:
        return dict(self.stats)
