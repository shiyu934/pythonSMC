"""
teacher_backtest.py — 教師回測系統 v2

核心思想：
  不是調高門檻來「過濾」到 70% 勝率，
  而是真正找出歷史上「哪些特徵組合」在回測中勝率 >= 70%。

  做法：
  1. 遍歷所有歷史 K 棒，模擬每個進場點
  2. 統計每種特徵組合的勝率
  3. 只把勝率 >= 70% 的特徵組合的樣本寫入 DB
  4. 讓 agent 學習「這些特徵組合歷史上確實賺錢了」
"""

import asyncio
import time
import numpy as np
import sys, os
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    STATE_DIM, STACK_SIZE, TOTAL_DIM, LEVERAGE,
    FEE_RATE, STOP_LOSS_PCT, TRAIL_ACTIVATE, TRAIL_DRAWDOWN
)
from feature_factory import FeatureFactory, StateStacker
from processor import DataProcessor
from trading_env import TradingArena
from experience_db import ExperienceDB

MAX_HOLD_BARS = 50
MIN_HOLD_BARS = 2
FEE_TOTAL     = FEE_RATE * 2 * LEVERAGE * 100


class TeacherBacktest:

    def __init__(self, db: ExperienceDB):
        self.db    = db
        self.arena = TradingArena()
        self.dp    = DataProcessor(num_agents=1)
        self.stats = {
            "total_bars": 0, "long_trades": 0, "short_trades": 0,
            "win_trades": 0, "loss_trades": 0, "samples_added": 0,
        }

    # ══════════════════════════════════════════════════════════════════
    # 特徵組合分類（把連續值離散化成「條件組合」）
    # ══════════════════════════════════════════════════════════════════
    def _get_condition_key(self, feat, action: int) -> str:
        """
        把特徵向量轉成條件字串，用來統計每種條件的勝率
        例如："h4_bull|kz|bos|sweep" 代表這四個條件同時成立
        """
        h4t   = float(feat.get("h4_trend",   0.0))
        bos   = float(feat.get("bos",         0.0))
        choch = float(feat.get("choch",       0.0))
        sweep = float(feat.get("is_sweep",    0.0))
        is_ny = float(feat.get("is_ny",       0.0))
        delta = float(feat.get("delta_ratio", 0.0))
        fvg   = float(feat.get("fvg_dist",    0.0))
        ob_b  = float(feat.get("ob_bull",     0.0))
        ob_s  = float(feat.get("ob_bear",     0.0))

        parts = []

        # H4 方向
        if action == 1:
            if h4t > 0.5:    parts.append("h4_strong_bull")
            elif h4t > 0.2:  parts.append("h4_bull")
            elif h4t < -0.2: parts.append("h4_bear")   # 逆向
        else:
            if h4t < -0.5:   parts.append("h4_strong_bear")
            elif h4t < -0.2: parts.append("h4_bear")
            elif h4t > 0.2:  parts.append("h4_bull")   # 逆向

        # KZ 時段
        if is_ny > 0.5:      parts.append("kz_strong")
        elif is_ny > 0.25:   parts.append("kz")

        # BOS
        if action == 1:
            if bos > 0.5:    parts.append("bos_strong")
            elif bos > 0.2:  parts.append("bos")
        else:
            if bos < -0.5:   parts.append("bos_strong")
            elif bos < -0.2: parts.append("bos")

        # Sweep
        if action == 1 and sweep > 0.3: parts.append("sweep")
        if action == 2 and sweep < -0.3: parts.append("sweep")

        # CHoCH
        if action == 1 and choch > 0.3: parts.append("choch")
        if action == 2 and choch < -0.3: parts.append("choch")

        # Delta
        if action == 1 and delta > 0.3:  parts.append("delta_bull")
        if action == 2 and delta < -0.3: parts.append("delta_bear")

        # FVG
        if abs(fvg) > 0.05: parts.append("fvg")

        # OB 接近
        if abs(ob_b) < 1.0 or abs(ob_s) < 1.0: parts.append("near_ob")

        return "|".join(sorted(parts)) if parts else "no_condition"

    # ══════════════════════════════════════════════════════════════════
    # 分批抓取歷史 K 棒
    # ══════════════════════════════════════════════════════════════════
    async def fetch_history(
        self, exchange, symbol: str, months: int, timeframe: str
    ) -> List:
        bars_map     = {"15m": 96, "4h": 6}
        total_target = months * 30 * bars_map.get(timeframe, 96)
        start_ms     = int((time.time() - months * 30 * 86400) * 1000)
        since        = start_ms
        all_data     = []
        batch_n      = 0

        print(f"  [{timeframe}] 目標 {total_target} 根，分批抓取...")

        while len(all_data) < total_target:
            batch_n += 1
            try:
                batch = await asyncio.wait_for(
                    exchange.fetch_ohlcv(
                        symbol, timeframe,
                        since=since, limit=1499),
                    timeout=30.0)
            except Exception as e:
                print(f"  第 {batch_n} 批失敗：{e}")
                break

            if not batch:
                break
            if all_data:
                batch = [c for c in batch if c[0] > all_data[-1][0]]
            if not batch:
                break

            all_data.extend(batch)
            since = batch[-1][0] + 1
            print(f"  [{timeframe}] 第 {batch_n} 批，累計 {len(all_data)} 根")

            if batch[-1][0] >= time.time() * 1000 - 900000:
                break
            await asyncio.sleep(0.5)

        print(f"  [{timeframe}] 完成：{len(all_data)} 根")
        return all_data

    # ══════════════════════════════════════════════════════════════════
    # 模擬單筆交易
    # ══════════════════════════════════════════════════════════════════
    def _simulate_trade(
        self, candles: List, entry_idx: int,
        action: int, entry_price: float
    ) -> Optional[Tuple[float, int, float]]:

        peak_pnl     = 0.0
        trail_active = False

        for bar in range(MIN_HOLD_BARS, MAX_HOLD_BARS + 1):
            idx = entry_idx + bar
            if idx >= len(candles):
                break

            p = float(candles[idx][4])
            if action == 1:
                pnl = (p - entry_price) / entry_price * LEVERAGE * 100
            else:
                pnl = (entry_price - p) / entry_price * LEVERAGE * 100
            pnl -= FEE_TOTAL / 2

            if pnl <= STOP_LOSS_PCT:
                return (p, bar, pnl - FEE_TOTAL / 2)

            if pnl > peak_pnl:
                peak_pnl = pnl
            if pnl >= TRAIL_ACTIVATE:
                trail_active = True
            if trail_active and (peak_pnl - pnl) >= TRAIL_DRAWDOWN:
                return (p, bar, pnl - FEE_TOTAL / 2)

        eidx = min(entry_idx + MAX_HOLD_BARS, len(candles) - 1)
        p    = float(candles[eidx][4])
        if action == 1:
            pnl = (p - entry_price) / entry_price * LEVERAGE * 100
        else:
            pnl = (entry_price - p) / entry_price * LEVERAGE * 100
        return (p, MAX_HOLD_BARS, pnl - FEE_TOTAL)

    # ══════════════════════════════════════════════════════════════════
    # 核心回測：找出歷史上勝率 >= 70% 的特徵組合
    # ══════════════════════════════════════════════════════════════════
    async def run_backtest(
        self,
        raw_candles:       List,
        h4_candles:        List,
        win_rate_target:   float = 0.70,
        min_pnl_win:       float = 5.0,
        min_trades:        int   = 5,    # 每個條件組合至少出現幾次才統計
        force_short_ratio: float = 0.4,
    ) -> Dict:
        """
        第一步：遍歷所有 K 棒，記錄每個進場點的結果
        第二步：統計每種特徵組合的勝率
        第三步：只把勝率 >= win_rate_target 的樣本寫入 DB

        這樣寫入 DB 的樣本都是「歷史上確實有 70% 機率賺到的條件組合」
        """
        if len(raw_candles) < 60:
            return {"status": "not_enough_data"}

        print(f"\n[教師] 第一步：遍歷 {len(raw_candles)} 根 K 棒模擬開倉...")

        stacker    = StateStacker(STATE_DIM, STACK_SIZE)
        all_trades = []   # 所有模擬交易記錄

        # 按條件組合統計勝率
        condition_stats = defaultdict(lambda: {"wins": 0, "total": 0})

        window_start = max(50, STACK_SIZE + 10)
        total_w      = len(raw_candles) - MAX_HOLD_BARS - window_start

        for i in range(window_start, len(raw_candles) - MAX_HOLD_BARS):
            window = raw_candles[:i+1]

            try:
                feat = await asyncio.to_thread(FeatureFactory.build_state, window)
                if feat is None or feat.empty:
                    continue
            except Exception:
                continue

            h4_w = [c for c in h4_candles if c[0] <= raw_candles[i][0]]
            if len(h4_w) >= 20:
                try:
                    ctx = await asyncio.to_thread(
                        self.dp.calculate_h4_context, h4_w)
                    feat["h4_trend"] = ctx.get("h4_trend", 0.0)
                    feat["h4_pos"]   = ctx.get("h4_pos",   0.5)
                except Exception:
                    pass

            try:
                smc = await asyncio.to_thread(
                    self.dp.calculate_global_smc, window)
                feat["bos"]      = float(smc.get("is_bos",   0.0))
                feat["choch"]    = float(smc.get("is_choch", 0.0))
                feat["is_sweep"] = float(smc.get("is_sweep", 0.0))
            except Exception:
                pass

            state_vec   = stacker.update_and_get(feat.values[:STATE_DIM])
            entry_price = float(raw_candles[i][4])

            for action in [1, 2]:
                result = self._simulate_trade(
                    raw_candles, i, action, entry_price)
                if result is None:
                    continue

                exit_price, hold_bars, final_pnl = result
                is_win  = final_pnl >= min_pnl_win
                cond_key = self._get_condition_key(feat, action)

                condition_stats[cond_key]["total"] += 1
                if is_win:
                    condition_stats[cond_key]["wins"] += 1

                # 取 next_state
                eidx = min(i + hold_bars, len(raw_candles) - 1)
                try:
                    ef  = await asyncio.to_thread(
                        FeatureFactory.build_state, raw_candles[:eidx+1])
                    nst = stacker.update_and_get(
                        ef.values[:STATE_DIM]) if ef is not None else state_vec
                except Exception:
                    nst = state_vec

                smc_info = {
                    "is_bos":      float(feat.get("bos",         0.0)),
                    "is_choch":    float(feat.get("choch",        0.0)),
                    "is_sweep":    float(feat.get("is_sweep",     0.0)),
                    "is_ny":       float(feat.get("is_ny",        0.0)),
                    "h4_trend":    float(feat.get("h4_trend",     0.0)),
                    "delta_ratio": float(feat.get("delta_ratio",  0.0)),
                    "p_pos":       float(feat.get("p_pos",        0.5)),
                    "peak_pnl":    max(0.0, final_pnl),
                }

                all_trades.append({
                    "state":      state_vec.copy(),
                    "action":     action,
                    "reward":     self.arena.calculate_reward(
                                    entry_price, exit_price, action, smc_info),
                    "next_state": nst.copy(),
                    "pnl":        final_pnl,
                    "smc_info":   smc_info,
                    "entry":      entry_price,
                    "cond_key":   cond_key,
                    "is_win":     is_win,
                })

            self.stats["total_bars"] += 1
            done = i - window_start
            if total_w > 0 and done % 300 == 0 and done > 0:
                pct = done / total_w * 100
                print(f"  {pct:.0f}% ({done}/{total_w})"
                      f" 累計 {len(all_trades)} 筆模擬")

        # ── 第二步：統計每個條件組合的勝率 ─────────────────────────
        print(f"\n[教師] 第二步：分析 {len(condition_stats)} 種條件組合的勝率...")

        winning_conditions = set()
        cond_report = []

        for cond, stat in condition_stats.items():
            if stat["total"] < min_trades:
                continue
            wr = stat["wins"] / stat["total"]
            avg_pnl_wins = np.mean([
                t["pnl"] for t in all_trades
                if t["cond_key"] == cond and t["is_win"]
            ]) if stat["wins"] > 0 else 0

            cond_report.append({
                "cond":   cond,
                "wr":     wr,
                "wins":   stat["wins"],
                "total":  stat["total"],
                "avg_pnl": avg_pnl_wins,
            })

            if wr >= win_rate_target:
                winning_conditions.add(cond)

        # 印出前 10 個最高勝率的組合
        cond_report.sort(key=lambda x: x["wr"], reverse=True)
        print(f"\n  勝率最高的條件組合（前 10）：")
        for cr in cond_report[:10]:
            flag = " ✅" if cr["wr"] >= win_rate_target else ""
            print(f"    {cr['wr']*100:.0f}% ({cr['wins']}/{cr['total']})"
                  f"  PnL:{cr['avg_pnl']:.1f}%"
                  f"  [{cr['cond']}]{flag}")

        print(f"\n  達標條件（>={win_rate_target*100:.0f}%）：{len(winning_conditions)} 種")

        # ── 第三步：只收錄勝率達標的樣本 ────────────────────────────
        print(f"\n[教師] 第三步：篩選高品質樣本寫入 DB...")

        good_long  = [t for t in all_trades
                      if t["is_win"] and t["action"] == 1
                      and t["cond_key"] in winning_conditions]
        good_short = [t for t in all_trades
                      if t["is_win"] and t["action"] == 2
                      and t["cond_key"] in winning_conditions]
        bad        = [t for t in all_trades
                      if not t["is_win"]
                      and t["cond_key"] in winning_conditions]

        # 也加入「學習失敗案例」：勝率低的組合中的虧損樣本
        # 讓 agent 知道「這些條件下虧錢了」
        losing_conditions = {
            cr["cond"] for cr in cond_report
            if cr["wr"] < 0.40 and cr["total"] >= min_trades
        }
        bad_contrast = [t for t in all_trades
                        if not t["is_win"]
                        and t["cond_key"] in losing_conditions]
        bad_contrast = bad_contrast[:len(good_long + good_short)]

        all_bad = bad + bad_contrast

        total_good = len(good_long) + len(good_short)
        total_all  = total_good + len(all_bad)
        actual_wr  = total_good / total_all * 100 if total_all > 0 else 0
        avg_pnl    = (np.mean([t["pnl"] for t in good_long + good_short])
                      if total_good > 0 else 0)

        print(f"  好多：{len(good_long)}  好空：{len(good_short)}")
        print(f"  壞樣本：{len(all_bad)}")
        print(f"  樣本勝率：{actual_wr:.1f}%  平均獲利PnL：{avg_pnl:.2f}%")

        n_added = await self._write_to_db(
            good_long, good_short, all_bad, force_short_ratio)

        self.stats["samples_added"] += n_added
        self.stats["long_trades"]   += len(good_long)
        self.stats["short_trades"]  += len(good_short)
        self.stats["win_trades"]    += total_good
        self.stats["loss_trades"]   += len(all_bad)

        return {
            "status":             "ok",
            "good_long":          len(good_long),
            "good_short":         len(good_short),
            "bad":                len(all_bad),
            "samples_added":      n_added,
            "win_rate":           round(actual_wr, 1),
            "avg_pnl":            round(avg_pnl, 2),
            "winning_conditions": len(winning_conditions),
            "total_conditions":   len(cond_report),
        }

    # ══════════════════════════════════════════════════════════════════
    # 寫入 DB
    # ══════════════════════════════════════════════════════════════════
    async def _write_to_db(
        self,
        good_long: List, good_short: List,
        bad: List, short_ratio: float,
    ) -> int:
        import random

        total_good = len(good_long) + len(good_short)
        if total_good == 0 and not bad:
            return 0

        n_short = int(total_good * short_ratio)
        n_long  = total_good - n_short
        n_bad   = min(len(bad), total_good)

        to_write = (
            random.sample(good_short, min(len(good_short), n_short)) +
            random.sample(good_long,  min(len(good_long),  n_long))  +
            random.sample(bad,        min(len(bad),        n_bad))
        )
        random.shuffle(to_write)

        for s in to_write:
            self.db.add_trade(
                agent_id   = "TEACHER",
                state      = s["state"],
                action     = s["action"],
                entry      = s["entry"],
                reward     = s["reward"],
                pnl        = s["pnl"],
                cur_pnl    = s["pnl"],
                next_state = s["next_state"],
                smc_info   = s["smc_info"],
            )

        self.db.flush()
        return len(to_write)

    # ══════════════════════════════════════════════════════════════════
    # 預訓練入口
    # ══════════════════════════════════════════════════════════════════
    async def pretrain_backtest(
        self, exchange, symbol: str = "BTC/USDT", months: int = 2
    ) -> Dict:
        print(f"\n{'='*55}")
        print(f"[教師預訓練] {months} 個月歷史回測")
        print(f"  策略：找出歷史勝率 >= 70% 的特徵組合")
        print(f"{'='*55}")

        raw = await self.fetch_history(exchange, symbol, months, "15m")
        if len(raw) < 100:
            return {"status": "not_enough_data"}

        h4 = await self.fetch_history(exchange, symbol, months, "4h")

        print(f"\n[教師] 資料就緒：{len(raw)} 根 15m + {len(h4)} 根 4h")

        result = await self.run_backtest(
            raw_candles       = raw,
            h4_candles        = h4,
            win_rate_target   = 0.70,
            min_pnl_win       = 5.0,
            min_trades        = 5,
            force_short_ratio = 0.4,
        )

        print(f"\n{'='*55}")
        print(f"[教師預訓練] 完成")
        print(f"  找到 {result.get('winning_conditions',0)} 種勝率>=70%的條件組合")
        print(f"  樣本勝率：{result.get('win_rate',0)}%"
              f"  平均獲利PnL：{result.get('avg_pnl',0)}%")
        print(f"  寫入DB：{result.get('samples_added',0)} 筆")
        print(f"{'='*55}\n")
        return result

    async def daily_backtest(
        self, exchange, symbol: str = "BTC/USDT"
    ) -> Dict:
        try:
            raw = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, "15m", limit=200), timeout=20.0)
            h4  = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, "4h",  limit=100), timeout=20.0)
            if not raw or len(raw) < 100:
                return {"status": "fetch_failed"}
        except Exception as e:
            return {"status": "fetch_failed", "error": str(e)}

        return await self.run_backtest(raw, h4)

    def print_stats(self):
        s     = self.stats
        total = s["win_trades"] + s["loss_trades"]
        wr    = s["win_trades"] / total * 100 if total > 0 else 0
        print("\n[教師統計]")
        print(f"  審查 K 棒：{s['total_bars']:,} 根")
        print(f"  做多：{s['long_trades']}  做空：{s['short_trades']}")
        print(f"  勝率：{wr:.1f}%  ({s['win_trades']}/{total})")
        print(f"  寫入 DB：{s['samples_added']:,} 筆")