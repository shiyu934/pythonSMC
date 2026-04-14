import asyncio
import os
import sys
import gc
import subprocess
import traceback
from datetime import datetime

# --- 1. CSIE 環境優化 (減少競爭，確保穩定性) ---
from agent_consciousness import AgentConsciousness
from review_agent import ReviewAgent
from teacher_backtest import TeacherBacktest

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()

import torch
import numpy as np
from config import STATE_DIM, STACK_SIZE, TOTAL_DIM, NUM_AGENTS, LEVERAGE
from data_engine import DataEngine
from physic_engine import PhysicEngine
from risk_manager import RiskManager
from processor import DataProcessor
from experience_db import ExperienceDB
from trading_env import TradingArena
from agent import TradingAgent
from feature_factory import FeatureFactory

GLOBAL_WARRIORS = []

def start_data_bridge():
    """喚醒數據轉運引擎 (bridge.py)"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bridge_path = os.path.join(current_dir, "bridge.py")
        print(f"[*] 正在背景喚醒數據轉運引擎...")
        subprocess.Popen([sys.executable, bridge_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=False)
        print("[✅] Bridge 指令已送出。")
    except Exception as e:
        print(f"[⚠️] Bridge 啟動異常: {e}")

async def main_loop():
    global GLOBAL_WARRIORS

    # --- 2. 維度設定（從 config.py 統一管理）---
    num_agents = NUM_AGENTS
    state_dim  = STATE_DIM
    stack_size = STACK_SIZE
    total_dim  = TOTAL_DIM

    start_data_bridge()
    await asyncio.sleep(2)

    # 背景定時更新 H4 數據（每 15 分鐘一次）
    async def h4_refresh_loop():
        while True:
            await asyncio.sleep(900)
            await engine.refresh_h4()
    asyncio.create_task(h4_refresh_loop())

    db = ExperienceDB()
    rm = RiskManager(num_agents)
    dp = DataProcessor(num_agents, state_dim=state_dim, stack_size=stack_size)
    arena = TradingArena(leverage=LEVERAGE)

    print(f"[*] 正在召喚 16 位混合戰士大腦 (CSIE {total_dim}-Neural Mode)...")
    agents = []
    for i in range(num_agents):
        is_lstm = True if i >= 8 else False
        a = TradingAgent(agent_id=str(i), state_dim=state_dim, stack_size=stack_size, use_lstm=is_lstm)
        agents.append(a)
    GLOBAL_WARRIORS = agents

    engine = DataEngine(symbol='BTC/USDT', timeframe='15m')
    physic = PhysicEngine(symbol="BTC/USDT:USDT")  # CCXT 合約統一格式
    last_states  = [None] * num_agents
    last_actions = [0] * num_agents
    candle_count = 0

    # 每個 agent 的自我意識實例
    consciousness = {
        str(i): AgentConsciousness(str(i), min_confidence=0.30)
        for i in range(num_agents)
    }
    print(f"[*] {num_agents} 個 Agent 意識模組已啟動")

    # 覆盤 Agent：每 48 根 K 棒（12 小時）覆盤一次
    reviewer         = ReviewAgent(db_path=db.db_path)
    teacher          = TeacherBacktest(db=db)
    REVIEW_INTERVAL  = 48
    BACKTEST_INTERVAL = 192   # 每 192 根（約 2 天）跑一次回測

    # --- 3. 數據連線 ---
    print(f"[*] 正在向交易所請求數據...")
    try:
        await physic.start()   # 啟動 AggTrade 訂閱（背景）
        if not await asyncio.wait_for(engine.connect(), timeout=20.0):
            print("[🚨] 數據抓取失敗。")
            return
    except:
        print("[🚨] 連線超時！")
        return

    full_history = list(engine.raw_data)
    full_h4 = list(engine.h4_raw_data)

    # ── 聖盾條件診斷計數器 ────────────────────────────────────────────
    diag = {
        "total":      0,
        "h4_clear":   0,   # H4 方向明確
        "h4_pos_ok":  0,   # H4 位置合理
        "in_kz":      0,   # 在 Killzone
        "has_sweep":  0,   # 有 Sweep
        "has_struct": 0,   # 有 BOS/CHoCH
        "on_ob":      0,   # 正在回踩 OB
        "all_pass":   0,   # 全部條件通過（理論開倉機會）
        "opened":     0,   # 實際開倉次數
    }
    diag_interval = 96    # 每 96 根（約一天）印一次報告
    try:
        db.conn.execute("PRAGMA busy_timeout = 5000;")
        # 【暴力校準】：啟動時自動清理那些「Action=0 但 PnL 亂跳」的垃圾紀錄
        db.conn.execute("DELETE FROM experiences WHERE action = 0 AND ABS(pnl) > 0.0001")
        for candle in full_history:
            db.add_ohlcv(engine.symbol, engine.timeframe, candle)
        db.conn.commit()
    except Exception as e:
        print(f"[⚠️] DB 初始化異常: {e}")

    # --- 4. 預熱階段 ---
    print(f"[*] 正在執行 SMC 特徵預算與 LSTM 序列同步...")
    warmup_depth = stack_size + 2
    pre_calc_steps = []
    for step in range(-warmup_depth, 0):
        current_slice = full_history if step == -1 else full_history[:step]
        feat = await asyncio.to_thread(FeatureFactory.build_state, current_slice)
        pre_calc_steps.append(feat)
        print(f"    [+] 預熱進度: {step + warmup_depth + 1}/{warmup_depth}", end='\r')

    print(f"\n[*] 正在同步全體戰士大腦...")
    for step_idx, step_feats in enumerate(pre_calc_steps):
        for i in range(num_agents):
            state_val, _ = dp.prepare_state(i, [], use_lstm=(i >= 8), shared_features=step_feats)
            if step_idx == len(pre_calc_steps) - 1:
                last_states[i] = state_val

    print(f"[✅] 全系統就緒！")

    # --- 5. 教師預訓練（啟動時執行一次，學習 2 個月歷史）---
    print(f"[*] 教師預訓練中（2個月歷史回測，請稍候）...")
    try:
        pretrain_result = await teacher.pretrain_backtest(
            exchange = engine.exchange,
            symbol   = engine.symbol,
            months   = 2,
        )
        if pretrain_result.get("status") == "ok":
            added = pretrain_result.get("samples_added", 0)
            print(f"[✅] 教師預訓練完成，{added:,} 筆歷史樣本已寫入 DB")
            # 預訓練結束後讓所有 agent 立刻學習一輪
            print(f"[*] 全體 agent 吸收預訓練知識...")
            for _agent in agents:
                _samples = db.sample_high_quality(batch_size=64, target_dim=total_dim)
                if _samples:
                    for _ in range(5):   # 5 輪加速吸收
                        _agent.learn_from_wisdom(_samples)
            print(f"[✅] 知識吸收完成，啟動實時獵取模式...")
        else:
            print(f"[!] 教師預訓練跳過（{pretrain_result.get('status')}），直接啟動")
    except Exception as _e:
        print(f"[!] 教師預訓練失敗：{_e}，直接啟動")

    # --- 6. 實時核心循環 ---
    try:
        async for raw_candles, h4_candles in engine.get_latest_stream():
            now_ts = datetime.now()
            price_now = raw_candles[-1][4]
            rm.load_config()

            global_smc_feats = await asyncio.to_thread(dp.calculate_global_smc, raw_candles)

            # H4 大框架分析（書上：大時區定方向）
            h4_context = await asyncio.to_thread(dp.calculate_h4_context, h4_candles)
            global_smc_feats.update(h4_context)   # 注入 h4_trend, h4_pos

            # ── 聖盾條件診斷（對齊 agent.py 的實際條件）────────────
            diag["total"] += 1
            _h4t = h4_context.get("h4_trend", 0.0)
            _h4p = h4_context.get("h4_pos", 0.5)
            _sw  = global_smc_feats.get("is_sweep", 0.0)
            _bos = global_smc_feats.get("is_bos", 0.0)
            _choch = global_smc_feats.get("is_choch", 0.0)
            _kz  = global_smc_feats.get("is_kz", 0)
            _ob_bull_ret = global_smc_feats.get("ob_bull_retrace", 0.0)
            _ob_bear_ret = global_smc_feats.get("ob_bear_retrace", 0.0)

            # 完全對齊 agent.py select_action 的實際條件
            _c1 = abs(_h4t) > 0.3                          # H4 方向明確
            _c2 = not (_h4p > 0.75 or _h4p < 0.25)        # H4 位置合理（不在極端區）
            _c3 = _kz == 1                                  # Killzone
            _c4 = abs(_sw) > 0.3                            # 有 Sweep
            # C5：結構確認要有方向，不能只看絕對值
            # 診斷發現 bos=-1.678 也算通過，造成誤導
            # 現在分開記錄：多方結構 / 空方結構
            _c5_bull = (_bos > 0.3) or (_choch > 0.3)    # 多方結構
            _c5_bear = (_bos < -0.3) or (_choch < -0.3)  # 空方結構
            _c5 = _c5_bull or _c5_bear                    # 任一方向有結構確認
            # C6：用 processor 的 ob_retrace（此時 shared_feat 還未賦值）
            _ob_bull_dist = float(global_smc_feats.get("ob_bull_retrace", 0.0))
            _ob_bear_dist = float(global_smc_feats.get("ob_bear_retrace", 0.0))
            _c6 = _ob_bull_dist > 0 or _ob_bear_dist > 0

            if _c1: diag["h4_clear"]   += 1
            if _c2: diag["h4_pos_ok"]  += 1
            if _c3: diag["in_kz"]      += 1
            if _c4: diag["has_sweep"]  += 1
            if _c5: diag["has_struct"] += 1
            if _c6: diag["on_ob"]      += 1
            if all([_c1, _c2, _c3, _c4, _c5, _c6]):
                diag["all_pass"] += 1

            # 每 diag_interval 根印一次報告 + DB 統計
            if diag["total"] % diag_interval == 0 and diag["total"] > 0:
                # DB 統計
                stats = db.get_stats()
                print(f"\n[DB] 記憶庫：{stats.get('total',0):,} 筆 | "
                      f"正:{stats.get('positive',0):,} 負:{stats.get('negative',0):,} | "
                      f"{stats.get('size_mb',0)} MB")

                # 每 24 小時（96 × 15 = 1440 根）執行一次 VACUUM
                if diag["total"] % (diag_interval * 15) == 0:
                    print("[DB] 執行定期 VACUUM...")
                    await asyncio.to_thread(db.vacuum)
                t = diag["total"]
                print(f"\n{'='*60}")
                print(f"[DIAG] 聖盾觸發率（{t} 根 K 棒）  BTC:{price_now:.0f}")
                print(f"  C1 H4方向明確 (|h4t|>0.3):  {diag['h4_clear']:4d}/{t} = {diag['h4_clear']/t*100:.1f}%  now={_h4t:.3f}")
                print(f"  C2 H4位置合理 (0.25~0.75): {diag['h4_pos_ok']:4d}/{t} = {diag['h4_pos_ok']/t*100:.1f}%  now={_h4p:.3f}")
                print(f"  C3 Killzone:               {diag['in_kz']:4d}/{t} = {diag['in_kz']/t*100:.1f}%")
                print(f"  C4 Sweep (|sw|>0.3):       {diag['has_sweep']:4d}/{t} = {diag['has_sweep']/t*100:.1f}%  now={_sw:.3f}")
                print(f"  C5 結構確認:               {diag['has_struct']:4d}/{t} = {diag['has_struct']/t*100:.1f}%  bos={_bos:.3f} choch={_choch:.3f} 多:{_c5_bull} 空:{_c5_bear}")
                print(f"  C6 接近OB(<0.5):           {diag['on_ob']:4d}/{t} = {diag['on_ob']/t*100:.1f}%  bull_dist={_ob_bull_dist:.3f} bear_dist={_ob_bear_dist:.3f}")
                print(f"  ── 全部通過:               {diag['all_pass']:4d}/{t} = {diag['all_pass']/t*100:.1f}%")
                print(f"  ── 實際開倉:               {diag['opened']:4d} 次（由聖盾放行）")
                # 找出最嚴的瓶頸條件
                rates = [
                    ("H4方向", diag["h4_clear"]),
                    ("H4位置", diag["h4_pos_ok"]),
                    ("Killzone", diag["in_kz"]),
                    ("Sweep", diag["has_sweep"]),
                    ("BOS/CHoCH", diag["has_struct"]),
                    ("回踩OB", diag["on_ob"]),
                ]
                bottleneck = min(rates, key=lambda x: x[1])
                print(f"  >> 瓶頸條件: {bottleneck[0]} ({bottleneck[1]/t*100:.1f}%)")
                print(f"  物理驗證: {physic.get_diagnosis()}")
                print(f"{'='*60}")

            # 先算一次共享特徵，16 個 agent 共用，避免重複計算
            shared_feat = await asyncio.to_thread(FeatureFactory.build_state, raw_candles)

            # 把計算好的 SMC 特徵注入 shared_feat
            # 這樣 AI 的 state 和 reward 的 smc_info 用同一套數字，學習訊號一致
            if shared_feat is not None and not shared_feat.empty:
                # H4 大框架
                shared_feat["h4_trend"] = h4_context.get("h4_trend", 0.0)
                shared_feat["h4_pos"]   = h4_context.get("h4_pos", 0.5)
                # SMC 特徵從 processor 注入
                shared_feat["bos"]   = float(global_smc_feats.get("is_bos",   0.0))
                shared_feat["choch"] = float(global_smc_feats.get("is_choch", 0.0))

                # is_sweep：processor 的條件更嚴格（需要收回），
                # feature_factory 的 sweep 持續性更好
                # 優先用 processor 的值，若為 0 則保留 feature_factory 的值
                _proc_sweep = float(global_smc_feats.get("is_sweep", 0.0))
                if abs(_proc_sweep) > 0.01:
                    shared_feat["is_sweep"] = _proc_sweep
                # 否則保留 feature_factory 算的（shared_feat 裡已有）

                # ob_bull/ob_bear：processor 的 ob_retrace 幾乎永遠 0
                # 改用 feature_factory 的距離特徵（持續有非零值）
                # shared_feat 在第 212 行已由 feature_factory 計算，這裡不覆蓋
                # 只在 feature_factory 值為 0 時才用 processor 的備援值
                _ff_ob_bull = float(shared_feat.get("ob_bull", 0.0))
                _ff_ob_bear = float(shared_feat.get("ob_bear", 0.0))
                if abs(_ff_ob_bull) < 0.001:
                    shared_feat["ob_bull"] = float(global_smc_feats.get("ob_bull_retrace", 0.0))
                if abs(_ff_ob_bear) < 0.001:
                    shared_feat["ob_bear"] = float(global_smc_feats.get("ob_bear_retrace", 0.0))

                # ── Volume Delta + 微觀結構特徵注入 ─────────────
                phys_feat = physic.get_features()
                shared_feat["delta_sum"]       = phys_feat.get("delta_sum",       0.0)
                shared_feat["delta_ratio"]     = phys_feat.get("delta_ratio",     0.0)
                shared_feat["trade_intensity"] = phys_feat.get("trade_intensity", 0.0)
                shared_feat["absorption"]      = phys_feat.get("absorption",      0.0)
                # hurst 由 feature_factory 計算（在 shared_feat 裡已有正確值）
                # 不要用 physic.get_features() 覆蓋它，physic 不計算 hurst
                # 只在 shared_feat 裡沒有 hurst 時才用預設值
                if "hurst" not in shared_feat or shared_feat.get("hurst") == 0.5:
                    # 嘗試從 feature_factory 的最新計算結果取
                    _ff_hurst = global_smc_feats.get("hurst", None)
                    if _ff_hurst is not None and _ff_hurst != 0.5:
                        shared_feat["hurst"] = float(_ff_hurst)
                    # 否則保留 shared_feat 裡的值（可能已由 feature_factory 設定）
                shared_feat["vpin"]            = phys_feat.get("vpin",            0.0)
                shared_feat["obi"]             = phys_feat.get("obi",             0.0)
                # 感官缺失旗標
                global_smc_feats["obi_available"] = phys_feat.get("obi_available", False)
                global_smc_feats.update(phys_feat)

                # 物理特徵診斷（每 48 根印一次）
                if diag["total"] % 48 == 1:
                    dr  = phys_feat.get("delta_ratio",     0.0)
                    ds  = phys_feat.get("delta_sum",       0.0)
                    ti  = phys_feat.get("trade_intensity", 0.0)
                    vp  = phys_feat.get("vpin",            0.0)
                    ab  = phys_feat.get("absorption",      0.0)
                    ob  = phys_feat.get("obi",             0.0)
                    ok  = phys_feat.get("obi_available",   False)
                    all_zero = all(abs(v) < 0.001 for v in [dr, ds, ti, vp, ab])
                    status = "❌ 全零！WebSocket可能斷線" if all_zero else "✅ 正常"
                    print(f"\n[PhysicEngine診斷] {status}")
                    print(f"  delta_ratio={dr:.3f} delta_sum={ds:.3f}"
                          f" intensity={ti:.3f}")
                    print(f"  vpin={vp:.3f} absorption={ab:.3f}"
                          f" obi={ob:.3f} obi_ok={ok}")

            pnl_report = []
            for i in range(num_agents):
                try:
                    a_id = str(i)

                    # 1. 準備狀態（傳入 shared_features，特徵來源與 smc_info 一致）
                    current_state, _ = dp.prepare_state(
                        i, raw_candles, use_lstm=(i >= 8), shared_features=shared_feat
                    )
                    if current_state is None or len(current_state.flatten()) != total_dim:
                        continue

                    # 2. 決策
                    real_current_pos = rm.stats[a_id]['pos']
                    # 探索率 0.15
                    try:
                        q_vals = agents[i].get_q_values(current_state)
                    except AttributeError:
                        q_vals = np.array([0.0, 0.0, 0.0])
                    ai_intent = agents[i].select_action(
                        current_state,
                        epsilon=0.15,
                        current_pos=real_current_pos
                    )

                    # 自我意識：開倉前評估信心
                    if real_current_pos == 0 and ai_intent != 0:
                        c = consciousness.get(a_id)
                        if c is not None:
                            feat_dict = dict(shared_feat) if shared_feat is not None else {}
                            ai_intent, conf, reason = c.pre_action(
                                q_vals, ai_intent, feat_dict
                            )
                            if ai_intent == 0:
                                pass   # 信心不足，靜默放棄

                    # 3. 損益計算
                    # 把追蹤止盈的峰值資訊注入 smc_info，讓 reward 能引導 AI 止盈時機
                    peak_info = dict(global_smc_feats)
                    peak_info["peak_pnl"] = rm.stats[a_id].get("peak_pnl", 0.0)

                    score, cur_pnl, tot_pnl, actual_action = rm.update_pnl(
                        a_id, ai_intent, price_now, arena, smc_info=peak_info
                    )

                    # 4. 寫入戰報
                    if last_states[i] is not None:
                        db.add_trade(
                            agent_id=a_id,
                            state=last_states[i],
                            action=actual_action,
                            entry=rm.stats[a_id].get('entry', 0),
                            reward=score,
                            pnl=tot_pnl,
                            cur_pnl=cur_pnl,
                            next_state=current_state,
                            smc_info=global_smc_feats
                        )

                    last_states[i] = current_state

                    # 意識：開倉記錄
                    if last_actions[i] == 0 and actual_action != 0:
                        c = consciousness.get(a_id)
                        if c is not None:
                            feat_dict = dict(shared_feat) if shared_feat is not None else {}
                            c.open_trade(actual_action, price_now, feat_dict)

                    # 意識：平倉後反省（用 cur_pnl 這筆的損益）
                    if last_actions[i] != 0 and actual_action == 0:
                        c = consciousness.get(a_id)
                        if c is not None:
                            c.post_trade(float(cur_pnl), price_now)

                    last_actions[i] = actual_action

                    # 診斷：記錄實際開倉
                    if i == 0 and actual_action != 0 and last_actions[0] == 0:
                        diag["opened"] += 1

                    # 5. 【修正】每個 agent 各自抽 batch 學習，不共用
                    # 原本只有 i%4==0 的 agent 學習，其他完全不學
                    # 每 2 根學一次常規 replay
                    if candle_count % 2 == 0:
                        agent_samples = db.sample_high_quality(batch_size=64, target_dim=total_dim)
                        if agent_samples:
                            agents[i].learn_from_wisdom(agent_samples)

                    # 轉世邏輯 (對齊 3 參數介面)
                    reborn = rm.check_evolution(a_id, agents[i], tot_pnl)
                    if reborn:
                        agents[i] = reborn
                        GLOBAL_WARRIORS[i] = agents[i]
                        dp.stackers[i].reset()
                        last_states[i] = None

                    if i < 4:
                        act_desc = "多" if actual_action == 1 else ("空" if actual_action == 2 else "💤")
                        pnl_report.append(f"A{a_id}{act_desc}:{tot_pnl:.1f}%")

                except Exception as e:
                    print(f"\n[⚠️] 戰士 A{i} 發生運算異常: {e}")
                    continue

            level = global_smc_feats.get('level_desc', 'Stable')
            print(f"\r[{now_ts.strftime('%H:%M:%S')}] {level} | BTC: {price_now:.1f} | {' | '.join(pnl_report)}", end='')
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"\n[❌] 核心崩潰: {e}")
        traceback.print_exc()
    finally:
        await physic.stop()
        await engine.close()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_loop())
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[💾] 正在執行最終記憶固化程序...")
        if GLOBAL_WARRIORS:
            for a in GLOBAL_WARRIORS:
                try: a.save_brain()
                except: pass
        loop.close()
        print("[🌙] 系統已釋放。")