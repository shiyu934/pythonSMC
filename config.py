# config.py — 全域設定
# 【全面重構版】針對診斷報告：勝率31.8%、Reward -1.037、68.7%觀望
# -----------------------------------------------------------------------

# ── 神經網路維度 ──────────────────────────────────────────────────────
STATE_DIM   = 29
STACK_SIZE  = 12
TOTAL_DIM   = STATE_DIM * STACK_SIZE   # 348

# ── 趨勢相關設定 ──────────────────────────────────────────────────────
# h4_trend 的有效信號門檻（低於此值視為盤整）
H4_TREND_THRESHOLD = 0.2
# 趨勢持倉時的信心門檻（順勢時可以降低信心要求）
TREND_CONFIDENCE_BOOST = 0.1

# ── 戰士設定 ──────────────────────────────────────────────────────────
NUM_AGENTS  = 32

# ── 交易設定 ──────────────────────────────────────────────────────────
LEVERAGE             = 30
FEE_RATE             = 0.001                         # 0.1% 單次
FEE_LEVERAGED        = FEE_RATE * LEVERAGE * 100.0   # 3.0% 開+平
FEE_OPEN_LEVERAGED   = FEE_LEVERAGED / 2             # 1.5%
FEE_CLOSE_LEVERAGED  = FEE_LEVERAGED / 2             # 1.5%

# 【調整】陣亡線從 -25% 提高到 -15%
# 診斷：大部分 agent 在 -6~-8% 就已經腦死，拖到 -25% 只是浪費時間
DEAD_THRESHOLD   = -15.0

# 【調整】持倉鎖從 900s 降到 300s（一根K棒 = 900s，但允許更快反應）
LOCK_SECONDS     = 300

# 轉世冷卻維持
REBORN_COOLDOWN  = 600

# 【調整】止損從 -12% 放寬到 -8%
# 診斷：持倉週期 30-50 根，持倉太久虧太多才止損
STOP_LOSS_PCT    = -8.0

# 【調整】追蹤止盈啟動門檻降低
# 診斷：勝率低，能賺到 4% 的機會太少，降到 2% 讓更多盈利被鎖住
TRAIL_ACTIVATE   = 2.0
TRAIL_DRAWDOWN   = 1.5

# ── H4 大框架設定 ─────────────────────────────────────────────────────
H4_LOOKBACK = 100

# ── 特徵索引 ──────────────────────────────────────────────────────────
F_RSI        = 0
F_ATR        = 1
F_P_POS      = 2
F_FVG        = 3
F_SWEEP      = 4
F_HOUR_SIN   = 5
F_HOUR_COS   = 6
F_AMD_PHASE  = 7
F_ASIA_POS   = 8
F_PDH_DIST   = 9
F_PDL_DIST   = 10
F_BOS        = 11
F_CHOCH      = 12
F_OB_BULL    = 13
F_OB_BEAR    = 14
F_P_CHG      = 15
F_POC_DIST   = 16
F_EQ_HL      = 17
F_IS_NY      = 18
F_IS_TUE_WED = 19
F_H4_TREND   = 20
F_H4_POS     = 21
F_DELTA_SUM   = 22
F_DELTA_RATIO = 23
F_INTENSITY   = 24
F_ABSORPTION  = 25
F_HURST       = 26
F_VPIN        = 27
F_OBI         = 28