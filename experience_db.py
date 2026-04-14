import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from config import TOTAL_DIM, NUM_AGENTS


class ExperienceDB:
    def __init__(self, db_path="./wisdom_book.db"):
        self.db_path = db_path
        # [資工優化] WAL 模式是處理多執行緒寫入的唯一解，timeout 設 60 秒防止 Lock
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA cache_size=-256000;")  # 256MB 快取，大 DB 必要
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")   # 暫存放記憶體，加速查詢
        self.conn.execute("PRAGMA mmap_size=2147483648;") # 2GB mmap，大 DB 讀取更快
        self.conn.execute("PRAGMA page_size=4096;")       # 最佳頁面大小
        self._set_up_db()

    def _set_up_db(self):
        cursor = self.conn.cursor()
        # --- 【自動重置門衛】：檢查欄位是否對齊「三位一體 + Entry」架構 ---
        try:
            cursor.execute("SELECT entry FROM experiences LIMIT 1")
            # 【新增】偵測 is_sweep 欄位型別，INTEGER 會截斷連續值必須重建
            col_info = cursor.execute("PRAGMA table_info(experiences)").fetchall()
            col_names = {row[1] for row in col_info}
            col_types = {row[1]: row[2] for row in col_info}
            if col_types.get("is_sweep", "").upper() == "INTEGER":
                print("[RESET] is_sweep 型別為 INTEGER，重建 DB...")
                cursor.execute("DROP TABLE IF EXISTS experiences")
            else:
                # 自動補齊缺少的欄位（不重建，保留舊資料）
                new_cols = [
                    ("delta_sum",       "0.0"),
                    ("delta_ratio",     "0.0"),
                    ("trade_intensity", "0.0"),
                    ("absorption",      "0.0"),
                    ("hurst",           "0.5"),
                    ("vpin",            "0.0"),
                    ("obi",             "0.0"),
                ]
                added = []
                for col, default in new_cols:
                    if col not in col_names:
                        cursor.execute(
                            f"ALTER TABLE experiences ADD COLUMN {col} REAL DEFAULT {default}"
                        )
                        added.append(col)
                if added:
                    self.conn.commit()
                    print(f"[DB] 自動補齊欄位：{added}，舊資料保留")
        except sqlite3.OperationalError:
            print("[RESET] 偵測到舊版格式，執行物理重置...")
            cursor.execute("DROP TABLE IF EXISTS experiences")

        # 1. 建立核心戰報表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestamp DATETIME DEFAULT (datetime('now','localtime')),
                state TEXT,
                action INTEGER,
                reward REAL,       -- 大腦積分
                entry REAL,        -- 進場價格 (關鍵：讓 Bridge 計算物理盈虧)
                pnl REAL,          -- 生涯累計
                cur_pnl REAL,      -- 當下盈虧
                next_state TEXT,
                vah REAL DEFAULT 0,
                val REAL DEFAULT 0,
                is_sweep REAL DEFAULT 0,      -- 連續掃蕩深度
                is_choch REAL DEFAULT 0,      -- 連續反轉幅度
                is_bos   REAL DEFAULT 0,      -- 連續突破幅度
                delta_sum   REAL DEFAULT 0,   -- Volume Delta 累計（正規化）
                delta_ratio REAL DEFAULT 0,   -- Delta 比例（正規化）
                trade_intensity REAL DEFAULT 0, -- 成交活躍度（正規化）
                absorption  REAL DEFAULT 0,   -- OFI 吸收得分（陷阱偵測）
                hurst       REAL DEFAULT 0.5, -- Hurst 指數（趨勢持久性）
                vpin        REAL DEFAULT 0,   -- Rolling VPIN（訂單流毒性）
                obi         REAL DEFAULT 0,   -- Order Book Imbalance
                level_desc TEXT
            )
        ''')

        # 2. 建立 K 線緩存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT, timeframe TEXT, ts INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                datetime TEXT, PRIMARY KEY (symbol, timeframe, ts)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_ts ON experiences (agent_id, id DESC)')
        self.conn.commit()
        self._print_db_stats()

    def _print_db_stats(self):
        """啟動時印出 DB 統計，不刪任何資料"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiences")
            count = cursor.fetchone()[0]
            size_mb = os.path.getsize(self.db_path) / 1024 / 1024 if os.path.exists(self.db_path) else 0
            print(f"[DB] 記憶庫：{count:,} 筆 | 檔案大小：{size_mb:.1f} MB")
        except Exception:
            pass

    def add_trade(self, agent_id, state, action, entry=0.0, reward=0.0, pnl=0.0, cur_pnl=0.0, next_state=None,
                  smc_info=None):
        """
        三位一體同步寫入：
        reward  -> AI 獎勵 (Score)
        entry   -> 物理進場價
        pnl     -> 生涯累計 (Total PnL)
        cur_pnl -> 當前單筆 (Current PnL)
        """
        # 【優化】state 改用 float16 壓縮儲存，節省 50% 空間
        # float32 → float16：精度從 7 位降到 3 位，對 RL 訓練影響極小
        def _compress(arr):
            if isinstance(arr, np.ndarray):
                return json.dumps(arr.astype(np.float16).tolist())
            return json.dumps(arr)

        state_str   = _compress(state)
        n_state_str = _compress(next_state) if next_state is not None else state_str

        s = smc_info if isinstance(smc_info, dict) else {}

        # [資工校準]：如果觀望(0)，Entry 必須是 0；若持倉但沒傳 Entry，才用 VAH 頂替
        final_entry = float(entry)
        if int(action) == 0:
            final_entry = 0.0
        elif final_entry <= 0:
            final_entry = float(s.get('vah', 0))

        # 嚴格對齊 INSERT 順序：20 個變數
        data = (
            str(agent_id), state_str, int(action), float(reward),
            final_entry, float(pnl), float(cur_pnl), n_state_str,
            float(s.get('vah', 0)), float(s.get('val', 0)),
            float(s.get('is_sweep', 0.0)), float(s.get('is_choch', 0.0)),
            float(s.get('is_bos', 0.0)),
            float(s.get('delta_sum', 0.0)),
            float(s.get('delta_ratio', 0.0)),
            float(s.get('trade_intensity', 0.0)),
            float(s.get('absorption', 0.0)),
            float(s.get('hurst', 0.5)),
            float(s.get('vpin', 0.0)),
            float(s.get('obi', 0.0)),
            str(s.get('level_desc', 'Stable'))
        )

        # 批次寫入：先 execute 不立刻 commit，由外部批次 commit
        # 減少磁碟 fsync 次數，寫入速度提升 10~20 倍
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.conn.execute('''INSERT INTO experiences (
                    agent_id, state, action, reward,
                    entry, pnl, cur_pnl, next_state,
                    vah, val, is_sweep, is_choch, is_bos,
                    delta_sum, delta_ratio, trade_intensity,
                    absorption, hurst, vpin, obi,
                    level_desc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', data)
                # 不在這裡 commit，由 batch_commit() 統一處理
                self._pending_writes = getattr(self, "_pending_writes", 0) + 1
                if self._pending_writes >= NUM_AGENTS:
                    self.conn.commit()
                    self._pending_writes = 0
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                break
            except Exception as e:
                print(f"[!] Agent {agent_id} 寫入失敗: {e}")
                break

    def flush(self):
        """強制 commit 所有待寫入的資料（程式結束前呼叫）"""
        try:
            self.conn.commit()
            self._pending_writes = 0
        except Exception:
            pass

    def vacuum(self):
        """
        整理 DB 碎片（建議每跑 24 小時執行一次）
        DB 大量寫入後會有碎片，VACUUM 後檔案變小、查詢變快
        注意：執行時會短暫鎖定 DB，約 10~30 秒
        """
        try:
            self.conn.execute("VACUUM;")
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiences")
            count = cursor.fetchone()[0]
            size_mb = os.path.getsize(self.db_path) / 1024 / 1024
            print(f"[DB] VACUUM 完成：{count:,} 筆 | {size_mb:.1f} MB")
        except Exception as e:
            print(f"[DB] VACUUM 失敗: {e}")

    def get_stats(self):
        """回傳 DB 統計資訊"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiences")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM experiences WHERE reward > 0")
            pos = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM experiences WHERE reward < 0")
            neg = cursor.fetchone()[0]
            size_mb = os.path.getsize(self.db_path) / 1024 / 1024
            return {
                "total": total,
                "positive": pos,
                "negative": neg,
                "size_mb": round(size_mb, 1)
            }
        except Exception:
            return {}

    def add_ohlcv(self, symbol, timeframe, candle):
        dt_str = datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.conn.execute("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)",
                              (symbol, timeframe, candle[0], candle[1], candle[2], candle[3], candle[4], candle[5],
                               dt_str))
            self.conn.commit()
        except:
            pass

    def sample_high_quality(self, batch_size=32, target_dim=TOTAL_DIM):
        """
        分層抽樣：正樣本 50% + 負樣本 25% + 隨機 25%

        32 agent 大 DB 優化：
        - pool_size 從 500 降到 batch_size*4，減少每次掃描量
        - 三個查詢都限制在最近 50 萬筆的 rowid 範圍內
        - ORDER BY RANDOM() 在有 id >= ? 條件時會用 index scan，不全表掃
        """
        import random

        def _parse_rows(rows):
            result = []
            for r in rows:
                try:
                    s_np = np.array(json.loads(r[0]), dtype=np.float16).astype(np.float32).flatten()
                    n_np = np.array(json.loads(r[3]), dtype=np.float16).astype(np.float32).flatten()
                    if len(s_np) == target_dim and len(n_np) == target_dim:
                        result.append({
                            "state":      s_np,
                            "action":     int(r[1]),
                            "reward":     float(r[2]),
                            "next_state": n_np,
                        })
                except Exception:
                    continue
            return result

        try:
            cursor = self.conn.cursor()

            # rowid 範圍：最近 50 萬筆
            cursor.execute("SELECT MAX(id) FROM experiences")
            max_id = cursor.fetchone()[0] or 0
            min_id = max(1, max_id - 500000)

            # 【調整抽樣比例】
            # 診斷：平均 Reward -1.037，系統學歪了
            # 增加正樣本比例讓 agent 看到「成功是什麼樣子」
            # 正:負:隨機 = 50:30:20
            n_pos  = int(batch_size * 0.5)
            n_neg  = int(batch_size * 0.3)
            n_rand = batch_size - n_pos - n_neg
            pool   = batch_size * 5   # 稍微加大pool確保有足夠樣本

            # 多空平衡抽樣：確保做空樣本佔一定比例
            # 診斷：做多 40%+ 做空 7%，DB 裡做空樣本太少
            # 強制抽取做空樣本，讓 TCN 能學到做空策略
            n_short = batch_size // 6   # 約 17% 強制做空樣本
            n_long  = batch_size // 6   # 約 17% 強制做多樣本
            n_pos_remain = n_pos - n_long
            n_neg_remain = n_neg - n_short

            # 強制做空正樣本（做空且獲利）
            short_win = cursor.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE id >= ? AND next_state IS NOT NULL
                  AND action = 2 AND reward > 0
                ORDER BY RANDOM() LIMIT ?
            """, (min_id, n_short * 3)).fetchall()

            # 強制做多正樣本
            long_win = cursor.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE id >= ? AND next_state IS NOT NULL
                  AND action = 1 AND reward > 0
                ORDER BY RANDOM() LIMIT ?
            """, (min_id, n_long * 3)).fetchall()

            pos_rows = cursor.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE id >= ? AND next_state IS NOT NULL AND reward > 0
                ORDER BY RANDOM() LIMIT ?
            """, (min_id, pool)).fetchall()

            neg_rows = cursor.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE id >= ? AND next_state IS NOT NULL AND reward < 0
                ORDER BY RANDOM() LIMIT ?
            """, (min_id, pool)).fetchall()

            rand_rows = cursor.execute("""
                SELECT state, action, reward, next_state
                FROM experiences
                WHERE id >= ? AND next_state IS NOT NULL
                ORDER BY RANDOM() LIMIT ?
            """, (min_id, pool)).fetchall()

            if len(pos_rows) + len(neg_rows) + len(rand_rows) < batch_size:
                return None

            # 組合：強制多空樣本 + 一般正負樣本 + 隨機
            short_s = _parse_rows(short_win)[:n_short]
            long_s  = _parse_rows(long_win)[:n_long]
            pos_s   = _parse_rows(pos_rows)[:max(0, n_pos_remain)]
            neg_s   = _parse_rows(neg_rows)[:max(0, n_neg_remain)]
            rand_s  = _parse_rows(rand_rows)[:n_rand]
            samples = short_s + long_s + pos_s + neg_s + rand_s

            # 不足時用隨機補滿
            if len(samples) < batch_size:
                extra = _parse_rows(rand_rows)[n_rand : n_rand + (batch_size - len(samples))]
                samples += extra

            random.shuffle(samples)
            return samples[:batch_size] if len(samples) >= batch_size else None

        except Exception as e:
            print(f"[DB] 抽樣失敗: {e}")
            return None