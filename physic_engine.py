"""
physic_engine.py — Volume Delta 物理驗證引擎 v2

架構：Active-Active 雙交易所（幣安 + OKX）
  - 幣安正常 → 用幣安
  - 幣安斷線 → 自動切換 OKX
  - 用 ccxt.pro watch_trades，比原始 WebSocket 更穩定

特徵輸出（符號鎖定縮放，方向永遠正確）：
  delta_sum      : 多空力道累計（正=多方主導）
  delta_ratio    : 買賣比例（正=多方，負=空方）
  trade_intensity: 成交活躍度（高=機構在行動）
"""

import asyncio
import time
import json
import os
import ssl
import certifi
import numpy as np
from collections import deque

try:
    import ccxt.pro as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("[PhysicEngine] ccxt.pro 未安裝，使用降級模式（特徵固定為 0）")

def _make_ssl_context():
    """
    建立強制 SNI 的 SSL Context
    幣安 2026 年新增 TLS SNI 驗證，沒有正確傳送 SNI 會被防火牆擋掉
    certifi 提供最新的 CA 憑證，避免憑證驗證失敗
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = True          # 強制驗證 hostname（觸發 SNI）
    ctx.verify_mode    = ssl.CERT_REQUIRED
    # 明確指定支援的 TLS 版本，排除舊版
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


class PhysicEngine:
    def __init__(self, symbol: str = "BTC/USDT:USDT"):  # CCXT 合約格式
        self.symbol       = symbol
        self._running     = False
        self._output_file = "physic_data.json"

        # ── 雙交易所 Active-Active ────────────────────────────────
        if CCXT_AVAILABLE:
            _ssl = _make_ssl_context()
            _ws_opts = {
                "ws": {
                    "options": {
                        "max_msg_size": 100 * 1024 * 1024,   # 100MB，防止 OB 大訊息截斷
                    }
                }
            }
            self.exchanges = {
                "binance": ccxt.binance({
                    "enableRateLimit": True,
                    "options":  {"defaultType": "future"},
                    "verify":   True,
                    "aiohttp_connector_args": {"ssl": _ssl},
                    **_ws_opts,
                }),
                "bingx": ccxt.bingx({
                    "enableRateLimit": True,
                    "options":  {"defaultType": "swap"},
                    "verify":   True,
                    "aiohttp_connector_args": {"ssl": _ssl},
                    "ws": {
                        "options": {
                            "max_msg_size": 100 * 1024 * 1024,
                            # BingX 要求 JSON 格式心跳，不是標準 WebSocket ping frame
                            # ccxt.pro 的 ping 設定讓它送正確格式
                            "ping": True,
                        }
                    },
                }),
            }
            # ob_exchange 已移除，OB 改用原生 WebSocket 直連
            self.ob_exchange = None
        else:
            self.exchanges = {}

        # ── 每分鐘 bucket 累計 ────────────────────────────────────
        self.stats = {
            ex: {
                "delta_sum":   0.0,
                "total_vol":   0.0,
                "count":       0,
                "last_update": 0.0,
            }
            for ex in ["binance", "bingx"]
        }

        # ── 歷史基準（符號鎖定縮放用）────────────────────────────
        self._abs_ratio_hist   = deque(maxlen=288)
        self._abs_delta_hist   = deque(maxlen=288)
        self._intensity_hist   = deque(maxlen=288)

        # ── 輸出特徵 ──────────────────────────────────────────────
        self.features = {
            "delta_sum":       0.0,
            "delta_ratio":     0.0,
            "trade_intensity": 0.0,
            "absorption":      0.0,
            "vpin":            0.0,
            "obi":             0.0,
        }
        # forward-fill 快取：感測器失效時用最後一次有效值
        # 避免 TCN 在空資料上學習（零值佔 60%+ 的根本原因）
        self._last_valid_features = dict(self.features)

        self._vpin_history: deque = deque(maxlen=50)
        self._obi_latest    = 0.0    # 最新 OBI 快照
        self._obi_available = False  # 感官是否正常（斷線時為 False）
        self._obi_lock      = asyncio.Lock()

        self._lock         = asyncio.Lock()
        self._last_reset   = time.time()
        self._tasks        = []

    # ──────────────────────────────────────────────────────────────
    # 公開介面
    # ──────────────────────────────────────────────────────────────
    async def start(self):
        """啟動背景任務"""
        self._running = True
        if not CCXT_AVAILABLE:
            print("[PhysicEngine] 降級模式，特徵固定為 0")
            return

        # load_markets 只在啟動時做一次，帶 timeout 防止卡死
        # 斷線重連時不再重複呼叫，避免 REST 輪詢炸 API 權重
        # 錯開 DataEngine 的請求：等 3 秒後再載入市場
        # 避免跟主程式的 fetch_ohlcv 同時打 API 觸發 429
        await asyncio.sleep(3)

        for name, ex in self.exchanges.items():
            for attempt in range(3):
                try:
                    print(f"[PhysicEngine] {name} 載入市場（一次性）...")
                    await asyncio.wait_for(ex.load_markets(), timeout=20.0)
                    print(f"[PhysicEngine] {name} 市場載入完成")
                    break
                except asyncio.TimeoutError:
                    print(f"[PhysicEngine] {name} 載入市場超時，跳過")
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "Too Many Requests" in err:
                        wait = 60 * (attempt + 1)
                        print(f"[PhysicEngine] {name} 429 限流，等 {wait}s 後重試...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"[PhysicEngine] {name} 載入市場失敗：{e}，跳過")
                        break

        # OB 用原生 WebSocket，不需要 load_markets

        self._tasks = [
            asyncio.create_task(self._watch_exchange("binance")),
            asyncio.create_task(self._watch_exchange("bingx")),
            asyncio.create_task(self._bingx_heartbeat()),   # BingX 專用心跳
            asyncio.create_task(self._watch_orderbook()),
            asyncio.create_task(self._aggregator()),
        ]
        print(f"[PhysicEngine] 啟動（幣安 + OKX 雙備援）")

    async def stop(self):
        """安全關閉（確保所有 aiohttp session 都被釋放）"""
        self._running = False
        for t in self._tasks:
            t.cancel()
        # 等所有 task 完成（每個 task 的 CancelledError 會觸發 ex.close()）
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # 最後再確認一次全部關閉
        for name, ex in self.exchanges.items():
            try:
                await ex.close()
                print(f"[PhysicEngine] {name} 連線已釋放")
            except Exception:
                pass

        print("[PhysicEngine] 已關閉")

    def get_features(self) -> dict:
        """主程式每根 K 棒呼叫，取得最新物理特徵"""
        # 同時嘗試從 JSON 讀（支援獨立進程部署）
        try:
            if os.path.exists(self._output_file):
                mtime = os.path.getmtime(self._output_file)
                if time.time() - mtime < 120:   # 2 分鐘內的資料才算有效
                    with open(self._output_file) as f:
                        data = json.load(f)
                    if data.get("status") == "HEALTHY":
                        return {
                            "delta_sum":       float(data.get("delta_sum_norm",   0.0)),
                            "delta_ratio":     float(data.get("delta_ratio_norm", 0.0)),
                            "trade_intensity": float(data.get("trade_intensity_norm", 0.0)),
                            "absorption":      float(data.get("absorption", 0.0)),
                            "vpin":            float(data.get("vpin", 0.0)),
                            "obi":             float(data.get("obi",  0.0)),
                            "obi_available":   bool(data.get("obi_available", False)),
                        }
        except Exception:
            pass
        # 連線失效時回傳最後一次有效值（forward-fill）
        # 比全回傳 0 好，至少不會讓 TCN 在空資料上學習
        if all(v == 0.0 for v in self.features.values()):
            return dict(self._last_valid_features)
        return dict(self.features)

    def get_diagnosis(self) -> str:
        f = self.get_features()
        # 讀原始數值
        try:
            with open(self._output_file) as fp:
                d = json.load(fp)
            src    = d.get("source", "?")
            raw_r  = d.get("delta_ratio", 0)
            raw_d  = d.get("delta_sum", 0)
            raw_i  = d.get("trade_intensity", 0)
            status = d.get("status", "?")
        except Exception:
            src = raw_r = raw_d = raw_i = "N/A"
            status = "NO_FILE"

        return (
            f"[{src}|{status}] "
            f"Delta:{raw_d}→{f['delta_sum']:+.2f} | "
            f"Ratio:{raw_r}→{f['delta_ratio']:+.2f} | "
            f"活躍:{raw_i}→{f['trade_intensity']:+.2f}"
        )

    # ──────────────────────────────────────────────────────────────
    # 內部邏輯
    # ──────────────────────────────────────────────────────────────
    async def _watch_orderbook(self, levels: int = 25):
        """
        原生 WebSocket 直連 Kraken 現貨 Order Book
        Kraken 對台灣 IP 無封鎖

        Kraken v2 WebSocket：
          wss://ws.kraken.com/v2
          訂閱 book channel，BTC/USD，25 檔深度
          推送格式：snapshot（完整）/ update（增量）
          本地維護 order book，qty=0 的掛單刪除
        """
        import websockets as ws_lib

        url         = "wss://ws.kraken.com/v2"
        last_calc   = 0.0
        retry_delay = 1.0

        # 本地 order book（Kraken 增量更新，需本地維護）
        local_bids = {}   # price → qty
        local_asks = {}

        def _calc_obi():
            top_bids = sorted(local_bids.items(), reverse=True)[:levels]
            top_asks = sorted(local_asks.items())[:levels]
            bids_vol = sum(q for _, q in top_bids)
            asks_vol = sum(q for _, q in top_asks)
            raw = (bids_vol - asks_vol) / (bids_vol + asks_vol + 1e-9)
            return float(np.clip(raw * 3.0, -3, 3))

        while self._running:
            try:
                print("[PhysicEngine] OB Kraken 連線中...")
                local_bids.clear()
                local_asks.clear()

                async with ws_lib.connect(
                    url,
                    ping_interval = None,   # 關閉自動 ping，Kraken 自己管心跳
                    open_timeout  = 15,
                    max_size      = 100 * 1024 * 1024,
                ) as ws:
                    # 訂閱 BTC/USD 25 檔 order book
                    sub_msg = json.dumps({
                        "method": "subscribe",
                        "params": {
                            "channel": "book",
                            "symbol":  ["BTC/USD"],
                            "depth":   25
                        }
                    })
                    await ws.send(sub_msg)
                    print("[PhysicEngine] OB Kraken 連線成功，訂閱中...")
                    retry_delay = 1.0

                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg     = json.loads(raw)
                            channel = msg.get("channel", "")

                            # 跳過非 book 訊息（heartbeat / status / subscriptions）
                            if channel != "book":
                                continue

                            msg_type = msg.get("type", "")
                            data_list = msg.get("data", [])

                            for data in data_list:
                                bids = data.get("bids", [])
                                asks = data.get("asks", [])

                                if msg_type == "snapshot":
                                    # 完整快照：重建本地 order book
                                    local_bids.clear()
                                    local_asks.clear()
                                    local_bids.update({float(e["price"]): q for e in bids if (q := float(e["qty"])) > 0})
                                    local_asks.update({float(e["price"]): q for e in asks if (q := float(e["qty"])) > 0})

                                elif msg_type == "update":
                                    # 增量更新：qty=0 刪除，否則更新
                                    for entry in bids:
                                        p, q = float(entry["price"]), float(entry["qty"])
                                        if q == 0:
                                            local_bids.pop(p, None)
                                        else:
                                            local_bids[p] = q
                                    for entry in asks:
                                        p, q = float(entry["price"]), float(entry["qty"])
                                        if q == 0:
                                            local_asks.pop(p, None)
                                        else:
                                            local_asks[p] = q

                            # 每 3 秒計算一次 OBI
                            now = time.time()
                            if now - last_calc < 3:
                                continue
                            last_calc = now

                            if not local_bids or not local_asks:
                                continue

                            obi_norm = _calc_obi()
                            async with self._obi_lock:
                                self._obi_latest    = obi_norm
                                self._obi_available = True

                        except Exception:
                            continue   # 單筆解析失敗不斷線

            except asyncio.CancelledError:
                return

            except Exception as e:
                async with self._obi_lock:
                    self._obi_latest    = 0.0
                    self._obi_available = False

                print(f"[PhysicEngine] OB Kraken 斷線：{e}，{retry_delay:.0f}s 後重連...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def _bingx_heartbeat(self):
        """
        BingX 專用手動心跳
        BingX 要求每 20 秒送一次 JSON 格式的 ping，不是標準 WebSocket ping frame
        ccxt.pro 底層送的是標準 frame，BingX 不認識會斷線（1006）
        """
        ex = self.exchanges.get("bingx")
        while self._running:
            await asyncio.sleep(20)
            try:
                if ex and hasattr(ex, "client") and ex.client:
                    # 直接對 ccxt.pro 底層的 ws client 送 JSON ping
                    ping_msg = json.dumps({"ping": int(time.time() * 1000)})
                    for client in ex.clients.values():
                        try:
                            await client.socket.send(ping_msg)
                        except Exception:
                            pass
            except Exception:
                pass   # 心跳失敗不影響主流程

    async def _watch_exchange(self, name: str):
        """
        監控單一交易所，斷線自動重連

        核心原則：
        1. 不重建實例 → 避免資源洩漏（Unclosed client session）
        2. 不重新 load_markets → 避免 GET /instruments 頻率限制
        3. 斷線只 sleep，讓 ccxt.pro 內部機制自動重連 WebSocket
        """
        ex = self.exchanges[name]
        retry_delay = 1.0
        print(f"[PhysicEngine] {name} 開始 WebSocket 監聽...")

        while self._running:
            try:
                trades = await ex.watch_trades(self.symbol)
                retry_delay = 1.0

                async with self._lock:
                    for t in trades:
                        side   = 1.0 if t["side"] == "buy" else -1.0
                        amount = float(t.get("amount", 0))
                        self.stats[name]["delta_sum"] += side * amount
                        self.stats[name]["total_vol"] += amount
                        self.stats[name]["count"]     += 1
                        self.stats[name]["last_update"] = time.time()

                        # 每收到 1000 筆印一次確認
                        c = self.stats[name]["count"]
                        if c % 1000 == 0 and c > 0:
                            print(f"[PhysicEngine] {name} 累計 {c} 筆成交 "
                                  f"delta={self.stats[name]['delta_sum']:.2f}")

            except asyncio.CancelledError:
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "Too Many Requests" in err:
                    wait = min(retry_delay * 4, 120)
                    print(f"[PhysicEngine] {name} 429限流，等 {wait:.0f}s...")
                    await asyncio.sleep(wait)
                    retry_delay = wait
                else:
                    print(f"[PhysicEngine] {name} 暫時中斷：{e}，{retry_delay:.0f}s 後繼續...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)

    async def _aggregator(self):
        """每秒讀取數據，每分鐘封存一次 bucket，每 15 分鐘重置"""
        last_minute  = int(time.time() // 60)
        last_reset_m = int(time.time() // 900)

        while self._running:
            await asyncio.sleep(1)
            now        = time.time()
            now_minute = int(now // 60)
            now_reset  = int(now // 900)

            # ── lock 範圍縮到最小：只讀 stats，計算和 IO 在鎖外 ──
            active     = None
            delta_sum  = 0.0
            total_vol  = 0.0
            count      = 0

            async with self._lock:
                if now - self.stats["binance"]["last_update"] < 10:
                    active = "binance"
                elif now - self.stats["bingx"]["last_update"] < 10:
                    active = "bingx"

                if active is not None:
                    s = self.stats[active]
                    # 快照當前累計值
                    delta_sum = s["delta_sum"]
                    total_vol = s["total_vol"]
                    count     = s["count"]
                    # 每分鐘封存後重置（讓下一分鐘從 0 重新累計）
                    if now_minute != last_minute:
                        s["delta_sum"] = 0.0
                        s["total_vol"] = 0.0
                        s["count"]     = 0

            if active is None:
                self._write_json({"status": "DISCONNECTED", "timestamp": now})
                continue

            # ── 鎖外做計算和 IO，不阻塞 _watch_exchange ──────────
            if now_minute != last_minute:
                last_minute = now_minute
                # count 是上一分鐘的總成交筆數
                # 正常 BTC 合約每分鐘 300~800 筆
                # count=0 代表 WebSocket 沒收到資料，可能斷線
                avg_trades = float(count)
                raw_ratio  = delta_sum / (total_vol + 1e-9)

                print(f"[PhysicEngine] 封存 {active}: "
                      f"count={count} delta={delta_sum:.2f} "
                      f"ratio={raw_ratio:.4f}")   # 診斷輸出

                self._abs_ratio_hist.append(abs(raw_ratio))
                self._abs_delta_hist.append(abs(delta_sum))
                self._intensity_hist.append(avg_trades)

                delta_ratio_norm = self._sign_locked(raw_ratio, self._abs_ratio_hist)
                delta_sum_norm   = self._sign_locked(delta_sum,  self._abs_delta_hist)
                # cold_start_ref = 500 筆/分鐘（BTC 合約正常範圍）
                intensity_norm   = self._zscore(avg_trades, self._intensity_hist,
                                               cold_start_ref=500.0)

                # ── Absorption Score（OFI 吸收得分）──────────────────
                # 吸收陷阱：Delta 強（有人主動買賣）但價格不動
                # 計算方法：用 delta_ratio 的「變化量」判斷
                #   delta_ratio 高但本分鐘 ratio 比上分鐘小 → 力道在被吸收
                #   用歷史 ratio 的標準差來衡量「相對停滯」
                if not hasattr(self, "_prev_ratio_norm"):
                    self._prev_ratio_norm = 0.0

                # 本分鐘 delta 強度
                current_strength = abs(delta_ratio_norm)
                # 上分鐘 delta 強度
                prev_strength    = abs(self._prev_ratio_norm)

                # 吸收得分：delta 很強（>1）但強度在下降（被吸收）
                # 方向從 raw_ratio 繼承（正=多方被吸收，負=空方被吸收）
                if current_strength > 0.5 and prev_strength > current_strength:
                    # 力道在減弱：吸收發生中
                    absorb_strength = prev_strength - current_strength
                    absorption = float(np.sign(raw_ratio) * np.clip(absorb_strength * 2, 0, 3))
                elif current_strength > 1.5:
                    # Delta 極強但後續會知道有沒有被吸收
                    # 先輸出正值（強買盤或強賣盤）
                    absorption = float(np.sign(raw_ratio) * np.clip(current_strength * 0.5, 0, 2))
                else:
                    absorption = 0.0

                # 更新上一次的值
                self._prev_ratio_norm = delta_ratio_norm

                # ── Rolling VPIN（訂單流毒性）────────────────────────
                # 舊版問題：用自身歷史百分位正規化，rolling_vpin 永遠等於
                # 歷史均值，百分位永遠 50%，輸出永遠 0
                #
                # 修法：用符號鎖定縮放（跟 delta_ratio 同邏輯）
                # VPIN = |delta/total|，方向從 delta_ratio 繼承
                # 高 VPIN + 正 delta_ratio = 強多方毒性（知情買家）
                # 高 VPIN + 負 delta_ratio = 強空方毒性（知情賣家）
                bucket_vpin = abs(raw_ratio)
                self._vpin_history.append(bucket_vpin)
                if len(self._vpin_history) >= 10:
                    # 用絕對值歷史做符號鎖定縮放
                    hist_arr  = np.array(list(self._vpin_history))
                    magnitude = float(np.mean(hist_arr < bucket_vpin))  # 0~1
                    # 繼承 raw_ratio 的方向（正=多方毒性，負=空方毒性）
                    vpin_norm = float(np.sign(raw_ratio) * magnitude * 3.0)
                else:
                    # 資料不足：直接用放大的 raw_ratio
                    vpin_norm = float(np.clip(raw_ratio * 10.0, -3, 3))

                # 讀取最新 OBI（若感官缺失則輸出 0）
                async with self._obi_lock:
                    obi_now       = self._obi_latest if self._obi_available else 0.0
                    obi_ok        = self._obi_available

                new_feats = {
                    "delta_sum":       float(np.clip(delta_sum_norm,   -3, 3)),
                    "delta_ratio":     float(np.clip(delta_ratio_norm, -3, 3)),
                    "trade_intensity": float(np.clip(intensity_norm,   -3, 3)),
                    "absorption":      float(np.clip(absorption,       -3, 3)),
                    "vpin":            float(np.clip(vpin_norm,        -3, 3)),
                    "obi":             float(obi_now),
                }

                # ffill：如果物理特徵有有效值就更新快取
                # delta_sum != 0 代表 trades WebSocket 正常
                if abs(new_feats["delta_sum"]) > 0.01 or abs(new_feats["delta_ratio"]) > 0.01:
                    self._last_valid_features = dict(new_feats)
                    self._stale_count = 0
                else:
                    self._stale_count += 1

                # 超過容忍次數才真正歸零，否則用上次有效值（ffill）
                if self._stale_count <= self._MAX_STALE:
                    self.features = dict(self._last_valid_features)
                    # obi 仍用最新值（obi 有自己的連線）
                    self.features["obi"] = float(obi_now)
                else:
                    self.features = new_feats
                # forward-fill：每次有效更新就存起來
                self._last_valid_features = dict(self.features)

                self._write_json({
                    "source":               active,
                    "status":               "HEALTHY",
                    "timestamp":            now,
                    "delta_sum":            round(delta_sum,   4),
                    "delta_ratio":          round(raw_ratio,   6),
                    "trade_intensity":      round(avg_trades,  2),
                    "delta_sum_norm":       round(delta_sum_norm,   3),
                    "delta_ratio_norm":     round(delta_ratio_norm, 3),
                    "trade_intensity_norm": round(intensity_norm,   3),
                    "absorption":          round(absorption,        3),
                    "vpin":                round(vpin_norm,         3),
                    "obi":                 round(obi_now,           3),
                    "obi_available":       self._obi_available,
                })

            # ── 每 15 分鐘重置（在鎖內做，保證原子性）───────────
            if now_reset != last_reset_m:
                last_reset_m = now_reset
                async with self._lock:
                    for k in self.stats:
                        self.stats[k] = {
                            "delta_sum":   0.0,
                            "total_vol":   0.0,
                            "count":       0,
                            "last_update": self.stats[k]["last_update"],
                        }

    def _sign_locked(self, value: float, hist: deque) -> float:
        """符號鎖定縮放：絕對值百分位 × 原始符號"""
        if len(hist) < 10:
            return float(np.clip(value * 10, -3, 3))
        arr       = np.array(list(hist))
        magnitude = float(np.mean(arr < abs(value)))
        return float(np.sign(value) * magnitude * 3.0)

    def _zscore(self, value: float, hist: deque,
               cold_start_ref: float = 500.0) -> float:
        """
        z-score 正規化（適合無方向的活躍度）

        冷啟動問題修正：
        歷史不足 5 筆時不輸出 0（會讓 AI 誤以為死盤）
        改用 cold_start_ref 作為基準做粗估
          - BTC 合約正常成交約 300~800 筆/分鐘
          - cold_start_ref = 500 作為中性基準
          - 超過 500 輸出正值，低於 500 輸出負值
        """
        if len(hist) < 5:
            # 冷啟動：用絕對值相對基準做粗估
            ratio = value / (cold_start_ref + 1e-9)
            # log scale 壓縮：正常市場約 ±1，極端市場 ±2~3
            import math
            if ratio <= 0:
                return 0.0
            norm = math.log(ratio + 1e-9) / math.log(2)  # log2 縮放
            return float(np.clip(norm, -3, 3))

        arr = np.array(list(hist))
        mu  = float(np.mean(arr))
        std = float(np.std(arr) + 1e-9)
        return float(np.clip((value - mu) / std, -3, 3))

    def _write_json(self, data: dict):
        try:
            with open(self._output_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass