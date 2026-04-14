import asyncio
from ccxt.pro import binance as ccxt_binance
from collections import deque


class DataEngine:
    def __init__(self, symbol="BTC/USDT", timeframe="15m"):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.exchange  = ccxt_binance({
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {"defaultType": "future"}
        })
        self.raw_data    = deque(maxlen=1000)
        self.h4_raw_data = deque(maxlen=200)   # H4 大框架數據

    async def connect(self):
        print(f"[*] 正在準備交易所 API 握手 (Symbol: {self.symbol})...")
        try:
            print(f"[*] 載入 15分K 歷史...")
            ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000),
                timeout=15.0
            )
            if not ohlcv:
                print("[ERROR] 交易所回傳空數據！")
                return False
            self.raw_data.extend(ohlcv)

            # 同時載入 H4 歷史（大框架方向判斷）
            print(f"[*] 載入 H4 歷史（大框架）...")
            h4_ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(self.symbol, "4h", limit=100),
                timeout=15.0
            )
            if h4_ohlcv:
                self.h4_raw_data.extend(h4_ohlcv)
                print(f"[OK] 15分K {len(self.raw_data)} 根 + H4 {len(self.h4_raw_data)} 根 載入完成")
            return True

        except asyncio.TimeoutError:
            print("[ERROR] 連線超時！")
            return False
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            return False

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            print("[*] 交易所連線已安全關閉。")

    async def get_latest_stream(self):
        if len(self.raw_data) > 0:
            yield list(self.raw_data), list(self.h4_raw_data)

        while True:
            try:
                candle = await asyncio.wait_for(
                    self.exchange.watch_ohlcv(self.symbol, self.timeframe),
                    timeout=10.0
                )
                if candle:
                    new_data = candle[0]
                    if len(self.raw_data) > 0 and self.raw_data[-1][0] == new_data[0]:
                        self.raw_data[-1] = new_data
                    else:
                        self.raw_data.append(new_data)

                # H4 每 4 小時才會有新 K 棒，用 REST 輕量更新
                # 每次 15分K 更新時都傳最新的 H4 快照出去
                yield list(self.raw_data), list(self.h4_raw_data)

            except asyncio.TimeoutError:
                if len(self.raw_data) > 0:
                    yield list(self.raw_data), list(self.h4_raw_data)
            except Exception as e:
                print(f"[!] WebSocket 異常: {e}")
                await asyncio.sleep(5)

    async def refresh_h4(self):
        """定時更新 H4 數據（在 main.py 的背景 task 中呼叫）"""
        try:
            h4_ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(self.symbol, "4h", limit=10),
                timeout=10.0
            )
            if h4_ohlcv:
                for candle in h4_ohlcv:
                    if len(self.h4_raw_data) > 0 and self.h4_raw_data[-1][0] == candle[0]:
                        self.h4_raw_data[-1] = candle
                    else:
                        self.h4_raw_data.append(candle)
        except Exception:
            pass