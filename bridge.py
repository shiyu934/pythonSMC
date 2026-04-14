import socket
import json
import time
import ccxt
import sqlite3
import subprocess
import threading
import pandas as pd
from config import NUM_AGENTS  # 從 config 讀，不再寫死

JAVA_PROJECT_DIR = r"E:\pythonSMC"
MAVEN_BIN_PATH = r"E:\pythonSMC\apache-maven-3.9.14-bin\apache-maven-3.9.14\bin\mvn.cmd"
JAVA_START_CMD = f'"{MAVEN_BIN_PATH}" spring-boot:run'
JAVA_PORT = 9998
JAVA_SERVER_ADDR = ('127.0.0.1', JAVA_PORT)

market_data = {"price": 0.0, "agents": [], "smc": {}}
stop_event = threading.Event()


def wait_for_java():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', JAVA_PORT)) == 0:
            print("[OK] Java 戰情中心已在背景待命。")
            return
    full_cmd = f'start "Java_SMC_Tactical" cmd /k "{JAVA_START_CMD}"'
    print("[*] 正在自動喚醒 Java 伺服器...")
    subprocess.Popen(full_cmd, cwd=JAVA_PROJECT_DIR, shell=True)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', JAVA_PORT)) == 0:
                print("[OK] Java 握手成功！")
                break
        time.sleep(2)


def data_fetcher_thread():
    """
    從 DB 讀取所有 agent 最新狀態

    優化：用 rowid 範圍查詢取代 GROUP BY MAX(id)
    GROUP BY 在大 DB 要掃描全表，32 個 agent 更慢
    改成每個 agent 只查最後 1 筆，用 index 加速
    """
    exchange = ccxt.binance({'enableRateLimit': True})

    while not stop_event.is_set():
        try:
            # A. 抓即時價格
            ticker = exchange.fetch_ticker('BTC/USDT')
            market_data["price"] = float(ticker['last'])

            # B. 讀取所有 agent 最新狀態
            # 優化：用 agent_id + id DESC index，每個 agent 只取最後 1 筆
            # 比 GROUP BY MAX(id) 快很多，不會隨 DB 大小變慢
            conn = sqlite3.connect("wisdom_book.db", timeout=10)
            conn.execute("PRAGMA cache_size=-32000;")  # 32MB 快取加速讀取

            agents_list = []
            smc_set = False

            for i in range(NUM_AGENTS):
                aid = str(i)
                try:
                    row = conn.execute("""
                        SELECT action, reward, pnl, cur_pnl, vah, val
                        FROM experiences
                        WHERE agent_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                    """, (aid,)).fetchone()

                    if row:
                        action, reward, pnl, cur_pnl, vah, val = row
                        agents_list.append({
                            "id": i,
                            "action": int(action or 0),
                            "score": float(reward or 0.0),
                            "cur_pnl": float(cur_pnl or 0.0),
                            "total_pnl": float(pnl or 0.0),
                        })
                        if not smc_set:
                            market_data["smc"] = {
                                "vah": float(vah or 0),
                                "val": float(val or 0)
                            }
                            smc_set = True
                    else:
                        agents_list.append({
                            "id": i, "action": 0,
                            "score": 0.0, "cur_pnl": 0.0, "total_pnl": 0.0
                        })

                except Exception:
                    agents_list.append({
                        "id": i, "action": 0,
                        "score": 0.0, "cur_pnl": 0.0, "total_pnl": 0.0
                    })

            conn.close()
            market_data["agents"] = agents_list
            time.sleep(0.5)

        except Exception as e:
            print(f"\n[!] Fetcher 錯誤: {e}")
            time.sleep(2)


def run_bridge():
    wait_for_java()
    exchange = ccxt.binance({'enableRateLimit': True})

    print("\n[*] 正在搬運初始歷史 K 棒...")
    history_data = {}
    for tf in ['1m', '3m', '5m', '15m']:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=400)
        history_data[f"history_{tf}"] = [
            {"time": int(x[0]), "open": float(x[1]), "high": float(x[2]),
             "low": float(x[3]), "close": float(x[4])} for x in ohlcv
        ]
    print(f"[OK] K 棒載入完成，監控 {NUM_AGENTS} 個 Agent")

    threading.Thread(target=data_fetcher_thread, daemon=True).start()

    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(JAVA_SERVER_ADDR)
            print("[OK] 數據通道已開啟")

            # INIT：只送一次歷史 K 棒
            init_payload = {
                "type": "INIT",
                "num_agents": NUM_AGENTS,  # 告訴前端現在有幾個 agent
                "data": history_data
            }
            s.sendall((json.dumps(init_payload) + "\n").encode('utf-8'))
            time.sleep(1)

            while True:
                payload = {
                    "type": "TACTICAL_UPDATE",
                    "is_full_load": False,
                    "btc_price": market_data["price"],
                    "smc": market_data["smc"],
                    "agents": market_data["agents"],
                    "num_agents": NUM_AGENTS,
                }
                s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
                print(f"\r[*] BTC:{market_data['price']:.1f} | "
                      f"Agents:{len(market_data['agents'])}/{NUM_AGENTS}", end='')
                time.sleep(1.0)

        except Exception as e:
            print(f"\nSocket Error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    try:
        run_bridge()
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[*] 關閉橋接器")