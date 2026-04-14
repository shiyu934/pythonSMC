import sqlite3


def reset_and_tune_db():
    conn = sqlite3.connect("wisdom_book.db")
    # 開啟 WAL 模式，防止 Dashboard 和 Main 互相鎖死
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # 建立全新的 60 維 experiences 表
    conn.execute("DROP TABLE IF EXISTS experiences;")
    conn.execute("DROP TABLE IF EXISTS ohlcv;")

    # ... 讓 main.py 重新建立表格即可 ...
    conn.close()
    print("[✅] 資料庫已清空並優化。")


if __name__ == "__main__":
    reset_and_tune_db()