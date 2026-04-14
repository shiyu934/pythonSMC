import torch
import torch.nn as nn
import numpy as np
import os
from config import (
    STATE_DIM, STACK_SIZE, TOTAL_DIM,
    F_P_POS, F_SWEEP, F_AMD_PHASE, F_CHOCH,
    F_IS_NY, F_EQ_HL, F_PDH_DIST, F_PDL_DIST,
    F_H4_TREND, F_H4_POS,
    F_BOS, F_OB_BULL, F_OB_BEAR,
    F_DELTA_SUM, F_DELTA_RATIO, F_INTENSITY,
    F_ABSORPTION, F_HURST, F_VPIN, F_OBI
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── TCN 殘差塊 ─────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    """
    時序卷積殘差塊
    比 LSTM 優勢：
    1. 平行計算（LSTM 必須序列處理）
    2. 梯度路徑短，不會消失
    3. 可任意控制感受野（dilation）
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1   = nn.LayerNorm(out_ch)
        self.norm2   = nn.LayerNorm(out_ch)
        self.drop    = nn.Dropout(dropout)
        self.relu    = nn.GELU()
        # 殘差連接：如果輸入輸出維度不同需要投影
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.residual(x)
        out = self.conv1(x)[:, :, :x.size(2)]   # 因果裁剪
        out = self.relu(self.norm1(out.transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.conv2(out)[:, :, :x.size(2)]  # 因果裁剪
        out = self.relu(self.norm2(out.transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        return out + residual


# ── TCN Dueling DQN（全體 16 位戰士統一架構）─────────────────────────
class TCNQNetwork(nn.Module):
    """
    時序卷積網路 + Dueling DQN

    架構：
    輸入 (batch, TOTAL_DIM) -> reshape -> (batch, STACK_SIZE, STATE_DIM)
    -> 轉置 -> (batch, STATE_DIM, STACK_SIZE) 當作 (batch, features, seq)
    -> 四層 TCN（dilation 1,2,4,8）
    -> 取最後一幀 (last frame) 輸出
    -> Dueling 分流輸出

    對比 LSTM：
    - 平行計算：訓練速度 3~5x
    - 梯度不消失：穩定性更好
    - 多尺度感受野：同時看短期和中期模式
    """
    def __init__(self, state_dim=STATE_DIM, stack_size=STACK_SIZE,
                 action_dim=3, hidden=256, dropout=0.2)；
        super().__init__()
        self.state_dim  = state_dim
        self.stack_size = stack_size

        # 四層 TCN，dilation [1,2,4,8]
        # 感受野 = 1 + (3-1)×(1+2+4+8) = 29 幀 ≈ 7.25 小時歷史
        # dilation=8 讓模型能看到更久以前的吸收陷阱佈局
        self.tcn = nn.Sequential(
            TCNBlock(state_dim, hidden,      dilation=1, dropout=dropout),
            TCNBlock(hidden,    hidden,      dilation=2, dropout=dropout),
            TCNBlock(hidden,    hidden,      dilation=4, dropout=dropout),
            TCNBlock(hidden,    hidden * 2,  dilation=8, dropout=dropout),
        )

        # Dueling 分流
        self.value_stream     = nn.Linear(hidden * 2, 1)
        self.advantage_stream = nn.Linear(hidden * 2, action_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        # (batch, TOTAL_DIM) -> (batch, STACK_SIZE, STATE_DIM) -> (batch, STATE_DIM, STACK_SIZE)
        x = x.view(x.size(0), self.stack_size, self.state_dim)
        x = x.transpose(1, 2)   # (batch, features, seq)

        out = self.tcn(x)        # (batch, hidden*2, seq)
        # 取最後一幀 (最新時間點的特徵)，取代原本的 out.mean(dim=2) 全局平均池化
        feat = out[:, :, -1]     # (batch, hidden*2)

        v = self.value_stream(feat)
        a = self.advantage_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── 戰士控制 ───────────────────────────────────────────────────────────
class TradingAgent:
    def __init__(self, agent_id, state_dim=STATE_DIM, stack_size=STACK_SIZE,
                 action_dim=3, use_lstm=False):  # use_lstm 參數保留相容性，實際用 TCN
        self.agent_id   = str(agent_id)
        self.state_dim  = state_dim
        self.stack_size = stack_size
        self.total_dim  = state_dim * stack_size
        self.action_dim = action_dim
        self.use_lstm   = use_lstm  # 保留欄位，不再影響架構選擇

        self.gamma               = 0.95
        self.learn_step_counter  = 0
        self.replace_target_iter = 1000

        # 全部統一用 TCN，告別 DQN/LSTM 分裂問題
        self.model        = TCNQNetwork(state_dim, stack_size, action_dim).to(device)
        self.target_model = TCNQNetwork(state_dim, stack_size, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.SmoothL1Loss()   # Huber Loss，對大誤差更穩定

        self.model_path = f"./models/agent_{self.agent_id}_tcn.pth"
        self._load_brain()

    def _load_brain(self):
        if os.path.exists(self.model_path):
            try:
                sd = torch.load(self.model_path, map_location=device, weights_only=True)
                self.model.load_state_dict(sd)
                self.target_model.load_state_dict(sd)
                print(f"[*] 戰士 {self.agent_id} (TCN) 經驗同步完成。")
            except Exception as e:
                print(f"[!] 戰士 {self.agent_id} (TCN) 結構變更，重置學習。({e})")

    def select_action(self, state, epsilon=0.01, current_pos=0):
        self.model.eval()

        latest_feat = state.flatten()[-self.state_dim:]
        try:
            pd_ratio      = float(latest_feat[F_P_POS])
            delta_sum     = float(latest_feat[F_DELTA_SUM])
            delta_ratio   = float(latest_feat[F_DELTA_RATIO])
            trade_intens  = float(latest_feat[F_INTENSITY])
            is_sweep  = float(latest_feat[F_SWEEP])    # 連續值：正=多方掃蕩 負=空方掃蕩
            amd_phase = float(latest_feat[F_AMD_PHASE])
            choch     = float(latest_feat[F_CHOCH])    # 連續值：正=多方反轉 負=空方反轉
            is_ny     = float(latest_feat[F_IS_NY])    # 連續值：越接近 NY 越大
            eq_hl     = float(latest_feat[F_EQ_HL])
            ob_bull   = float(latest_feat[F_OB_BULL])  # OB 多方距離
            ob_bear   = float(latest_feat[F_OB_BEAR])  # OB 空方距離
            bos       = float(latest_feat[F_BOS])      # 連續值：正=多方突破 負=空方突破
            h4_trend  = float(latest_feat[F_H4_TREND]) # H4 EMA 斜率
            h4_pos    = float(latest_feat[F_H4_POS])
        except Exception:
            pd_ratio = 0.5; is_sweep = 0.0; amd_phase = 1.0
            choch = 0.0; is_ny = 0.0; eq_hl = 0.0
            ob_bull = 0.0; ob_bear = 0.0; bos = 0.0
            h4_trend = 0.0; h4_pos = 0.5
            delta_sum = 0.0; delta_ratio = 0.0; trade_intens = 0.0

        with torch.no_grad():
            state_t = torch.as_tensor(
                state.flatten(), dtype=torch.float32, device=device
            ).unsqueeze(0)

            if np.random.rand() < epsilon:
                return np.random.randint(self.action_dim)

            q_values = self.model(state_t)
            action   = torch.argmax(q_values).item()

            # ══════════════════════════════════════════════════════════
            # 聖盾 v4【放寬版】
            #
            # 診斷報告：
            #   - 勝率 31.8%，平均 Reward -1.037
            #   - 68.7% 觀望 → agent 學會躺平
            #   - 根本原因：舊聖盾 9 個 AND 條件太嚴，
            #     agent 幾乎永遠被擋在門外，無法累積學習樣本
            #
            # 重構策略：
            #   - 從「9 個 AND」改為「評分制」
            #   - 每個 SMC 條件成立 +1 分
            #   - 超過門檻分數才開倉
            #   - 這樣 agent 能更多開倉，同時仍有過濾
            #   - 門檻隨訓練進展可調整
            # ══════════════════════════════════════════════════════════
            if current_pos == 0 and action != 0:

                score = 0   # SMC 品質評分

                # ── H4 方向 ──────────────────────────────────────────
                h4_is_bull = h4_trend > 0.2
                h4_is_bear = h4_trend < -0.2
                # 只在強烈逆向時否決（門檻放寬）
                if action == 1 and h4_trend < -0.8:
                    return 0   # 只在非常強空時才擋多
                if action == 2 and h4_trend > 0.8:
                    return 0   # 只在非常強多時才擋空
                # H4 方向一致加分
                if action == 1 and h4_is_bull:
                    score += 2
                if action == 2 and h4_is_bear:
                    score += 2
                # 做空稀缺補償：做空時就算 H4 不明確也加 1 分
                # 避免 H4 偏多的環境裡做空永遠得不到分數
                if action == 2 and not h4_is_bull:
                    score += 1

                # ── KZ 時段（最重要，不在時段大幅扣分）─────────────────
                if is_ny > 0.4:
                    score += 2
                elif is_ny > 0.2:
                    score += 1
                elif is_ny < 0.1:
                    score -= 1   # 深度非 KZ 時段開倉扣分

                # ── BOS 結構確認 ──────────────────────────────────────
                if action == 1 and bos > 0.3:
                    score += 2
                elif action == 2 and bos < -0.3:
                    score += 2

                # ── Sweep 流動性 ──────────────────────────────────────
                if action == 1 and is_sweep > 0.3:
                    score += 1
                elif action == 2 and is_sweep < -0.3:
                    score += 1

                # ── CHoCH 反轉確認 ────────────────────────────────────
                if action == 1 and choch > 0.3:
                    score += 1
                elif action == 2 and choch < -0.3:
                    score += 1

                # ── Delta 物理驗證 ────────────────────────────────────
                if action == 1 and delta_ratio > 0.1:
                    score += 1
                elif action == 2 and delta_ratio < -0.1:
                    score += 1
                elif action == 1 and delta_ratio < -0.5:
                    score -= 1
                elif action == 2 and delta_ratio > 0.5:
                    score -= 1

                # ── OBI 感測器（連線不穩時不封鎖，只扣分）──────────────
                obi_val = float(latest_feat[F_OBI])
                if action == 1 and obi_val > 0.3:
                    score += 1
                elif action == 2 and obi_val < -0.3:
                    score += 1
                elif action == 1 and obi_val < -0.8:
                    score -= 1   # 強烈賣壓掛單，扣分但不封鎖

                # ── 開倉門檻 ─────────────────────────────────────────
                # 診斷：agent 幾乎不開單，降低門檻到 2 分
                # 現在只有 H4+BOS 或 KZ+BOS 就能達到 2 分
                # 讓 agent 能夠開倉，累積更多學習樣本
                if score < 2:
                    return 0

            return action

    def learn_from_wisdom(self, samples):
        if not samples or len(samples) < 32:
            return
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.learn_step_counter += 1
        self.model.train()
        try:
            # torch.as_tensor 比 FloatTensor 更高效
            # 直接指定 device，避免先建 CPU tensor 再搬到 GPU
            v_s  = torch.as_tensor(
                np.array([s["state"].flatten() for s in samples], dtype=np.float32),
                device=device
            )
            v_ns = torch.as_tensor(
                np.array([s["next_state"].flatten() for s in samples], dtype=np.float32),
                device=device
            )
            v_a  = torch.as_tensor(
                np.array([int(s["action"]) for s in samples], dtype=np.int64),
                device=device
            ).view(-1, 1)
            v_r  = torch.as_tensor(
                np.array([s["reward"] for s in samples], dtype=np.float32),
                device=device
            ).view(-1, 1)

            with torch.no_grad():
                # Double DQN (DDQN) 實作
                # 1. 用 online model 找出 next_state 中 Q 值最大的動作
                best_next_actions = self.model(v_ns).argmax(1).view(-1, 1)
                # 2. 用 target model 評估該動作的 Q 值，避免 Q 值高估
                next_q = self.target_model(v_ns).gather(1, best_next_actions)
                target_q = v_r + self.gamma * next_q

            current_q = self.model(v_s).gather(1, v_a)
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        except Exception as e:
            print(f"[!] 戰士 {self.agent_id} 訓練異常: {e}")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """回傳原始 Q 值，供意識系統評估確定性"""
        self.model.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(
                state.flatten(), dtype=torch.float32, device=device
            ).unsqueeze(0)
            q = self.model(state_t)
            return q.cpu().numpy().flatten()

    def save_brain(self):
        os.makedirs("./models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)