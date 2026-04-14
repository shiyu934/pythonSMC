import torch


def perform_tcn_surgery(old_path, new_path):
    # 1. 載入舊大腦
    old_state = torch.load(old_path, map_location='cpu')
    new_state = {}

    print(f"[*] 偵測到模型結構，準備針對殘差網絡動刀...")

    # 定義需要擴張的層名稱
    # 根據你的輸出，這兩層是 input 入口
    target_layers = {
        'tcn.0.conv1.weight': 3,  # kernel_size = 3
        'tcn.0.residual.weight': 1  # kernel_size = 1
    }

    for name, param in old_state.items():
        if name in target_layers:
            # 取得原始維度 [out_channels, in_channels, kernel_size]
            # 例如: [128, 22, 3]
            out_c, in_c, k_size = param.shape
            print(f"[!] 正在擴張層: {name} | 原始形狀: {list(param.shape)}")

            # 創建新的權重張量 [128, 25, k_size]
            new_param = torch.zeros((out_c, 25, k_size))

            # 核心移植：將前 22 維的記憶平移過去
            new_param[:, :22, :] = param

            # 其餘 3 維 (Delta_sum, delta_ratio, intensity) 保持為 0
            new_state[name] = new_param
            print(f" -> 手術完成，新形狀: {list(new_param.shape)}")

        else:
            # 其他所有層 (tcn.1, tcn.2, value_stream 等) 維度都與 input 無關，直接複製
            new_state[name] = param

    # 3. 儲存手術成果
    torch.save(new_state, new_path)
    print(f"\n[*] 手術成功！")
    print(f"[*] 舊靈魂: {old_path} (22維)")
    print(f"[*] 新戰士: {new_path} (25維)")


if __name__ == "__main__":
    # 請修改為你實際的文件路徑
    perform_tcn_surgery('agent_0_tcn.pth', 'agent_0_tcn_upgraded.pth')