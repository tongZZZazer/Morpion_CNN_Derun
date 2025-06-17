# train_alpha_zero.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ai.alpha_model import AlphaZeroNet
from ai.self_play import play_self_game

# 🔧 超参数
NUM_SELF_PLAY_GAMES = 300        # 每轮自我博弈局数
EPOCHS_PER_TRAINING = 20         # 每轮训练轮数
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 20000       # 最大经验池大小
LR = 1e-4                         # 学习率
TOTAL_ITERATIONS = 100           # 训练迭代次数
SAVE_EVERY = 10                  # 每 N 轮保存一次模型

def train(model, data, optimizer):
    model.train()
    losses = []

    for epoch in range(EPOCHS_PER_TRAINING):
        np.random.shuffle(data)
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]

            states = torch.tensor([s for s, _, _ in batch], dtype=torch.float32).unsqueeze(1)
            pis = torch.tensor([p for _, p, _ in batch], dtype=torch.float32)
            zs = torch.tensor([z for _, _, z in batch], dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            policy_logits, values = model(states)

            value_loss = nn.MSELoss()(values, zs)
            policy_loss = nn.KLDivLoss(reduction="batchmean")(
                nn.LogSoftmax(dim=1)(policy_logits), pis)

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return np.mean(losses)

def main():
    model = AlphaZeroNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    replay_buffer = []

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n=== Iteration {iteration} ===")

        # 1. 自我博弈收集数据
        all_data = []
        for _ in range(NUM_SELF_PLAY_GAMES):
            game_data = play_self_game(model, simulations=50)  # 🔁 更强搜索
            all_data.extend(game_data)

        # 2. 更新经验池
        replay_buffer.extend(all_data)
        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer = replay_buffer[-REPLAY_BUFFER_SIZE:]  # 保留最新经验

        # 3. 用经验池训练模型
        avg_loss = train(model, replay_buffer, optimizer)
        print(f"平均 Loss : {avg_loss:.4f}，样本数 : {len(replay_buffer)}")

        # 4. 定期保存模型
        if iteration % SAVE_EVERY == 0 or iteration == TOTAL_ITERATIONS:
            filename = f"checkpoint_alpha_iter{iteration}.pt"
            torch.save(model.state_dict(), filename)
            print(f"模型已保存：{filename}")

if __name__ == "__main__":
    main()
