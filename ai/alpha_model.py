# ai/alpha_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=3):
        super().__init__()
        self.board_size = board_size
        # 卷积层定义
        """
            输入通道数 = 1 只有一个平面棋盘
            输出通道数 = 32 学习出 32 个特征图
            卷积核大小 = 3*3
            padding=1 保证输出维度仍然是 3*3
        """
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 第二个卷积层
        """
            输入通道数 = 32 来自第一个卷积层的输出
            输出通道数 = 64 学习出 64 个特征图
            卷积核大小 = 3*3
            padding=1 保证输出维度仍然是 3*3
        """
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # 策略头 policy head
        """
            将 64 个特征图变成 2 个（每个格子有 2 个特征）
            卷积核大小 1*1 不会融合空间信息，只提取每个格子上的特征
        """
        self.policy_conv = nn.Conv2d(64, 2, 1)
        """
            输入维度是 2*3*3 = 18
            输出维度是 9 每个格子一个动作偏好值
        """
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # 价值头 value head
        """
            将 64 个特征图降为 1 个通道，每个格子一个 scalar
        """
        self.value_conv = nn.Conv2d(64, 1, 1)
        """
            将 3*3 的扁平化值变成 64 维中间层
        """
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        """
            输出一个标量，代表当前局面对我的“胜率评估”
        """
        self.value_fc2 = nn.Linear(64, 1)

    # 前向传播逻辑
    def forward(self, x):
        x = x.view(-1, 1, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 策略头
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # 输出范围[-1,1]

        return p, v