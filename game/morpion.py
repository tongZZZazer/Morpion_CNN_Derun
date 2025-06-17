# 引入 numpy 库，主要用于处理棋盘（3x3 矩阵）和计算（例如和、转置等）。
import numpy as np

# MorpionGame 是游戏的核心类，封装了游戏状态、规则与接口，供MCTS和AlphaZero调用
class MorpionGame:

    # 初始化3x3空棋盘，0表示空位，1表示玩家1，-1表示玩家2
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        # 自定义为玩家1为先手
        self.current_player = 1

    # 创建当前游戏状态的副本（用于 MCTS 模拟），避免不同模拟之间共享同一个棋盘。
    def clone(self):
        clone_game = MorpionGame()
        clone_game.board = self.board.copy()
        clone_game.current_player = self.current_player
        return clone_game

    # 返回所有可走的位置，作为 (i, j) 坐标对
    # 用于 MCTS 扩展节点和神经网络掩码
    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    # 执行一步棋 并自动切换玩家
    # 如果位置非法（已被占），抛出异常
    def make_move(self, move):
        # 拆解move坐标
        i, j = move
        # 判断是否该坐标已经有子
        if self.board[i, j] != 0:
            raise ValueError("Invalid move!")
        # 如果该坐标是空闲的，则当前玩家下子，相当于更新棋盘，但是返回的是数字而不是XO
        self.board[i, j] = self.current_player
        # 切换到另一个玩家 1 -> -1
        self.current_player *= -1

    # 判断是否有玩家胜利
    def check_winner(self):
        # 保存所有需要判断的行/列/对角线
        lines = []
        # 添加所有行
        lines.extend(self.board)
        # 添加所有列（转置后的行）
        lines.extend(self.board.T)
        # 主对角线（左上到右下）
        lines.append(np.diag(self.board))
        # 副对角线（右上到左下）
        lines.append(np.diag(np.fliplr(self.board)))

        for line in lines:
            # 如果某一行/列/对角线加起来是3或-3，说明一方赢了
            total = np.sum(line)
            # 玩家1胜
            if total == 3:
                return 1
            # 玩家2胜
            elif total == -3:
                return -1
        # 没有胜者
        return 0

    # 判断游戏是否结束
    def is_terminal(self):
        # 有一方赢了，结束
        if self.check_winner() != 0:
            return True
        # 棋盘满了，平局，结束
        if np.all(self.board != 0):
            return True
        return False

    # 打印棋盘 让玩家可视化
    def print_board(self):
        # 显示符号映射
        symbols = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        # 打印一个空行，美化输出，在棋盘下方留空格，提升可读性。
        print()

    # 返回当前局面（规范化视角）
    # 这个步骤对 AlphaZero 至关重要，因为我们要让神经网络训练的数据都从“当前玩家”的视角统一表示。
    def get_canonical_form(self):
        return self.current_player * self.board

    # 为神经网络的输出层设计提供标准：输出大小必须等于动作空间维度。
    def get_action_size(self):
        return 9  # 3x3 board 

    # 获取 1D 动作掩码（用于 NN 输出）
    # 专门为神经网络输出和训练使用的方法
    def get_flat_valid_moves(self):
        # 创建一个长度为 9 的一维数组，全是 0：
        valid = np.zeros(9, dtype=np.float32)
        # 遍历所有合法落子，并将它们变成1，之后放进valid中
        for i, j in self.get_valid_moves():
            valid[i * 3 + j] = 1
        return valid

    """
        AAA 为了训练强化学习模型后加的函数
    """
    def get_flat_board(self):
        return self.board.flatten().astype(np.float32)
