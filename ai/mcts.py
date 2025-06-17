# ai/mcts.py
import numpy as np
import copy
import torch
import torch.nn.functional as F

# 树节点定义
class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = copy.deepcopy(game) # 保存这个节点的游戏状态
        self.parent = parent # 父节点
        self.move = move # 从父节点走到这里的动作
        self.children = {} # 子节点: move -> MCTSNode
        self.visit_count = 0 # N(s)
        self.total_value = 0.0 # 累计值 Q(s)
        self.prior = 0.0  # 策略网络输出的 prior probability π(a)
        self.player = game.current_player

    # 获取平均值 Q(s, a)
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

class MCTS:
    def __init__(self, model, simulations=100, c_puct=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    # 启动搜索
    def run(self, root_game):
        root = MCTSNode(root_game)

        # 第一次调用模型估值所有动作的先验概率 π
        board_tensor = torch.tensor(root_game.get_flat_board(), dtype=torch.float32).view(1, 1, 3, 3)
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()

        for move in root_game.get_valid_moves():
            idx = move[0] * 3 + move[1]
            child_game = copy.deepcopy(root_game)
            child_game.make_move(move)
            child = MCTSNode(child_game, parent=root, move=move)
            child.prior = policy[idx]
            root.children[move] = child

        for _ in range(self.simulations):
            self.simulate(root)

        return root

    def simulate(self, node):
        if node.game.is_terminal():
            winner = node.game.check_winner()
            return 1 if winner == node.player else -1 if winner == -node.player else 0

        if not node.children:
            board_tensor = torch.tensor(node.game.get_flat_board(), dtype=torch.float32).view(1, 1, 3, 3)
            with torch.no_grad():
                policy_logits, value = self.model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
            value = value.item()

            for move in node.game.get_valid_moves():
                idx = move[0] * 3 + move[1]
                child_game = copy.deepcopy(node.game)
                child_game.make_move(move)
                child = MCTSNode(child_game, parent=node, move=move)
                child.prior = policy[idx]
                node.children[move] = child

            return value

        # 选择最佳子节点
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        for move, child in node.children.items():
            u = self.c_puct * child.prior * np.sqrt(total_visits + 1) / (1 + child.visit_count)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        v = self.simulate(node.children[best_move])
        node.children[best_move].total_value += v
        node.children[best_move].visit_count += 1
        return -v  # 因为视角要对调

    def get_pi(self, root, temperature=1.0):
        pi = np.zeros(9)
        visits = np.array([root.children[m].visit_count for m in root.children])
        moves = list(root.children.keys())
        if temperature == 0:
            best_idx = np.argmax(visits)
            pi[moves[best_idx][0] * 3 + moves[best_idx][1]] = 1.0
        else:
            visits = visits ** (1 / temperature)
            visits = visits / np.sum(visits)
            for i, move in enumerate(moves):
                idx = move[0] * 3 + move[1]
                pi[idx] = visits[i]
        return pi
