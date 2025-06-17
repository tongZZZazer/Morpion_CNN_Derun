# ai/self_play.py
import torch
import numpy as np
from ai.mcts import MCTS
from game.morpion import MorpionGame

def encode_board(game):
    board = game.board.astype(np.float32)
    return board

def play_self_game(model, simulations=50, temperature=1.0):
    game = MorpionGame()
    mcts = MCTS(model, simulations)
    memory = []

    while not game.is_terminal():
        state = encode_board(game)
        root = mcts.run(game)                             # 得到 MCTS 根节点
        pi = mcts.get_pi(root, temperature=temperature)   # 正确获取 π 概率分布

        memory.append((state, pi, game.current_player))

        # 从 π 分布中采样一个动作
        action_idx = np.random.choice(len(pi), p=pi)
        move = (action_idx // 3, action_idx % 3)
        game.make_move(move)

    # 游戏结束后计算 z 值（胜负）
    winner = game.check_winner()
    data = []
    for state, pi, player in memory:
        if winner == 0:
            z = 0
        else:
            z = 1 if winner == player else -1
        data.append((state, pi, z))

    return data
