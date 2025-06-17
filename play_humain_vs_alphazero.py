# play_human_vs_alpha.py
import torch
import numpy as np
from ai.alpha_model import AlphaZeroNet
from ai.mcts import MCTS
from game.morpion import MorpionGame

def print_board(game):
    print("\n  1 2 3")
    for i in range(3):
        row = str(i + 1) + " "
        for j in range(3):
            val = game.board[i][j]
            if val == 1:
                row += "X "
            elif val == -1:
                row += "O "
            else:
                row += ". "
        print(row)

def get_human_move(game):
    while True:
        move = input("Votre tour (format 1,3): ")
        try:
            i, j = map(int, move.strip().split(','))
            i -= 1
            j -= 1
            if (i, j) in game.get_valid_moves():
                return (i, j)
            else:
                print("Coup invalide, essayez à nouveau.")
        except:
            print("Format invalide. Exemple correct : 2,1")

def ai_move(game, model, simulations=50):
    mcts = MCTS(model, simulations)
    root = mcts.run(game)  # 保存根节点
    pi = mcts.get_pi(root, temperature=0)
    action_index = np.argmax(pi)  # deterministic
    return (action_index // 3, action_index % 3)

def main():
    model = AlphaZeroNet()
    model.load_state_dict(torch.load("checkpoint_alpha_iter100.pt"))
    model.eval()

    print("Bienvenue dans Morpion avec AlphaZero AI !")
    player = int(input("Jouez-vous en premier (1) ou en second (-1) ? : "))

    game = MorpionGame()
    game.current_player = 1  # X always starts

    print_board(game)

    while not game.is_terminal():
        if game.current_player == player:
            move = get_human_move(game)
        else:
            move = ai_move(game, model)
            print(f"L'IA joue : {move[0]+1}, {move[1]+1}")
        game.make_move(move)
        print_board(game)

    winner = game.check_winner()
    if winner == 0:
        print("Match nul !")
    elif winner == player:
        print("Félicitations, vous avez gagné !")
    else:
        print("L'IA a gagné !")

if __name__ == "__main__":
    main()
