from environment import Grid2DSize, SnakesBackendSync
from agents import RandomPlayer, KeyboardPlayer, GreedyAgent, BetterGreedyAgent
from submission import MinimaxAgent, AlphaBetaAgent, TournamentAgent
from optparse import OptionParser
import sys
from main import start_custom_game

if __name__ == '__main__':
    t_greedy = {'time': 0, 'score': 0}
    t_Bgreedy = {'time': 0, 'score': 0}
    t_Minimax = {'time': 0, 'score': 0}
    t_ABMinimax = {'time': 0, 'score': 0}

    for i in range(10):
        # # Greedy
        # start_custom_game('GreedyAgent', 'GreedyAgent', 500, 50, 50, 51, False, True, False, t=t_greedy)
        #
        # # Better Greedy
        # start_custom_game('BetterGreedyAgent', 'GreedyAgent', 500, 50, 50, 51, False, True, False, t=t_Bgreedy)

        # Minimax
        start_custom_game('MinimaxAgent', 'GreedyAgent', 500, 50, 50, 51, False, True, False, t=t_Minimax)

        # AB minimax
        start_custom_game('AlphaBetaAgent', 'GreedyAgent', 500, 50, 50, 51, False, True, False, t=t_ABMinimax)

    f = open("experiments.csv", 'a')
    # f.write("GreedyAgent, 1, {}, {}\n".format(t_greedy["time"], t_greedy["score"]))
    # f.write("BetterGreedyAgent, 1, {}, {}".format(t_Bgreedy["time"], t_Bgreedy["score"]))
    f.write("MinimaxAgent, 4, {}, {}\n".format(t_Minimax["time"]/10, t_Minimax["score"]/10))
    f.write("AlphaBetaAgent, 4, {}, {}\n".format(t_ABMinimax["time"]/10, t_ABMinimax["score"]/10))
    f.close()
