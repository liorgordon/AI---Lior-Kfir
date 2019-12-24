from environment import Grid2DSize, SnakesBackendSync
from agents import RandomPlayer, KeyboardPlayer, GreedyAgent, BetterGreedyAgent
from submission import MinimaxAgent, AlphaBetaAgent, TournamentAgent
from optparse import OptionParser
import sys
from main import get_user_command


if __name__ == '__main__':

    for i in range(10):
        # Greedy
        get_user_command('--custom_game --p1 Greedy --p2 Greedy', t)

        # Better Greedy
        get_user_command('--custom_game --p1 BetterGreedyAgent --p2 Greedy')

        # Minimax
        get_user_command('--custom_game --p1 MinimaxAgent --p2 Greedy ')

        # AB minimax
        get_user_command('--custom_game --p1 AlphaBetaAgent --p2 Greedy ')
