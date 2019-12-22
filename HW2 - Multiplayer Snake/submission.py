from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
from scipy.spatial.distance import cityblock


def _fruit_distance(state: GameState, player_index: int) -> float:
    snake_manhattan_dists = sorted([cityblock(state.snakes[player_index].head, trophy_i)
                                    for trophy_i in state.fruits_locations])
    return (1 / (snake_manhattan_dists[0]+1)) * 100


def _wall_distance(state, player_index):
    snake_head = state.snakes[player_index].head
    x_dist = min(state.board_size[0]-snake_head[1], snake_head[1])
    y_dist = min(state.board_size[1] - snake_head[0], snake_head[0])
    return min(x_dist, y_dist)*0.5

def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    my_snake = state.snakes[player_index]
    fruit_cost = _fruit_distance(state, player_index)
    wall_cost = _wall_distance(state, player_index)
    if not my_snake.alive:
        return -1
    return my_snake.length*200 + fruit_cost + wall_cost*1.25
    pass


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    depth = 0
    curr_turn = None


    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action


        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def minimax(self, state, action):
        # if self.curr_turn.curr_turn == MinimaxAgent.Turn.AGENT_TURN:
        #     print("entered minimax for agent, depth = {}\n".format(self.depth))
        # if self.curr_turn.curr_turn == MinimaxAgent.Turn.OPPONENTS_TURN:
        #     print("entered minimax for opponent, depth = {}\n".format(self.depth))
        # if not state.is_within_grid_boundaries(state.snakes[self.player_index].head):
        #     print("in the fucking if")
        # if not state.snakes[self.player_index].alive:
        #     print("in the fucking if2")
        if self.depth > 3:
            return heuristic(state, self.player_index)
        best_value = -np.inf
        worst_value = np.inf
        self.depth = self.depth + 1
        if self.curr_turn.curr_turn == MinimaxAgent.Turn.AGENT_TURN:
            for our_action in state.get_possible_actions(player_index=self.player_index):
                self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
                h_value = self.minimax(state, our_action)
                if h_value > best_value:
                    best_value = h_value
            self.depth = self.depth - 1
            return best_value
        else:
            for opponents_actions in state.get_possible_actions_dicts_given_action(action,
                                                                        player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                self.curr_turn.curr_turn = MinimaxAgent.Turn.AGENT_TURN
                h_value = self.minimax(next_state, None)
                if h_value < worst_value:
                    worst_value = h_value
            self.depth = self.depth - 1
            return worst_value

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        if self.curr_turn is None:
            self.curr_turn = self.TurnBasedGameState(state, None)
        best_value = -np.inf
        best_actions = []
        self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
        for our_action in state.get_possible_actions(player_index=self.player_index):
            h_value = self.minimax(state, our_action)
            if h_value > best_value:
                best_value = h_value
                best_actions = [our_action]
            elif h_value == best_value:
                best_actions.append(our_action)
        return np.random.choice(best_actions)



class AlphaBetaAgent(MinimaxAgent):
    depth = 0
    curr_turn = None

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def abminimax(self, state, action, alpha, beta):
        # if self.curr_turn.curr_turn == MinimaxAgent.Turn.AGENT_TURN:
        #     print("entered ab for agent, depth = {}\n".format(self.depth))
        # if self.curr_turn.curr_turn == MinimaxAgent.Turn.OPPONENTS_TURN:
        #     print("entered ab for opponent, depth = {}\n".format(self.depth))
        # if not state.is_within_grid_boundaries(state.snakes[self.player_index].head):
        #     print("in the fucking if")
        # if not state.snakes[self.player_index].alive:
        #     print("in the fucking if2")
        if self.depth > 3:
            return heuristic(state, self.player_index)
        best_value = -np.inf
        worst_value = np.inf
        self.depth = self.depth + 1
        if self.curr_turn.curr_turn == MinimaxAgent.Turn.AGENT_TURN:
            for our_action in state.get_possible_actions(player_index=self.player_index):
                self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
                h_value = self.abminimax(state, our_action, alpha, beta)
                if h_value > best_value:
                    best_value = h_value
                    alpha = max(alpha, best_value)
                if best_value >= beta:
                    print("cut beta in depth: {}, beta is: {}, alpha is: {}".format(self.depth, beta, alpha))
                    self.depth = self.depth - 1
                    return np.inf
            self.depth = self.depth - 1
            return best_value
        else:
            for opponents_actions in state.get_possible_actions_dicts_given_action(action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                self.curr_turn.curr_turn = MinimaxAgent.Turn.AGENT_TURN
                h_value = self.abminimax(next_state, None, alpha, beta)
                if h_value < worst_value:
                    worst_value = h_value
                    beta = min(worst_value, beta)
                if worst_value <= alpha:
                    print("cut alpha in depth: {}, beta is: {}, alpha is: {}".format(self.depth, beta, alpha))
                    self.depth = self.depth - 1
                    return -np.inf
            self.depth = self.depth - 1
            return worst_value

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        if self.curr_turn is None:
            self.curr_turn = self.TurnBasedGameState(state, None)
        best_value = -np.inf
        best_actions = []
        self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
        for our_action in state.get_possible_actions(player_index=self.player_index):
            h_value = self.abminimax(state, our_action, -np.inf, np.inf)
            if h_value > best_value:
                best_value = h_value
                best_actions = [our_action]
            elif h_value == best_value:
                best_actions.append(our_action)
        return np.random.choice(best_actions)


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
