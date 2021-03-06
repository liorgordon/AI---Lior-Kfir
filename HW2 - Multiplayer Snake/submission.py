import random
import time
from copy import copy
from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
from scipy.spatial.distance import cityblock


def _fruit_distance(state: GameState, player_index: int) -> float:
    snake_manhattan_dists = sorted([cityblock(state.snakes[player_index].head, trophy_i)
                                    for trophy_i in state.fruits_locations])
    if not snake_manhattan_dists:
        return 0
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
        return my_snake.length
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

    def minimax(self, state : TurnBasedGameState, D : int):
        if D == 0 or not state.game_state.snakes[self.player_index].alive or state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        best_value = -np.inf
        worst_value = np.inf
        if state.turn == self.Turn.AGENT_TURN:
            for our_action in state.game_state.get_possible_actions(player_index=self.player_index):
                h_value = self.minimax(self.TurnBasedGameState(state.game_state, our_action), D)
                if h_value > best_value:
                    best_value = h_value
            return best_value
        else:
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                        player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                h_value = self.minimax(self.TurnBasedGameState(next_state, None), D-1)
                if h_value < worst_value:
                    worst_value = h_value
            return worst_value

    def get_action(self, state: GameState) -> GameAction:
        if self.curr_turn is None:
            self.curr_turn = self.TurnBasedGameState(state, None)
        best_value = -np.inf
        best_actions = []
        self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
        for our_action in state.get_possible_actions(player_index=self.player_index):
            h_value = self.minimax(self.TurnBasedGameState(state, our_action), 2)
            if h_value > best_value:
                best_value = h_value
                best_actions = [our_action]
            elif h_value == best_value:
                best_actions.append(our_action)
        return np.random.choice(best_actions)



class AlphaBetaAgent(MinimaxAgent):
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

    def abminimax(self, state : TurnBasedGameState, D : int, alpha, beta):
        if D == 0 or not state.game_state.snakes[self.player_index].alive or state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        best_value = -np.inf
        worst_value = np.inf
        if state.agent_action is None:
            for our_action in state.game_state.get_possible_actions(player_index=self.player_index):
                h_value = self.abminimax(self.TurnBasedGameState(state.game_state, our_action), D, alpha, beta)
                best_value = max(h_value, best_value)
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    return np.inf
            return best_value
        else:
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                        player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                h_value = self.abminimax(self.TurnBasedGameState(next_state, None), D-1, alpha, beta)
                worst_value = min(worst_value, h_value)
                beta = min(worst_value, beta)
                if worst_value <= alpha:
                    return -np.inf
            return worst_value

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        if self.curr_turn is None:
            self.curr_turn = self.TurnBasedGameState(state, None)
        best_value = -np.inf
        best_actions = []
        self.curr_turn.curr_turn = MinimaxAgent.Turn.OPPONENTS_TURN
        for our_action in state.get_possible_actions(player_index=self.player_index):
            h_value = self.abminimax(self.TurnBasedGameState(state, our_action), 2,  -np.inf, np.inf)
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
    N = 50
    init_state = [GameAction.STRAIGHT] * N
    sideways = 0
    limit = N/5
    for i in range(N):
        best_val = np.NINF
        best_states = None
        for j in range(3):
            tmp = init_state.copy()
            if j == 0:
                tmp[i] = GameAction.RIGHT
            elif j == 1:
                tmp[i] = GameAction.STRAIGHT
            elif j == 2:
                tmp[i] = GameAction.LEFT
            new_val = get_fitness(tmp)
            if new_val == best_val:
                best_states.append(tmp)

            if new_val > best_val:
                best_val = new_val
                best_states = [tmp]
        state_fitness = get_fitness(init_state)
        if best_val > state_fitness:
            chosen_state = np.random.choice(len(best_states))
            init_state = best_states[chosen_state]
            sideways = 0
            M=best_val
        elif best_val == state_fitness and sideways <= limit:
            chosen_state = np.random.choice(len(best_states))
            init_state = best_states[chosen_state]
            sideways = sideways + 1
    print("the best combination I found was {} ".format(init_state))
    print("and I got " + str(M))


def check_insert_to_newbeam(NewBeam, item, k):
    item[2] += 1
    item[1] = get_fitness(item[0])
    if len(NewBeam) < k:
        NewBeam.append([item[0].copy(), item[1], item[2]])
    elif NewBeam[0][1] < item[1]:
        NewBeam[0] = [item[0].copy(), item[1], item[2]]
    pass


def local_search():
    k = 4
    N = 50
    NewBeam = [[np.random.choice(list(GameAction), p=[0.1, 0.8, 0.1]) for _ in range(N)] for _ in range(k)]
    NewBeam = [[state, get_fitness(state), 0] for state in NewBeam]
    NewBeam = sorted(NewBeam, key=lambda x:x[1])
    Beam = []
    while True:
        if Beam and Beam[-1][1] == NewBeam[-1][1]:
            break
        Beam = NewBeam.copy()
        NewBeam = []
        for item in Beam:
            cur_turn = item[2]
            if cur_turn == N:
                check_insert_to_newbeam(NewBeam, item.copy(), k)
                NewBeam = sorted(NewBeam, key=lambda x: x[1])
            for j in range(3):
                if j == 0:
                    item[0][cur_turn] = GameAction.RIGHT
                elif j == 1:
                    item[0][cur_turn] = GameAction.STRAIGHT
                elif j == 2:
                    item[0][cur_turn] = GameAction.LEFT
                check_insert_to_newbeam(NewBeam, item.copy(), k)
                NewBeam = sorted(NewBeam, key=lambda x: x[1])
    print("my moves were {}".format(Beam[-1][0]))
    print("and I got {}".format(Beam[-1][1]))


class TournamentAgent(Player):
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

    def abminimax(self, state: TurnBasedGameState, D: int, alpha, beta):
        if D == 0 or not state.game_state.snakes[self.player_index].alive or state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        best_value = -np.inf
        worst_value = np.inf
        if state.agent_action is None:
            for our_action in state.game_state.get_possible_actions(player_index=self.player_index):
                h_value = self.abminimax(self.TurnBasedGameState(state.game_state, our_action), D, alpha, beta)
                if h_value > best_value:
                    best_value = h_value
                    alpha = max(alpha, best_value)
                if best_value >= beta:
                    # print("cut beta in depth: {}, beta is: {}, alpha is: {}".format(D, beta, alpha))
                    return np.inf
            # if not state.game_state.snakes[self.player_index].alive:
            # print("i entered agent with a dead snake, returning {}", format(best_value))
            return best_value
        else:
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                # print("entered None ")
                h_value = self.abminimax(self.TurnBasedGameState(next_state, None), D - 1, alpha, beta)
                if h_value < worst_value:
                    worst_value = h_value
                    beta = min(worst_value, beta)
                if worst_value <= alpha:
                    # print("cut alpha in depth: {}, beta is: {}, alpha is: {}".format(D, beta, alpha))
                    return -np.inf
            return worst_value

    def get_action(self, state: GameState) -> GameAction:
        D_arr = [2, 3, 4]

        best_value = -np.inf
        best_actions = []
        i = 0
        for our_action in state.get_possible_actions(player_index=self.player_index):
            t = time.time()
            h_value = self.abminimax(self.TurnBasedGameState(state, our_action), D_arr[i], -np.inf, np.inf)
            elapsed = time.time() - t
            if elapsed < 15 and i < 2:
                i += 1
            if elapsed > 20 and i > 0:
                i -= 1
            # elif elapsed
            if h_value > best_value:
                best_value = h_value
                best_actions = [our_action]
            elif h_value == best_value:
                best_actions.append(our_action)
        return np.random.choice(best_actions)


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
