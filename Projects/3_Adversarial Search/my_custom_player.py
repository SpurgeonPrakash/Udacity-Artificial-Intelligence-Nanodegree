import random
import math

from sample_players import DataPlayer
from math import sqrt

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # No Iterative Deepening
            move, value = self.alpha_beta(state, 3)
            self.queue.put(move)

            # Iterative Deepening
            # best_move = random.choice(state.actions())
            # best_score = float('-inf')
            # depth_limit = 100
            
            # for depth in range(1, depth_limit + 1): 
            #     move, value = self.alpha_beta(state, depth)
                                  
            #     if value > best_score:
            #         best_move = move
            #         best_score = value
                    
            #     self.queue.put(best_move)
        return
    
    ##  Alpha Beta Search Algorithm 
    def alpha_beta(self, state, depth):

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            
            value = float("inf")
            for move in state.actions():
                value = min(value, max_value(state.result(move), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value
        
        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            
            value = float("-inf")
            for move in state.actions():
                value = max(value, min_value(state.result(move), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
        
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        for move in state.actions():
            value = min_value(state.result(move), depth - 1, alpha, beta)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = move
        return best_move, best_score
    
    ##  Minimax Search Algorithm 
    def minimax(self, state, depth):
        def min_val(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            value = float("inf")
            for move in state.actions():
                value = min(value, max_val(state.result(move), depth - 1))
            return value

        def max_val(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            value = float("-inf")
            for move in state.actions():
                value = max(value, min_val(state.result(move), depth - 1))
            return value

        best_score = float("-inf")
        best_move = None
        
        for move in state.actions():
            value = min_value(state.result(move), depth - 1)
            
            if value > best_score:
                best_score = value
                best_move = move
                
        return best_move, best_score

# baseline score
def score(state, player_id, heuristic = 2):
    own_loc = state.locs[player_id]
    opp_loc = state.locs[1 - player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    own_moves = len(own_liberties)
    opp_moves = len(opp_liberties)
    ply_count = state.ply_count

    def index_xy(location):
        x = (location - 1) % 11
        y = (location - 1) / 11
        width = 11
        height = 9
        return ((width / 2) - x), ((height / 2) - y)

    if own_moves == 0: return float("inf")
    if opp_moves == 0: return float("-inf")

    # custom score
    if heuristic == 2:
        own_x, own_y = index_xy(own_loc);
        opp_x, opp_y = index_xy(opp_loc);
        manhattan_distance_own = abs(own_x) + abs(own_y)
        manhattan_distance_opp = abs(opp_x) + abs(opp_y)
        norm = sqrt(11/2*11/2 + 9/2*9/2)
        normalized_distance_own = sqrt(own_x * own_x + own_y * own_y)/norm
        normalized_distance_opp = sqrt(opp_x * opp_x + opp_y * opp_y)/norm
        own_moves_future  = sum(len(state.liberties(liberty)) for liberty in own_liberties)
        opp_moves_future = sum(len(state.liberties(liberty)) for liberty in opp_liberties)
        overlap = len([score for score in opp_liberties if score in own_liberties])
        score = 0.0
        if player_id == 0:
            if overlap > 0:
                score += 10.0
        else:
            if overlap > 0:
                score -= 10.0
        return score*((normalized_distance_opp+1.0)*own_moves-(normalized_distance_own+1.0)*opp_moves)

    # baseline score
    elif heuristic == 1:
        return own_moves - opp_moves