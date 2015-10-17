import networkx as nx
import random
from base_player import BasePlayer
from settings import *

class Player(BasePlayer):
    """
    You will implement this class for the competition. DO NOT change the class
    name or the base class.
    """

    # You can set up static state here
    has_built_station = False

    def __init__(self, state):
        """
        Initializes your Player. You can set up persistent state, do analysis
        on the input graph, engage in whatever pre-computation you need. This
        function must take less than Settings.INIT_TIMEOUT seconds.
        --- Parameters ---
        state : State
            The initial state of the game. See state.py for more information.
        """
        self.station_scores = [0] * len(state.get_graph().nodes())
        return

    def update_station_scores(self, state, new_order):
        new_scores = [0] * len(self.station_scores)
        def BFS(graph, nodes, money):
            next = []
            if money <= 0:
                return
            for node in nodes:
                if new_scores[node] < money:
                    new_scores[node] = money
                    next += graph.neighbors(node)
            BFS(graph, list(set(next)), money - DECAY_FACTOR)
        BFS(state.get_graph(), [new_order.get_node()], new_order.get_money())
        self.station_scores = map(lambda (x,y): x+y,
                zip(new_scores, self.station_scores))

    # Checks if we can use a given path
    def path_is_valid(self, state, path):
        graph = state.get_graph()
        for i in range(0, len(path) - 1):
            if graph.edge[path[i]][path[i + 1]]['in_use']:
                return False
        return True

    def step(self, state):
        """
        Determine actions based on the current state of the city. Called every
        time step. This function must take less than Settings.STEP_TIMEOUT
        seconds.
        --- Parameters ---
        state : State
            The state of the game. See state.py for more information.
        --- Returns ---
        commands : dict list
            Each command should be generated via self.send_command or
            self.build_command. The commands are evaluated in order.
        """

        # We have implemented a naive bot for you that builds a single station
        # and tries to find the shortest path from it to first pending order.
        # We recommend making it a bit smarter ;-)

        graph = state.get_graph()
        station = graph.nodes()[0]

        pending_orders = state.get_pending_orders()
        
        if (len(pending_orders) > 0 and
                pending_orders[-1].get_time_created() == state.get_time()):
            new_order = pending_orders[-1]
            self.update_station_scores(state, new_order)
            print max(self.station_scores), self.station_scores.index(max(self.station_scores))

        commands = []
        if not self.has_built_station:
            commands.append(self.build_command(station))
            self.has_built_station = True

        if len(pending_orders) != 0:
            order = random.choice(pending_orders)
            path = nx.shortest_path(graph, station, order.get_node())
            if self.path_is_valid(state, path):
                commands.append(self.send_command(order, path))

        return commands
