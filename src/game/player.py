import networkx as nx
import random
from base_player import BasePlayer
from settings import *
from math import erf

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
        self.stations = set()
        return

    def should_build(self, state):
        return ((state.get_time() * 1.0 / GAME_LENGTH) <= 0.15 and
                len(self.stations) < max(HUBS, 2))

    def update_station_scores(self, state, new_order):
        new_scores = [0] * len(self.station_scores)
        neg_scores = [0] * len(self.station_scores)
        def BFS(graph, nodes, iteration):
            next = []
            if iteration > ORDER_VAR * 3:
                return
            for node in nodes:
                new_value = 1 - erf(iteration * 1.0 / ORDER_VAR)
                if new_scores[node] < new_value:
                    new_scores[node] = new_value
                    next += graph.neighbors(node)
            BFS(graph, list(set(next)), iteration + 1)
        def negBFS(graph, nodes, iteration):
            next = []
            if iteration > ORDER_VAR * 3:
                return
            for node in nodes:
                new_value = (erf(iteration * 1.0 / ORDER_VAR) - 1) / ORDER_VAR
                if neg_scores[node] > new_value:
                    neg_scores[node] = new_value
                    next += graph.neighbors(node)
            negBFS(graph, list(set(next)), iteration + 1)
        BFS(state.get_graph(), [new_order.get_node()], 0)
        negBFS(state.get_graph(), self.stations, 0)
        self.station_scores = map(lambda (x,y,z): x+y+z,
                zip(new_scores, neg_scores, self.station_scores))

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

        removed = self.removeUsedEdges(graph)

        pending_orders = state.get_pending_orders()

        if (len(pending_orders) > 0 and
                pending_orders[-1].get_time_created() == state.get_time()):
            self.update_station_scores(state, pending_orders[-1])
        """
        self.station_scores = map(lambda x: x*0.9, self.station_scores)
        for new_order in pending_orders:
            self.update_station_scores(state, new_order)
        """


        commands = []

        if len(self.stations) == 0: # first station
            initial_order = pending_orders[0]
            station = self.valuationFunction(graph, initial_order, .25, .5)
            commands.append(self.build_command(station))
            self.stations.add(station)
            self.has_built_station = True
        elif self.should_build(state):
            if state.get_money() >= INIT_BUILD_COST * (BUILD_FACTOR**len(self.stations)):
                #We have enough money
                n_tuples = sorted([(self.station_scores[i], i) for i in xrange(len(self.station_scores))])[::-1]
                for n in n_tuples:
                    if not n[1] in self.stations:
                        self.stations.add(graph.nodes()[n[1]])
                        commands.append(self.build_command(graph.nodes()[n[1]]))

                        node = graph.nodes()[n[1]]
                        to_remove = None
                        for i in xrange(len(pending_orders)):
                            if pending_orders[i].get_node() == node:
                                commands.append(self.send_command(pending_orders[i], [node]))
                                to_remove = i
                                break
                        if to_remove != None:
                            del pending_orders[to_remove]

                        break


        # Try to send orders until we have none left
        while len(pending_orders) != 0 and len(self.stations) > 0:
            # Get the best possible order to satisfy first
            bestOrder = (0, [], None)
            for order in pending_orders:
                # Find the best station for this order
                highVal, highPath = self.findBestStation(removed, order)

                if highVal > bestOrder[0]:
                    bestOrder = (highVal, highPath, order)

            # Add the order command if we could find one
            if bestOrder[0] > SCORE_MEAN / 100.0:
                commands.append(self.send_command(bestOrder[2], bestOrder[1]))
            else:
                break

            # Remove this order and path and repeat
            pending_orders.remove(order)

            for i in xrange(len(bestOrder[1]) - 1):
                removed.remove_edge(bestOrder[1][i], bestOrder[1][i+1])

        return commands

    def findBestStation(self, graph, order):
        def mapToVal(station):
            try:
                path = nx.shortest_path(graph, station, order.get_node())
            except nx.NetworkXNoPath:
                return (-float('inf'), None)

            return ((order.get_money() - DECAY_FACTOR * len(path)), path)

        def reduceToPath((m1, p1), (m2, p2)):
            return (m1, p1) if m1 > m2 else (m2, p2)

        bestPath = reduce(reduceToPath, map(mapToVal, self.stations))
        return bestPath

    def removeUsedEdges(self, graph):
        removed = graph.copy()

        def isInUse(n):
            return n[2]["in_use"]

        used_edges = filter(isInUse, removed.edges(data=True))

        removed.remove_edges_from(used_edges)
        return removed

    def valuationFunction(self, graph, order, DIST_VAL, CONNECT_VAL):
        distances = nx.shortest_path(graph, target=order.get_node())
        distances = {node : len(distances[node]) for node in distances}

        out_edges = {node: len(graph.neighbors(node)) for node in graph.nodes()}

        results = [(node,
                    distances[node] * -DIST_VAL +
                    out_edges[node] * CONNECT_VAL) for node in distances]

        def bestValue((x1, v1), (x2, v2)):
            return (x1, v1) if v1 > v2 else (x2, v2)

        return reduce(bestValue, results)[0]
