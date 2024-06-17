import networkx as nx
from copy import deepcopy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from queue import Queue
import threading

K = 3
M = 2
N = 2
MAXLOOKAHEAD = 32

animation_queue = Queue()

fig, ax = plt.subplots()

G = nx.Graph()
G.add_node(0)

rng = np.random.default_rng()

def initialize_forbidden_vertices(G : nx.Graph):
	for v in G.nodes:
		G.nodes[v]['forbidden'] = False

initialize_forbidden_vertices(G)

def avoider_possibilities(G : nx.Graph, width = None):
	#vertices = rng.integers(low = 0, high=len(G.nodes), size=search_width)
	n = len(G.nodes)
	if n == 0:
		return

	shuffled_vertices = list(range(n))
	rng.shuffle(shuffled_vertices)
	
	if width == None:
		for v in shuffled_vertices:
			if not G.nodes[v]['forbidden']:
				g = deepcopy(G)
				g.nodes[v]['forbidden'] = True
				yield g
	else:
		for i in range(width):
			if not G.nodes[shuffled_vertices[i]]['forbidden']:
				g = deepcopy(G)
				g.nodes[shuffled_vertices[i]]['forbidden'] = True
				yield g


def builder_possibilities(G : nx.Graph):
	n = len(G.nodes)
	if n == 0:
		return
	
	shuffled_us = list(range(n-1))
	rng.shuffle(shuffled_us)
	for u in shuffled_us:
		shuffled_vs = list(range(u+1, n))
		rng.shuffle(shuffled_vs)
		for v in shuffled_vs:
			if u < v and\
				(u,v) not in G.edges \
					and ( not G.nodes[u]['forbidden'] or \
						not G.nodes[v]['forbidden'] ):
				g = deepcopy(G)
				g.add_edge(u,v)
				yield g
	shuffled_us = list(range(n))
	rng.shuffle(shuffled_us)
	for u in shuffled_us:
		g = deepcopy(G)
		g.add_node(n)
		g.nodes[n]['forbidden'] = False
		g.add_edge(u,n)
		yield g

def builder_turn_possibilities(G : nx.Graph, m : int):
	if m <= 0:
		yield G
	if m >= 1:
		for g in builder_possibilities(G):
			yield from builder_turn_possibilities(g, m-1)

def avoider_turn_possibilities(G : nx.Graph, n : int):
	if n <= 0:
		yield G
	if n >= 1:
		for g in avoider_possibilities(G):
			nextround = list(avoider_turn_possibilities(g, n-1))
			if len(nextround) == 0:
				yield g
			else:
				yield from nextround

def detect_builder_win(G : nx.Graph):
	return max(nx.node_clique_number(G).values()) >= K

def builder_turn(G : nx.Graph, m : int, n : int, maxlookahead : int, ax):
	global animation_queue

	turnnum = MAXLOOKAHEAD - maxlookahead

	if maxlookahead <= 0:
		return [([next(builder_turn_possibilities(G, m))], MAXLOOKAHEAD)] #possibly makes avoider more optimistic...

	possible_moves = list(builder_turn_possibilities(G, m))
	
	for move in possible_moves:
		if detect_builder_win(move):
			if ax : 
				f = (move, ax, f"Turn {turnnum} : buillder's winning move")
				animation_queue.put(f)
			return [([move], 0)]
		
	futures = []
	for move in possible_moves:
		afterg = avoider_turn(move, m, n, maxlookahead-1, ax=ax)
		for future, cost in afterg:
			futures.append(([move]+future, cost))

	future, best_cost = futures[0]
	best_futures = [(future, best_cost)]
	
	for future, cost in futures:
		if cost < best_cost:
			best_futures = []
			best_futures.append((future, cost + 1))
			best_cost = cost
		elif cost == best_cost:
			best_futures.append((future, cost + 1))
	if ax :
		for best_future in best_futures:
			f = (best_future[0][0], ax, f"Turn {turnnum} : buillder's best move")
			animation_queue.put(f)
	return best_futures

def avoider_turn(G : nx.Graph, m: int, n : int, maxlookahead : int, ax):
	global animation_queue

	turnnum = MAXLOOKAHEAD - maxlookahead

	possible_moves = list(avoider_turn_possibilities(G, n))
	
	if len(possible_moves) == 0:
		"""
		With this code, the avoider/forbidder wins when all vertices are forbidden.
		"""
		animation_queue.put((G, ax, f"Turn {turnnum} : avoider wins"))
		return [([], MAXLOOKAHEAD)]
		
	if maxlookahead <= 0:
		return [([possible_moves[0]], MAXLOOKAHEAD)] #possibly makes avoider more optimistic...
	
	futures = []
	for move in possible_moves:
		afterg = builder_turn(move, m, n, maxlookahead-1, ax=ax)
		for future, cost in afterg:
			futures.append(([move]+future, cost))
	
	future , best_cost = futures[0]
	best_futures = [(future,best_cost)]
	for future, cost in futures:
		if cost > best_cost:
			best_futures = []
			best_futures.append((future, cost+1))
			best_cost = cost
		elif cost == best_cost:
			best_futures.append((future, cost+1))
	if ax :
		for best_future in best_futures:
			f = (best_future[0][0], ax, f"Turn {turnnum} : avoider's best move")
			animation_queue.put(f)
	return best_futures
	
def animate(argframe):
	g, ax, title = argframe
	ax.clear()
	ax.set_title(title)
	color_map = deque('red' if g.nodes[node]['forbidden'] else 'blue' for node in g)
	nx.draw(g, node_color=color_map, with_labels=True)

def game_frames():
	global animation_queue

	while True:
		frame = animation_queue.get()
		animation_queue.task_done()
		if frame:
			yield frame
		else:
			return

def game_worker():
	global history
	global cost
	global animation_queue
	history = builder_turn(G, M, N, MAXLOOKAHEAD, ax=ax)
	animation_queue.put(None)

def main():		
	global history
	game_thread = threading.Thread(target = game_worker, daemon=True)
	
	game_thread.start()
	
	anim = FuncAnimation(fig, animate, frames=game_frames, interval=100, cache_frame_data=False, repeat=False)

	plt.show()

	game_thread.join()

	for k, future in enumerate(history):
		print(f"future # {k}")
		print(future)
			

	n = len(history[0][0])
	for turnnum, graph in enumerate(history[0][0]):
		ax.clear()
		ax.set_title(f"Turn {turnnum} : {'builder' if turnnum % 2 == 1 else 'avoider'} : {'wins' if turnnum == n else ''}")
		color_map = deque('red' if graph.nodes[node]['forbidden'] else 'blue' for node in graph)
		nx.draw(graph, node_color=color_map, with_labels=True)
		plt.show()

if __name__ == '__main__':
	main()