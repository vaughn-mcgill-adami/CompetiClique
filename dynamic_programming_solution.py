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

K = 4
M = 2
N = 2
MAXLOOKAHEAD = 16

animation_queue = Queue()

history = None

fig, ax = plt.subplots()

G = nx.Graph()
G.add_node(0)

rng = np.random.default_rng()

def initialize_forbidden_vertices(G : nx.Graph):
	for v in G.nodes:
		G.nodes[v]['forbidden'] = False

initialize_forbidden_vertices(G)

def avoider_possibilities(G : nx.Graph, width = 2):
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
				yield g, v
	else:
		for u in shuffled_vertices[:width]:
			if not G.nodes[u]['forbidden']:
				g = deepcopy(G)
				g.nodes[u]['forbidden'] = True
				yield g, u


def builder_possibilities(G : nx.Graph, width = 2):
	n = len(G.nodes)
	if n == 0:
		return

	edges = None
	notvalid = True
	while notvalid:
		edges = rng.integers(n+1, size=(width, 2))
		edges = [tuple(edge) for edge in edges]
		edges = np.unique(edges, axis=0)

		edges = edges.tolist()

		if len(edges) == width:
			notvalid = False
		else:
			notvalid = True
		
		for u, v in edges:
			if (u, v) in G.edges:
				notvalid = True
				break
			elif u == v:
				notvalid = True
				break
			if u not in G.nodes and v not in G.nodes:
				notvalid = True
				break
			if u in G.nodes and v in G.nodes:
				if G.nodes[u]['forbidden'] and G.nodes[v]['forbidden']:
					notvalid = True
					break
	
	for u,v in edges:
		g = deepcopy(G)
		if u not in G.nodes:
			g.add_node(u)
			g.nodes[u]['forbidden'] = False
		elif v not in G.nodes:
			g.add_node(v)
			g.nodes[v]['forbidden'] = False
		if not g.nodes[u]['forbidden'] or not g.nodes[v]['forbidden']:
			g.add_edge(u,v)
			yield g, (u,v)

def builder_turn_possibilities(G : nx.Graph, m : int):
	if m <= 0:
		yield G
	if m >= 1:
		for g, _ in builder_possibilities(G):
			yield from builder_turn_possibilities(g, m-1)

def avoider_turn_possibilities(G : nx.Graph, n : int):
	if n <= 0:
		yield G
	if n >= 1:
		for g, _ in avoider_possibilities(G):
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

	possible_moves = list(builder_turn_possibilities(G, m))

	if maxlookahead <= 0:
		return [([possible_moves[0]], MAXLOOKAHEAD)] #possibly makes avoider more optimistic...
	
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
	best_futures = []
	
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
		#With this code, the avoider/forbidder wins when all vertices are forbidden.
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
	best_futures = []
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
	global animation_queue
	history = builder_turn(G, M, N, MAXLOOKAHEAD, ax=ax)
	animation_queue.put(None)

def main():		
	global history
	game_thread = threading.Thread(target = game_worker, daemon=True)
	game_thread.start()
	
	#anim = FuncAnimation(fig, animate, frames=game_frames, interval=100, cache_frame_data=False, repeat=False)
	#plt.show()

	

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