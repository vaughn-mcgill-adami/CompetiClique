import torch
import networkx as nx

from copy import deepcopy
from collections import deque

import numpy as np

from config import *


def edge_list_to_action_tensor(edges):
	return torch.cat([torch.tensor([u,v]) for u,v in edges])

def vertex_list_to_action_tensor(vertices):
	return torch.tensor([u for u in vertices]) 

class CompetiClique():
	def __init__(self, changable = True, width = 2):
		self.rng = np.random.default_rng()
		
		self.too_many_vertices_penalty = -2
		self.outside_graph_penalty = -2
		self.uisv_penalty = -2
		self.forbidden_edge_penalty = -2
		self.existing_edge_penalty = -2
		self.not_vertex_penalty = -2
		self.already_forbidden_vertex_penalty = -2

		self.player_won = {"builder": False, "forbidder" : False}
		self.win_reward = 1
		self.lose_reward = -1

		self.ordered_edges_cache = deque()

		#Used for collecting brute force search trajectories.
		self.width = width
		self.MAXLOOKAHEAD = None
		
		
	def initialize_graph(self, start_vertex):
		self.G = nx.Graph()
		self.G.add_node(start_vertex if start_vertex is not None else int(self.rng.integers(low=0, high=VERTEX_VOCABULARY)))
		for u in self.G.nodes:
			self.G.nodes[u]['forbidden'] = False
		self.ordered_edges_cache = deque()

	def detect_builder_win(self):
		"""
		Theta somethin'
		"""
		return max(nx.node_clique_number(self.G).values()) >= self.K

	def detect_builder_win_of(self, G):
		"""
		Theta somethin'
		"""
		return max(nx.node_clique_number(G).values()) >= self.K
	
	def detect_forbidder_win(self):
		return sum(1 if self.G.nodes[u]['forbidden'] else 0 for u in self.G.nodes) == len(self.G.nodes)

	def observe(self):
		"""
		uses ordered_edges_cache to give consistent observations.

		Note: currently only supports initialization from one vertex.
		"""
		if len(self.ordered_edges_cache) != 0:
			observation = [torch.tensor([u + VERTEX_VOCAB_STARTS_AT, 
										FORBIDDEN_TOKEN if self.G.nodes[u]['forbidden'] else AVAILABLE_TOKEN,
										v + VERTEX_VOCAB_STARTS_AT,
										FORBIDDEN_TOKEN if self.G.nodes[v]['forbidden'] else AVAILABLE_TOKEN])for u, v in self.ordered_edges_cache]
			observation = [torch.tensor([VERTEX_VOCAB_STARTS_AT + self.K, VERTEX_VOCAB_STARTS_AT + self.M, VERTEX_VOCAB_STARTS_AT + self.N, END_GAME_DESCRIPTION_TOKEN])] + observation #NOTE: K, M, N <= N_TOKENS is required
			observation = observation + [torch.tensor([END_OBSERVATION_TOKEN])]
			observation = torch.cat(observation, dim=0)
		else:
			observation = [torch.tensor([VERTEX_VOCAB_STARTS_AT + self.K, VERTEX_VOCAB_STARTS_AT + self.M, VERTEX_VOCAB_STARTS_AT + self.N, END_GAME_DESCRIPTION_TOKEN])]
			observation = observation + [torch.tensor([VERTEX_VOCAB_STARTS_AT + u, FORBIDDEN_TOKEN if self.G.nodes[u]['forbidden'] else AVAILABLE_TOKEN]) for u in self.G.nodes]
			observation = observation + [torch.tensor([END_OBSERVATION_TOKEN])]
			observation = torch.cat(observation, dim=0)

		observation = torch.unsqueeze(observation, dim=0)

		return observation
		
	def step(self, action, player):
		"""
		actions are 1d tensors of 2*M integers when player="builder"
		actions are 1d tensors of N integers when player="forbidder"
		"""
		if player == "builder":

			assert len(action) == 2*self.M
			
			if self.player_won['forbidder']:
				builder_reward = self.lose_reward
				return None, builder_reward, False

			for idx in range(0,len(action),2):
				u = action[idx].item()
				v = action[idx+1].item()

				if u < VERTEX_VOCAB_STARTS_AT or v < VERTEX_VOCAB_STARTS_AT:
					builder_reward = self.not_vertex_penalty
					return None, builder_reward, False
				u -= VERTEX_VOCAB_STARTS_AT
				v -= VERTEX_VOCAB_STARTS_AT
				if u == v:
					builder_reward = self.uisv_penalty
					return None, builder_reward, False
				if (u,v) in self.G.edges or (v,u) in self.G.edges:
					builder_reward = self.existing_edge_penalty
					return None, builder_reward, False
				if u not in self.G.nodes and v not in self.G.nodes:
					builder_reward = self.too_many_vertices_penalty
					return None, builder_reward, False
				if u not in self.G.nodes:
					self.G.add_node(u)
					self.G.nodes[u]['forbidden'] = False
				elif v not in self.G.nodes:
					self.G.add_node(v)
					self.G.nodes[v]['forbidden'] = False
				if not self.G.nodes[u]['forbidden'] or not self.G.nodes[v]['forbidden']:
					self.G.add_edge(u,v)
					self.ordered_edges_cache.append((u,v))
				else:
					builder_reward = self.forbidden_edge_penalty
					return None, builder_reward, False

				if self.detect_builder_win():
					builder_reward = self.win_reward
					self.player_won['builder'] = True
					return self.observe(), builder_reward, False
			
			builder_reward = 0
			return self.observe(), builder_reward, True

		elif player == "forbidder":
			assert len(action) == self.N

			if self.player_won['builder']:
				forbidder_reward = self.lose_reward
				return None, forbidder_reward, False

			for u in action:
				u = u.item()
				if u < VERTEX_VOCAB_STARTS_AT:
					forbidder_reward = self.not_vertex_penalty
					return None, forbidder_reward, False
				u -= VERTEX_VOCAB_STARTS_AT
				if u not in self.G.nodes:
					forbidder_reward = self.outside_graph_penalty
					return None, forbidder_reward, False
				if self.G.nodes[u]['forbidden']:
					forbidder_reward = self.already_forbidden_vertex_penalty
					return None, forbidder_reward, False
				else:
					self.G.nodes[u]['forbidden'] = True
			
				if self.detect_forbidder_win():
					forbidder_reward = self.win_reward
					self.player_won['forbidder'] = True
					return self.observe(), forbidder_reward, False

			forbidder_reward = 0	
			return self.observe(), forbidder_reward, True

	def reset(self, clique_size=None, edges_per_builder_turn=None, vertices_per_forbidder_turn=None, start_vertex=None):
		self.initialize_graph(start_vertex = start_vertex)
		if clique_size is None:
			self.K = int(self.rng.integers(low=MIN_CLIQUE_SIZE, high=MAX_CLIQUE_SIZE + 1))
			self.M = int(self.rng.integers(low=MIN_EDGES_PER_BUILDER_TURN, high=MAX_EDGES_PER_BUILDER_TURN + 1))
			self.N = int(self.rng.integers(low=MIN_VERTICES_PER_FORBIDDER_TURN, high=MAX_VERTICES_PER_FORBIDDER_TURN + 1))
		else:
			self.K = clique_size
			self.M = edges_per_builder_turn
			self.N = vertices_per_forbidder_turn

		self.player_won['builder'] = False
		self.player_won['forbidder'] = False
		
		obs = self.observe()
		return obs
	
	def set_brute_search_width(self, width):
		self.width = width
	
	def avoider_possibilities(self, G):
		#vertices = rng.integers(low = 0, high=len(G.nodes), size=search_width)
		n = len(G.nodes)
		width = self.width
		
		if n == 0:
			return

		shuffled_vertices = list(G.nodes)
		self.rng.shuffle(shuffled_vertices)
		
		for u in shuffled_vertices[:width]:
			if not G.nodes[u]['forbidden']:
				g = deepcopy(G)
				g.nodes[u]['forbidden'] = True
				yield g, u


	def builder_possibilities(self, G):
		missing_vertices = [u for u in range(VERTEX_VOCABULARY) if u not in G.nodes]
		missing_vertex = None
		working_G = deepcopy(G)
		
		if len(missing_vertices) != 0:
			missing_vertex = int(self.rng.choice(missing_vertices))
			working_G.add_node(missing_vertex)
			working_G.nodes[missing_vertex]['forbidden'] = False
		
		edges = list(nx.complement(working_G).edges)
		if len(edges) == 0:
			return

		edges = self.rng.choice(edges, size=self.width).tolist()

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

	def builder_turn_possibilities(self, G, m):
		if m <= 0:
			yield G, []
		if m >= 1:
			for g, action in self.builder_possibilities(G):
				for afterg, afteraction in self.builder_turn_possibilities(g, m-1):
					yield afterg, [action] + afteraction

	def avoider_turn_possibilities(self, G, n):
		if n <= 0:
			yield G, []
		if n >= 1:
			for g, action in self.avoider_possibilities(G):
				nextround = list(self.avoider_turn_possibilities(g, n-1))
				if len(nextround) == 0:
					yield g
				else:
					for afterg, afteraction in nextround:
						yield afterg, [action]+afteraction

	def get_brute_trajectories(self, width, maxlookahead):
		self.MAXLOOKAHEAD = maxlookahead
		self.width = width
		return self.builder_brute_turn(self.G, maxlookahead)

	
	def builder_brute_turn(self, G :  nx.Graph, maxlookahead : int):
		m = self.M
		n = self.N

		turnnum = self.MAXLOOKAHEAD - maxlookahead

		possible_moves = list(self.builder_turn_possibilities(G, self.M))

		if len(possible_moves) == 0:
			return [([], self.MAXLOOKAHEAD)]

		if maxlookahead <= 0:
			return [([(possible_moves[0][0], possible_moves[0][1], 0)], self.MAXLOOKAHEAD)] #possibly makes avoider more optimistic...
		
		assert len(possible_moves[0]) == 2

		for graph, action in possible_moves:
			if self.detect_builder_win_of(graph):
				return [([(graph, action, 1)], 0)] #note: graph is the graph resulting from action i.e. after action is performed
		
		futures = []
		for graph, action in possible_moves:
			afterg = self.avoider_brute_turn(graph, maxlookahead-1)
			#print(afterg)
			for future, cost in afterg:
				futures.append(([(graph, action, 0)]+future, cost))

		future, best_cost = futures[0]
		best_futures = []
		
		for future, cost in futures:
			if cost < best_cost:
				best_futures = []
				best_futures.append((future, cost + 1))
				best_cost = cost
			elif cost == best_cost:
				best_futures.append((future, cost + 1))
		
		return best_futures

	def avoider_brute_turn(self, G : nx.Graph, maxlookahead : int):
		turnnum = self.MAXLOOKAHEAD - maxlookahead

		possible_moves = list(self.avoider_turn_possibilities(G, self.N))
		
		if len(possible_moves) == 0:
			#With this code, the avoider/forbidder wins when all vertices are forbidden.
			return [([], self.MAXLOOKAHEAD)]

		if maxlookahead <= 0:
			return [([(possible_moves[0][0], possible_moves[0][1], 0)], self.MAXLOOKAHEAD)] #possibly makes avoider more optimistic... (for dyn prog, no affect on RL)
		
		futures = []

		assert len(possible_moves[0]) == 2

		for graph, action in possible_moves:
			afterg = self.builder_brute_turn(graph, maxlookahead-1)
			for future, cost in afterg:
				futures.append(([(graph, action, 0)]+future, cost))
		
		future, best_cost = futures[0]
		best_futures = []
		for future, cost in futures:
			if cost > best_cost:
				best_futures = []
				best_futures.append((future, cost + 1))
				best_cost = cost
			elif cost == best_cost:
				best_futures.append((future, cost + 1))
		
		return best_futures
