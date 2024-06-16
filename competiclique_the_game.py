import torch
import networkx as nx

from copy import deepcopy
from collections import deque

from numpy.random import default_rng

from config import *

"""
TODO: Possibly give the forbidder a reward when the builder connects all vertices in the vocabulary, 
			though this event will never happen.
"""

class CompetiClique():
	def __init__(self):
		self.rng = default_rng()
		
		self.too_many_vertices_penalty = -1
		self.outside_graph_penalty = -1
		self.uisv_penalty = -1
		self.forbidden_edge_penalty = -1
		self.existing_edge_penalty = -1
		self.not_vertex_penalty = -1
		self.already_forbidden_vertex_penalty = -1

		self.player_won = {"builder": False, "forbidder" : False}
		self.win_reward = 1
		self.lose_reward = -1

		self.ordered_edges_cache = deque()
		
		
	def initialize_graph(self):
		self.G = nx.Graph()
		self.G.add_node(self.rng.integers(low=VERTEX_VOCAB_STARTS_AT, high=N_TOKENS, size=1)[0])
		for u in self.G.nodes:
			self.G.nodes[u]['forbidden'] = False
		self.ordered_edges_cache = deque()
	
	def detect_builder_win(self):
		"""
		Theta somethin'
		"""
		return max(nx.node_clique_number(self.G).values()) >= self.K
	
	def detect_forbidder_win(self):
		return sum(1 if self.G.nodes[u]['forbidden'] else 0 for u in self.G.nodes) == len(self.G.nodes)

	def observe(self):
		"""
		uses ordered_edges_cache to give consistent observations.

		Note: currently only supports initialization from one vertex.
		"""
		if len(self.ordered_edges_cache) != 0:
			observation = [torch.tensor([u, 
										FORBIDDEN_TOKEN if self.G.nodes[u]['forbidden'] else AVAILABLE_TOKEN,
										v,
										FORBIDDEN_TOKEN if self.G.nodes[v]['forbidden'] else AVAILABLE_TOKEN])for u, v in self.ordered_edges_cache]
			observation = [torch.tensor([VERTEX_VOCAB_STARTS_AT + self.K, VERTEX_VOCAB_STARTS_AT + self.M, VERTEX_VOCAB_STARTS_AT + self.N, END_GAME_DESCRIPTION_TOKEN])] + observation #NOTE: K, M, N <= N_TOKENS is required
			observation = observation + [torch.tensor([END_OBSERVATION_TOKEN])]
			observation = torch.cat(observation, dim=0)
		else:
			observation = torch.cat([torch.tensor([VERTEX_VOCAB_STARTS_AT + self.K, VERTEX_VOCAB_STARTS_AT + self.M, VERTEX_VOCAB_STARTS_AT + self.N, END_GAME_DESCRIPTION_TOKEN])] + [torch.tensor([u, FORBIDDEN_TOKEN if self.G.nodes[u]['forbidden'] else AVAILABLE_TOKEN]) for u in self.G.nodes] + [torch.tensor([END_OBSERVATION_TOKEN])], dim=0)

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
				vertices_added = 0
				if u < VERTEX_VOCAB_STARTS_AT or v < VERTEX_VOCAB_STARTS_AT:
					builder_reward = self.not_vertex_penalty
					return None, builder_reward, False
				if u == v:
					builder_reward = self.uisv_penalty
					return None, builder_reward, False
				if (u,v) in self.G.edges or (v,u) in self.G.edges:
					builder_reward = self.existing_edge_penalty
					return None, builder_reward, False
				if u not in self.G.nodes:
					vertices_added = 1
					self.G.add_node(u)
					self.G.nodes[u]['forbidden'] = False
				if v not in self.G.nodes:
					vertices_added = 1
					if(vertices_added >= 2):
						builder_reward = self.too_many_vertices_penalty
						return None, builder_reward, False
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

	def reset(self, clique_size, edges_per_builder_turn, vertices_per_forbidder_turn):
		self.initialize_graph()
		self.K = clique_size
		self.M = edges_per_builder_turn
		self.N = vertices_per_forbidder_turn

		self.player_won['builder'] = False
		self.player_won['forbidder'] = False
		
		obs = self.observe()
		return obs