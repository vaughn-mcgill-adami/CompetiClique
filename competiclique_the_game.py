import torch
import networkx as nx

from copy import deepcopy
from collections import deque

from numpy.random import default_rng

from config import *
from math import e

class CompetiClique():
	def __init__(self, clique_size, edges_per_builder_turn, vertices_per_forbidder_turn):
		self.K = clique_size
		self.M = edges_per_builder_turn
		self.N = vertices_per_forbidder_turn

		self.rng = default_rng()

		self.G = nx.Graph()
		self.G.add_node(self.rng.integers(VERTEX_VOCABULARY, size=1)[0])

		self.too_many_vertices_penalty = -1
		self.outside_graph_penalty = -1
		self.uisv_penalty = -1

		self.ordered_edges_cache = deque()
	
	def initialize_graph(self):
		self.G = nx.Graph()
		self.G.add_node(self.rng.integers(VERTEX_VOCABULARY, size=1)[0])
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
			observation.append(torch.tensor([END_OBSERVATION_TOKEN]))
			observation = torch.cat(observation)
		else:
			observation = torch.cat([torch.tensor([u, FORBIDDEN_TOKEN if self.G.nodes[u]['forbidden'] else AVAILABLE_TOKEN]) for u in self.G.nodes] + [torch.tensor([END_OBSERVATION_TOKEN])], dim=0)

		observation = torch.unsqueeze(observation, dim=0)

		return observation
	
	def reward_win_lose(self, player):
		if self.detect_builder_win():
			if player == "builder":
				return 1
			if player == "forbidder":
				return -1
		elif self.detect_forbidder_win():
			if player == "builder":
				return -1
			if player == "forbidder":
				return 1
		else:
			return 0
		
	def step(self, action, player):
		"""
		actions are 1d tensors of 2*M integers when player="builder"
		actions are 1d tensors of N integers when player="forbidder"
		"""
		if player == "builder":
			
			builder_reward = 0
			if len(action) != 2*self.M:
				reward += -abs(len(action) - 2*self.M)
				return None, builder_reward
			for idx in range(0,len(action),2):
				u = action[idx].item()
				v = action[idx+1].item()
				vertices_added = 0
				if u == v:
					builder_reward += self.uisv_penalty
					return None, builder_reward
				if u not in self.G.nodes:
					vertices_added += 1
					self.G.add_node(u)
					self.G.nodes[u]['forbidden'] = False
				if v not in self.G.nodes:
					vertices_added += 1
					if(vertices_added >= 2):
						builder_reward += self.too_many_vertices_penalty
						return None, builder_reward
					self.G.add_node(v)
					self.G.nodes[v]['forbidden'] = False
				if not self.G.nodes[u]['forbidden'] or not self.G.nodes[v]['forbidden']:
					self.G.add_edge(u,v)
					self.ordered_edges_cache.append((u,v))
				else:
					builder_reward += self.forbidden_edge_penalty
					return None, builder_reward

			builder_reward += self.reward_win_lose(player)
			return self.observe(), builder_reward

		elif player == "forbidder":
			forbidder_reward = 0
			if len(action) != self.N:
				forbidder_reward += -abs(len(action) - self.N)
				return None, forbidder_reward
			for u in action:
				if u not in self.G.nodes:
					forbidder_reward += self.outside_graph_penalty
					return None, forbidder_reward
				else:
					self.G.nodes[u]['forbidden'] = True
			
			forbidder_reward += self.reward_win_lose(player)
			return self.observe(), forbidder_reward

	def reset(self):
		self.initialize_graph()
		obs = self.observe()
		return obs