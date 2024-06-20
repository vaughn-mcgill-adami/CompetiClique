import torch
from torch.distributions.uniform import Uniform

from main import Deterministic

import networkx as nx
import matplotlib.pyplot as plt

from config import *
from main import run_trajectory, update_final_layer, update_embedding_size
from simple_decoder_transformer import SimpleDecoderTransformer
from competiclique_the_game import CompetiClique

from copy import deepcopy

from time import sleep

builder_states = torch.load("builder_policy_opt.pt")
forbidder_states = torch.load("forbidder_policy_opt.pt")

training_stats = builder_states['training_stats']
builder_discounted_return = [eval_stats['average_builder_return'] for batch_stats, eval_stats in training_stats]
forbidder_discounted_return = [eval_stats['average_forbidder_return'] for batch_stats, eval_stats in training_stats]

builder_wins = [eval_stats['builder_wins'] for batch_stats, eval_stats in training_stats]
forbidder_wins = [eval_stats['forbidder_wins'] for batch_stats, eval_stats in training_stats]
nobody_wins = [eval_stats['nobody_wins'] for batch_stats, eval_stats in training_stats]

total = [b+f+n for b, f, n in zip(builder_wins, forbidder_wins, nobody_wins)]

builder_wins = [b/t for b, t in zip(builder_wins, total)]
forbidder_wins = [f/t for f, t in zip(forbidder_wins, total)]
nobody_wins = [n/t for n, t in zip(nobody_wins, total)]

sum_both_returns = [x+y for x, y in zip(builder_discounted_return, forbidder_discounted_return)]

plt.plot(range(len(builder_discounted_return)), builder_discounted_return)
plt.plot(range(len(forbidder_discounted_return)), forbidder_discounted_return)
plt.plot(range(len(sum_both_returns)), sum_both_returns)
plt.plot(range(len(builder_wins)), builder_wins)
plt.plot(range(len(forbidder_wins)), forbidder_wins)
plt.plot(range(len(nobody_wins)), nobody_wins)

plt.show()
builder_policy = SimpleDecoderTransformer(L=LAYERS,
											H=HEADS,
											d_e=EMBEDDING_DIM,
											d_mlp=MLP_DIM,
											n_tokens=N_TOKENS,
											n_positions=POSITIONS,
											n_out=N_OUT)
builder_policy.load_state_dict(builder_states['builder_policy_state_dict'])

forbidder_policy = SimpleDecoderTransformer(L=LAYERS,
											H=HEADS,
											d_e=EMBEDDING_DIM,
											d_mlp=MLP_DIM,
											n_tokens=N_TOKENS,
											n_positions=POSITIONS,
											n_out=N_OUT)
forbidder_policy.load_state_dict(forbidder_states['forbidder_policy_state_dict'])

game = CompetiClique()
device = torch.device(DEVICE)

with torch.no_grad():
	epsilon = 2**-10
	action_noise = Deterministic(N_TOKENS, device)	
	builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, builder_observations, forbidder_observations, graphs_each_turn, winner = run_trajectory(game, builder_policy, forbidder_policy, action_noise, device, evalu=True)

	print(winner)
	print(turn_number)
	print()
	print(builder_observations)
	print(builder_rewards)
	print()
	print(forbidder_rewards)
	print(forbidder_observations)
	print()
	print()

	for turn, graph in enumerate(graphs_each_turn):
		color_map = list('red' if graph.nodes[node]['forbidden'] else 'blue' for node in graph)
		nx.draw(graph, node_color=color_map, with_labels=True)
		plt.show()
