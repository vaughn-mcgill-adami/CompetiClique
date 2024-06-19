from competiclique_the_game import CompetiClique
from simple_decoder_transformer import SimpleDecoderTransformer#, set_batch_norm_momentum
from config import *

from copy import deepcopy
from collections import deque
import time
import argparse

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from torch import nn

from rich.progress import track
from rich.progress import Progress
#import networkx as nx
#from matplotlib import pyplot as plt

class Deterministic():
	"""
	A dummy class that is used in place of a distribution over non-negative "noise vectors" of shape size
	"""
	def __init__(self, size, device):
		self.size = size
		self.zeros = torch.zeros(self.size).to(device)
		return
	def sample(self):
		return self.zeros

def update_embedding_size(embedding, vocab_size):
	with torch.no_grad():
		embedding.weight = nn.Parameter(torch.cat((embedding.weight, torch.randn(vocab_size - embedding.weight.shape[0], EMBEDDING_DIM) ), dim=0))

def update_final_layer(layer, vocab_size):
	with torch.no_grad():
		layer.weight = nn.Parameter(torch.cat((layer.weight, torch.randn(vocab_size - layer.weight.shape[0], EMBEDDING_DIM) ), dim=0))
		layer.bias = nn.Parameter(torch.cat((layer.bias, torch.randn(vocab_size - layer.bias.shape[0]) ), dim=0))

def load_training_history(device):
	builder_state = torch.load(BUILDERPOLICYOPTPATH, map_location=device)
	forbidder_state = torch.load(FORBIDDERPOLICYOPTPATH, map_location=device)

	training_stats = builder_state['training_stats']
	
	print('builder_policy_state_dict : ')
	for name, val in builder_state['builder_policy_state_dict'].items():
		print(name, val.shape)
	print()

	print('forbidder_policy_state_dict : ')
	for name, val in forbidder_state['forbidder_policy_state_dict'].items():
		print(name, val.shape)
	print()

	builder_old_n_tokens = builder_state['builder_policy_state_dict']['vertex_embedding.weight'].shape[0]
	builder_old_n_positions = builder_state['builder_policy_state_dict']['position_embedding.weight'].shape[0]
	builder_old_n_out = builder_state['builder_policy_state_dict']['final_linear.bias'].shape[0]

	builder_policy = SimpleDecoderTransformer(L = LAYERS, 
											 H=HEADS, 
											 d_e=EMBEDDING_DIM,
											 d_mlp=MLP_DIM,
											 n_tokens=builder_old_n_tokens,
											 n_positions=builder_old_n_positions,
											 n_out=builder_old_n_out)

	forbidder_old_n_tokens = forbidder_state['forbidder_policy_state_dict']['vertex_embedding.weight'].shape[0]
	forbidder_old_n_positions = forbidder_state['forbidder_policy_state_dict']['position_embedding.weight'].shape[0]
	forbidder_old_n_out = forbidder_state['forbidder_policy_state_dict']['final_linear.bias'].shape[0]

	forbidder_policy = SimpleDecoderTransformer(L = LAYERS, 
											 H=HEADS, 
											 d_e=EMBEDDING_DIM,
											 d_mlp=MLP_DIM,
											 n_tokens=forbidder_old_n_tokens,
											 n_positions=forbidder_old_n_positions,
											 n_out=forbidder_old_n_out)

	builder_policy.load_state_dict(builder_state['builder_policy_state_dict'])
	print('loaded builder policy')
	forbidder_policy.load_state_dict(forbidder_state['forbidder_policy_state_dict'])
	print('loaded forbidder policy')

	assert len(training_stats) != 0
	print(len(training_stats))
	
	best_so_far = {"builder" : max(batch_stats['average_builder_return'] for batch_stats, eval_stats in training_stats),
								"forbidder" : max(batch_stats['average_forbidder_return'] for batch_stats, eval_stats in training_stats)
								}
	print('best builder average return :', best_so_far['builder'])
	print('best forbidder average return :', best_so_far['forbidder'])
	print('second latest builder average return :', training_stats[len(training_stats) - 1][0]['average_builder_return'])
	print('second latest forbidder average return :', training_stats[len(training_stats) - 1][0]['average_forbidder_return'])
	
	update_embedding_size(builder_policy.vertex_embedding, N_TOKENS)
	update_embedding_size(builder_policy.position_embedding, POSITIONS)
	update_embedding_size(forbidder_policy.vertex_embedding, N_TOKENS)
	update_embedding_size(forbidder_policy.position_embedding, POSITIONS)

	update_final_layer(builder_policy.final_linear, N_TOKENS)
	update_final_layer(forbidder_policy.final_linear, N_TOKENS)
	
	print('builder_policy_state_dict (2): ')
	for name, val in builder_policy.named_parameters():
		print(name, val.shape)
	print()

	print('forbidder_policy_state_dict (2): ')
	for name, val in forbidder_policy.named_parameters():
		print(name, val.shape)
	print()
	
	print('building optimizers')
	builder_optimizer = torch.optim.Adam(builder_policy.parameters(), lr=LEARNING_RATE)
	forbidder_optimizer = torch.optim.Adam(forbidder_policy.parameters(), lr=LEARNING_RATE)
	if builder_old_n_tokens == N_TOKENS and builder_old_n_positions == POSITIONS and builder_old_n_out == N_OUT:
		builder_optimizer.load_state_dict(builder_state['builder_optimizer_state_dict'])
	if forbidder_old_n_tokens == N_TOKENS and forbidder_old_n_positions == POSITIONS and forbidder_old_n_out == N_OUT:
		forbidder_optimizer.load_state_dict(forbidder_state['forbidder_optimizer_state_dict'])
	print('loaded optimizers')

	return training_stats, best_so_far, builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer


def collect_batch_of_trajectories(game, batch_size, batch : int, builder_policy, forbidder_policy, action_noise, device, evalu = False):
	#batch_builder_observations = deque()
	batch_builder_actions = deque()
	batch_builder_returns = deque()

	#batch_forbidder_observations = deque()
	batch_forbidder_actions = deque()
	batch_forbidder_returns = deque()

	batch_stats = {'average_game_length' : deque(),
					 'max_game_length' : 0,
					 'builder_wins' : 0,
					 'forbidder_wins' : 0,
					 'nobody_wins' : 0}

	for episode in track(range(batch_size), description = f'Batch: {batch}/{NUM_BATCHES} : playing {batch_size} games : '):
		builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner = run_trajectory(game, builder_policy, forbidder_policy, action_noise, device, evalu = evalu)
		#print(f"previous observation on turn {turn_number}:", prevobs)
		#print(f"previous action on turn {turn_number}", builder_actions_chosen[-1] if turn_number%2==1 else forbidder_actions_chosen[-1] if len(forbidder_actions_chosen)!=0 else "empty")
		batch_stats[winner] += 1
		batch_stats['average_game_length'].append(turn_number)
		batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_number)

		if len(builder_rewards) != 0:
			builder_rewards = torch.tensor(list(builder_rewards))

			#print("builder_rewards:", builder_rewards)

			builder_discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(builder_rewards[k:])) for k in range(len(builder_rewards))])
			builder_discounted_returns = builder_discounted_returns.repeat_interleave(2*game.M)
			#print("M = ", game.M)

			builder_actions_probs = torch.cat(list(builder_actions_probs))
			builder_actions_chosen = torch.cat(list(builder_actions_chosen))

			builder_actions = builder_actions_probs[torch.arange(len(builder_actions_probs)), builder_actions_chosen]

			batch_builder_actions.append(builder_actions)
			batch_builder_returns.append(builder_discounted_returns)


		if len(forbidder_rewards) != 0:
			forbidder_rewards = torch.tensor(list(forbidder_rewards))
			
			forbidder_discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(forbidder_rewards[k:])) for k in range(len(forbidder_rewards))])
			forbidder_discounted_returns = forbidder_discounted_returns.repeat_interleave(game.N) #TODO: Possibly move before the discounting? Might keep it the way it is b/c good heuristic for larger M, N
			#print("N = ", game.N)

			forbidder_actions_probs = torch.cat(list(forbidder_actions_probs))
			forbidder_actions_chosen = torch.cat(list(forbidder_actions_chosen))

			forbidder_actions = forbidder_actions_probs[torch.arange(len(forbidder_actions_probs)), forbidder_actions_chosen]

			batch_forbidder_actions.append(forbidder_actions)
			batch_forbidder_returns.append(forbidder_discounted_returns)

	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])
	
	return batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats


def collect_batch_of_brute_trajectories(game, batch_size, width, maxlookahead):
	builder_observations = deque() #not organized by trajectory; flattened "trajectory" dimension
	builder_actions = deque()
	forbidder_observations = deque()
	forbidder_actions = deque()
	batch_stats = dict()

	batch_stats = {'average_game_length' : deque(),
					 'max_game_length' : 0,
					 'builder_wins' : 0,
					 'forbidder_wins' : 0,
					 'nobody_wins' : 0}
	
	with Progress() as progress:
		batch_progress = progress.add_task("Generating batch: ", total = batch_size)
		episode_num = 0
		
		while episode_num < batch_size:
			game.reset() #sets K,N,M to random values.

			clique_size = game.K
			edges_per_builder_turn = game.M
			vertices_per_forbidder_turn = game.N 
			start_vertex = list(game.G.nodes)[0]

			#print(episode_num, clique_size, edges_per_builder_turn, vertices_per_forbidder_turn, start_vertex)

			trajectories = game.get_brute_trajectories(width, maxlookahead)
			for trajectory_num, (trajectory, cost) in enumerate(trajectories):
				#print("trajectory: ", trajectory_num)
				obs = game.reset(clique_size = clique_size, 
								 edges_per_builder_turn = edges_per_builder_turn, 
								 vertices_per_forbidder_turn = vertices_per_forbidder_turn, 
								 start_vertex = start_vertex)
				
				for turn_num, (graph, action, reward) in enumerate(trajectory):
					#color_map = deque('red' if graph.nodes[node]['forbidden'] else 'blue' for node in graph)
					#nx.draw(graph, node_color=color_map, with_labels=True)
					#plt.show()
					if turn_num%2 == 0:
						obs = torch.squeeze(obs)
						builder_observations.append(obs)
						action = torch.tensor(action)
						action = torch.flatten(action) + torch.tensor(VERTEX_VOCAB_STARTS_AT)
						prevobs = obs
						obs, rew, cont = game.step(action, player='builder')
						#print(prevobs, action, graph.nodes, game.G.nodes)

						builder_actions.append(action)
					else:
						obs = torch.squeeze(obs)
						forbidder_observations.append(obs)
						action = torch.tensor(action)
						action = torch.flatten(action) + torch.tensor(VERTEX_VOCAB_STARTS_AT)
						prevobs = obs
						obs, rew, cont = game.step(action, player='forbidder')
						#print(prevobs, action, graph.nodes, game.G.nodes)

						forbidder_actions.append(action)
						
				batch_stats['nobody_wins' if rew == 0 else 'builder_wins' if turn_num%2 == 0 else 'forbidder_wins'] += 1
				batch_stats['average_game_length'].append(turn_num)
				batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_num)

			advance = len(trajectories)
			episode_num += advance
			progress.update(batch_progress, advance=advance)

	assert len(builder_actions) == len(builder_observations)
	assert len(forbidder_actions) == len(forbidder_observations)
	
	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])
	
	return builder_observations, builder_actions, forbidder_observations, forbidder_actions, batch_stats

def run_trajectory(game, builder_policy, forbidder_policy, action_noise, device, evalu=False):
	builder_observations = deque()
	builder_actions_probs = deque()
	builder_actions_chosen = deque()
	builder_rewards = deque()
	forbidder_observations = deque()
	forbidder_actions_probs = deque()
	forbidder_actions_chosen = deque()
	forbidder_rewards = deque()

	observation = game.reset()
	turn_number = 0

	graphs_each_turn = deque()

	builder_continue = True
	forbidder_continue = True
	
	winner = ""

	while observation is not None and (builder_continue or forbidder_continue):
		#prevobs = observation
		if turn_number % 2 == 0:
			#builder_observations.append(observation)

			formatted_action_probs = []
			formatted_action_chosen = []
			for k in range(2*game.M):
				observation = observation.to(device)
				action_probs = builder_policy(observation)[0][-1]

				action_probs = action_probs + action_noise.sample()

				action_chosen = Categorical(probs = action_probs).sample()
				formatted_action_probs.append(action_probs)
				formatted_action_chosen.append(action_chosen)
				observation = torch.cat((observation,torch.unsqueeze(torch.unsqueeze(action_chosen,dim=0),dim=0)), dim=1)
			builder_observations.append(observation)
			formatted_action_probs = torch.stack(formatted_action_probs, dim=0)
			formatted_action_chosen = torch.stack(formatted_action_chosen, dim=0)

			#print("before\n", observation)

			observation, reward, builder_continue = game.step(formatted_action_chosen, 'builder')

			#print("after\n", observation)

			builder_actions_probs.append(formatted_action_probs)
			builder_actions_chosen.append(formatted_action_chosen)
			builder_rewards.append(reward)
		else:
			#forbidder_observations.append(observation)

			formatted_action_probs = []
			formatted_action_chosen = []
			for k in range(game.N):
				observation = observation.to(device)
				action_probs = forbidder_policy(observation)[0][-1]

				action_probs = action_probs + action_noise.sample()

				action_chosen = Categorical(probs = action_probs).sample()
				formatted_action_probs.append(action_probs)
				formatted_action_chosen.append(action_chosen)
				observation = torch.cat((observation,torch.unsqueeze(torch.unsqueeze(action_chosen,dim=0),dim=0)), dim=1)
			forbidder_observations.append(observation)
			formatted_action_probs = torch.stack(formatted_action_probs, dim=0)
			formatted_action_chosen = torch.stack(formatted_action_chosen, dim=0)

			observation, reward, forbidder_continue = game.step(formatted_action_chosen, 'forbidder')

			forbidder_actions_probs.append(formatted_action_probs)
			forbidder_actions_chosen.append(formatted_action_chosen)
			forbidder_rewards.append(reward)
		if evalu: 
			graphs_each_turn.append(deepcopy(game.G))
		turn_number += 1

		if observation is not None and builder_continue and not forbidder_continue:
			winner = "forbidder_wins"
		elif observation is not None and not builder_continue and forbidder_continue:
			winner = "builder_wins"
		elif observation is None and winner == "":
			winner = "nobody_wins"

	if evalu:
		return builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, builder_observations, forbidder_observations, graphs_each_turn, winner
	else:
		return builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner


def update_policies(builder_optimizer, forbidder_optimizer, batch_builder_actions, batch_forbidder_actions, batch_builder_returns, batch_forbidder_returns, device, batch_stats):
	if len(batch_builder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_builder_actions)

		batch_builder_actions = list(F.pad(actions, (0,longest_trajectory_len - len(actions)), value = 1.0) for actions in batch_builder_actions)
		batch_builder_actions = torch.stack(batch_builder_actions, dim = 0)
	
		batch_builder_returns = (list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value = 0.0) for returns in batch_builder_returns))
		batch_builder_returns = torch.stack(batch_builder_returns, dim = 0).to(device)
		
		batch_stats['average_builder_return'] = torch.mean(batch_builder_returns[:,0]).cpu().item()
		print('average_builder_return', batch_stats['average_builder_return'])

		if not EVAL_ONLY:
			intermediate = (torch.log(batch_builder_actions)*batch_builder_returns).sum(dim=-1)

			builder_loss = -torch.mean(intermediate)

			builder_loss.backward()
			builder_optimizer.step()
			builder_optimizer.zero_grad()

			batch_stats['builder_loss'] = builder_loss.cpu().item()
			print('builder_loss', batch_stats['builder_loss'])
	if len(batch_forbidder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_forbidder_actions)

		batch_forbidder_actions = list(F.pad(actions, (0, longest_trajectory_len - len(actions)), value= 1.0) for actions in batch_forbidder_actions)
		batch_forbidder_actions = torch.stack(batch_forbidder_actions, dim = 0)
		
		batch_forbidder_returns = list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value= 0.0) for returns in batch_forbidder_returns)
		batch_forbidder_returns = torch.stack(batch_forbidder_returns, dim = 0).to(device)

		batch_stats['average_forbidder_return'] = torch.mean(batch_forbidder_returns[:,0]).cpu().item()
		print('average_forbidder_return', batch_stats['average_forbidder_return'])

		if not EVAL_ONLY:
			intermediate = (torch.log(batch_forbidder_actions)*batch_forbidder_returns).sum(dim=-1)

			forbidder_loss = -torch.mean(intermediate)

			forbidder_loss.backward()
			forbidder_optimizer.step()
			forbidder_optimizer.zero_grad()

			batch_stats['forbidder_loss'] = forbidder_loss.cpu().item()
			print('forbidder_loss', batch_stats['forbidder_loss'])

def update_pretraining_policies(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, batch_builder_observations, batch_builder_actions, batch_forbidder_observations, batch_forbidder_actions, device, batch_stats):
	
	builder_batch = [torch.cat((observation, action)) for observation, action in zip(batch_builder_observations, batch_builder_actions)]
	forbidder_batch = [torch.cat((observation, action)) for observation, action in zip(batch_forbidder_observations, batch_forbidder_actions)]

	def pad_batch(batch):
		longest_trajectory_len = max(len(trajectory) for trajectory in batch)
		batch = list(F.pad(trajectory, (0, longest_trajectory_len - len(trajectory)), value=PAD_TOKEN) for trajectory in batch)
		batch = torch.stack(batch, dim = 0)
		return batch

	builder_batch = pad_batch(builder_batch).to(device)
	forbidder_batch = pad_batch(forbidder_batch).to(device)

	if len(builder_batch) != 0:
		BX = builder_batch[:-1]
		BY = builder_batch[1:]

		indices = torch.arange(N_TOKENS).to(device)[None, None, :] == BY[:, :, None]
		builder_loss = -torch.mean(torch.log(builder_policy(BX)[indices]))

		builder_loss.backward()
		builder_optimizer.step()
		builder_optimizer.zero_grad()
	if len(forbidder_batch) != 0:
		FX = forbidder_batch[:-1]
		FY = forbidder_batch[1:]

		indices = torch.arange(N_TOKENS).to(device)[None, None, :] == FY[:, :, None]
		forbidder_loss = -torch.mean(torch.log(forbidder_policy(FX)[indices]))

		forbidder_loss.backward()
		forbidder_optimizer.step()
		forbidder_optimizer.zero_grad()

def evaluate(game, batch, builder_policy, forbidder_policy, device):
	batch_stats = {}

	with torch.no_grad():
		builder_policy.eval()
		forbidder_policy.eval()
		action_noise = Deterministic(N_TOKENS, device)
		batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats = collect_batch_of_trajectories(game, NUM_EVAL_SAMPLES, batch, builder_policy, forbidder_policy, action_noise, device)
	builder_policy.train()
	forbidder_policy.train()

	if len(batch_builder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_builder_actions)
		batch_builder_returns = (list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value = 0.0) for returns in batch_builder_returns))
		batch_builder_returns = torch.stack(batch_builder_returns, dim = 0).to(device)

	if len(batch_forbidder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_forbidder_actions)
		batch_forbidder_returns = (list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value = 0.0) for returns in batch_forbidder_returns))
		batch_forbidder_returns = torch.stack(batch_forbidder_returns, dim = 0).to(device)

	batch_stats['average_builder_return'] = torch.mean(batch_builder_returns[:,0])
	batch_stats['average_forbidder_return'] = torch.mean(batch_forbidder_returns[:,0])

	return batch_stats
	

def checkpoint(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, training_stats, batch_stats, best_so_far):
	torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BUILDERPOLICYOPTPATH
		)
	torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, FORBIDDERPOLICYOPTPATH
		)
	if best_so_far['builder'] < batch_stats['average_builder_return']:
		torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BESTBUILDERPOLICYOPTPATH
		)
		best_so_far['builder'] = batch_stats['average_builder_return']
	if best_so_far['forbidder'] < batch_stats['average_forbidder_return']:
		torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, BESTFORBIDDERPOLICYOPTPATH
		)
		best_so_far['forbidder'] = batch_stats['average_forbidder_return']

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-pt', '--pretrain', action = 'store_true')
	args = parser.parse_args()

	device = torch.device(DEVICE)
	
	training_stats = []

	best_so_far = {"builder" : float('-inf'), 
									"forbidder" : float('-inf')
									}
	
	game = CompetiClique()

	if LOAD_SAVED_WEIGHTS:
		training_stats, best_so_far, builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer  = load_training_history(device)
	else:
		model = SimpleDecoderTransformer(L=LAYERS, 
									H=HEADS, 
									d_e=EMBEDDING_DIM, 
									d_mlp = MLP_DIM, 
									n_tokens = N_TOKENS, 
									n_positions = POSITIONS,
									n_out = N_OUT).to(device)
		builder_policy = model#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)
		forbidder_policy = deepcopy(model)#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)

		builder_optimizer = torch.optim.Adam(builder_policy.parameters(), lr=LEARNING_RATE)
		forbidder_optimizer = torch.optim.Adam(forbidder_policy.parameters(), lr=LEARNING_RATE)

		print('WARNING: training will overwrite existing weights because LOAD_SAVED_WEIGHTS = False in config.py')
	
	print(builder_policy.vertex_embedding.weight.shape)
	print(forbidder_policy.vertex_embedding.weight.shape)

	builder_policy.to(device)
	forbidder_policy.to(device)

	for batch in range(NUM_BATCHES):
		builder_policy.train()
		forbidder_policy.train()
		
		if args.pretrain:
			batch_builder_observations, batch_builder_actions, batch_forbidder_observations, batch_forbidder_actions, batch_stats = collect_batch_of_brute_trajectories(game, BATCH_SIZE, WIDTH, MAXLOOKAHEAD)

			print(f"Batch {batch} Statistics:")
			for key, value in batch_stats.items():
				print(key, value)

			start_backprop = time.time()
			update_pretraining_policies(builder_policy,
										forbidder_policy,
										builder_optimizer, 
										forbidder_optimizer, 
										batch_builder_observations,
										batch_builder_actions,
										batch_forbidder_observations,
										batch_forbidder_actions,
										device, 
										batch_stats)
			print(f"Backpropagation took: {time.time() - start_backprop} secs")

			eval_stats = evaluate(game, batch, builder_policy, forbidder_policy, device)
			
			print(f"Eval Statistics:")
			for key, value in eval_stats.items():
				print(key, value)
			
			training_stats.append((batch_stats, None))

			checkpoint(builder_policy, 
							forbidder_policy, 
							builder_optimizer, 
							forbidder_optimizer, 
							training_stats, 
							eval_stats, 
							best_so_far)

			print()
			print()
		else:
			action_noise = Deterministic(size=N_TOKENS,device = device)

			batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats = collect_batch_of_trajectories(game, 
																																						BATCH_SIZE,
																																						batch, 
																																						builder_policy, 
																																						forbidder_policy, 
																																						action_noise,
																																						device)
		
			print(f"Batch {batch} Statistics:")
			for key, value in batch_stats.items():
				print(key, value)
			
			start_backprop = time.time()
			update_policies(builder_optimizer, 
										forbidder_optimizer, 
										batch_builder_actions, 
										batch_forbidder_actions, 
										batch_builder_returns, 
										batch_forbidder_returns, 
										device, 
										batch_stats)
			print(f"Backpropagation took: {time.time() - start_backprop} secs")

			eval_stats = evaluate(game, batch, builder_policy, forbidder_policy, device)
			
			print(f"Eval Statistics:")
			for key, value in eval_stats.items():
				print(key, value)
			
			training_stats.append((batch_stats, None))

			checkpoint(builder_policy, 
							forbidder_policy, 
							builder_optimizer, 
							forbidder_optimizer, 
							training_stats, 
							batch_stats, 
							best_so_far)

			print()
			print()


if __name__ == '__main__':
	main()
