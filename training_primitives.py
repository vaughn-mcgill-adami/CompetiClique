from config import *

from copy import deepcopy
from collections import deque

import torch
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from rich.progress import track
from rich.progress import Progress

class Deterministic():
	"""
	 dummy class that is used in place of a distribution over non-negative "noise vectors" of shape size
	"""
	def __init__(self, size, device):
		self.size = size
		self.zeros = torch.zeros(self.size).to(device)
		return
	def sample(self):
		return self.zeros

def tensorify(observations, actions_probs, actions_chosen, rewards, actions_per_turn):
	assert len(observations) == len(actions_probs) == len(actions_chosen)== len(rewards)
	if len(rewards) != 0:
		rewards = torch.tensor(list(rewards))

		discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(rewards[k:])) for k in range(len(rewards))])
		discounted_returns = discounted_returns.repeat_interleave(actions_per_turn)
		
		actions_probs = torch.cat(list(actions_probs))
		actions_chosen = torch.cat(list(actions_chosen))
		
		actions = actions_probs[torch.arange(len(actions_probs)), actions_chosen]

		return observations, actions, discounted_returns
	else:
		return torch.tensor([]), torch.tensor([]), torch.tensor([])

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
			builder_observations.append(torch.squeeze(observation))

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
			
			formatted_action_probs = torch.stack(formatted_action_probs, dim=0)
			formatted_action_chosen = torch.stack(formatted_action_chosen, dim=0)

			#print("before\n", observation)

			observation, reward, builder_continue = game.step(formatted_action_chosen, 'builder')

			#print("after\n", observation)

			builder_actions_probs.append(formatted_action_probs)
			builder_actions_chosen.append(formatted_action_chosen)
			builder_rewards.append(reward)
		else:
			forbidder_observations.append(torch.squeeze(observation))

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
		return builder_observations, builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_observations, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner, graphs_each_turn
	else:
		return builder_observations, builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_observations, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner


def collect_batch_of_trajectories(game, batch_size, batch : int, builder_policy, forbidder_policy, action_noise, device, evalu = False):
	batch_builder_observations = deque()
	batch_builder_actions = deque()
	batch_builder_returns = deque()
	batch_forbidder_observations = deque()
	batch_forbidder_actions = deque()
	batch_forbidder_returns = deque()

	batch_stats = {'average_game_length' : deque(),
				   'max_game_length' : 0,
				   'builder_wins' : 0,
				   'forbidder_wins' : 0,
				   'nobody_wins' : 0}
	
	for episode in track(range(batch_size), description = f'Batch: {batch}/{NUM_BATCHES} : playing {batch_size} games : '):
		builder_observations, builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_observations, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner = run_trajectory(game, builder_policy, forbidder_policy, action_noise, device, evalu = evalu)
		
		#print(f'episode: {episode}')
		#print(builder_observations, builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_observations, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, winner, sep='\n')
		
		batch_stats[winner] += 1
		batch_stats['average_game_length'].append(turn_number)
		

		batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_number)
		
		builder_observations, builder_actions, builder_discounted_return = tensorify(builder_observations, builder_actions_probs, builder_actions_chosen, builder_rewards, 2*game.M)
		forbidder_observations, forbidder_actions, forbidder_discounted_return = tensorify(forbidder_observations, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, game.N)
		#print('after processing:')
		#print(builder_observations, builder_actions, builder_rewards, forbidder_observations, forbidder_actions, forbidder_rewards, turn_number, winner, sep='\n')
		#print()
		if len(builder_observations) > 0:
			batch_builder_observations.append(builder_observations)
			batch_builder_actions.append(builder_actions)
			batch_builder_returns.append(builder_discounted_return)
		if len(forbidder_observations) > 0:
			batch_forbidder_observations.append(forbidder_observations)
			batch_forbidder_actions.append(forbidder_actions)
			batch_forbidder_returns.append(forbidder_discounted_return)

	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])
	
	return batch_builder_observations, batch_builder_actions, batch_builder_returns, batch_forbidder_observations, batch_forbidder_actions, batch_forbidder_returns, batch_stats


def collect_batch_of_brute_trajectories(game, batch_size, width, maxlookahead):
	builder_observations = deque() #flattened "trajectory" dimension
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
						#prevobs = obs
						obs, rew, cont = game.step(action, player='builder')
						#print(prevobs, action, graph.nodes, game.G.nodes)

						builder_actions.append(action)
					else:
						obs = torch.squeeze(obs)
						forbidder_observations.append(obs)
						action = torch.tensor(action)
						action = torch.flatten(action) + torch.tensor(VERTEX_VOCAB_STARTS_AT)
						#prevobs = obs
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

def pad_jagged_batch(batch, pad, device, pad_to=None, dim=-1):
	if len(batch) > 0:

		longest_trajectory_len = max(trajectory.shape[0] for trajectory in batch) if pad_to is None else pad_to

		pad_seq = []
		for k in range(dim, -1, 1):
			pad_seq = pad_seq + [0,0]

		batch = list(F.pad(trajectory, pad=pad_seq + [0, longest_trajectory_len - trajectory.shape[0]], value = pad) if len(trajectory) > 0 else torch.tensor([pad for k in range(longest_trajectory_len)]) for trajectory in batch)
		batch = torch.stack(batch, dim = 0).to(device)
		
	else:
		print('empty batch!')
	return batch

def evaluate(game, batch, builder_policy, forbidder_policy, device):
	batch_stats = {}

	with torch.no_grad():
		builder_policy.eval()
		forbidder_policy.eval()
		action_noise = Deterministic(N_TOKENS, device)
		batch_builder_observations, batch_builder_actions, batch_builder_returns, batch_forbidder_observations, batch_forbidder_actions, batch_forbidder_returns, batch_stats = collect_batch_of_trajectories(game, NUM_EVAL_SAMPLES, batch, builder_policy, forbidder_policy, action_noise, device)
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
"""
def pretrain_batch(game, builder, forbidder, device, training_stats):
	batch_builder_observations, batch_builder_actions, batch_forbidder_observations, batch_forbidder_actions, batch_stats = collect_batch_of_brute_trajectories(game, PRETRAIN_BATCH_SIZE, WIDTH, MAXLOOKAHEAD)

	print(f"Batch {batch} (Supervised) Statistics:")
	for key, value in batch_stats.items():
		print(key, value)

	start_backprop = time.time()
	update_pretraining_policies(builder.policy,
								forbidder.policy,
								builder.optimizer, 
								forbidder.optimizer, 
								batch_builder_observations,
								batch_builder_actions,
								batch_forbidder_observations,
								batch_forbidder_actions,
								device, 
								batch_stats)
	print(f"Backpropagation took: {time.time() - start_backprop} secs")

	eval_stats = evaluate(game, batch, builder.policy, forbidder.policy, device)
	
	print(f"Eval Statistics:")
	for key, value in eval_stats.items():
		print(key, value)
	
	training_stats.append((batch_stats, eval_stats))

	builder.checkpoint(BUILDERPOLICYOPTPATH)
	forbidder.checkpoint(FORBIDDERPOLICYOPTPATH)

	print()
	print()
"""