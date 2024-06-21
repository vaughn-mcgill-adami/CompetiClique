from config import *
from agents import Deterministic

from copy import deepcopy
from collections import deque

import torch
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from rich.progress import track
from rich.progress import Progress

def tensorify(observations, actions_probs, actions_chosen, rewards):
	assert len(observations) == len(actions_probs) == len(actions_chosen)== len(rewards)
	assert len(rewards) != 0
	rewards = torch.tensor(list(rewards))

	discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(rewards[k:])) for k in range(len(rewards))])
	discounted_returns = discounted_returns.repeat_interleave(2*game.M)
	
	actions_probs = torch.cat(list(actions_probs))
	actions_chosen = torch.cat(list(actions_chosen))

	actions = actions_probs[torch.arange(len(actions_probs)), actions_chosen]

	return observations, actions_probs, discounted_returns

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
		builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, builder_observations, forbidder_observations, winner = run_trajectory(game, builder_policy, forbidder_policy, action_noise, device, evalu = evalu)
		#print(f"previous observation on turn {turn_number}:", prevobs)
		#print(f"previous action on turn {turn_number}", builder_actions_chosen[-1] if turn_number%2==1 else forbidder_actions_chosen[-1] if len(forbidder_actions_chosen)!=0 else "empty")
		batch_stats[winner] += 1
		batch_stats['average_game_length'].append(turn_number)
		batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_number)
		
		builder_observations, builder_actions, builder_discounted_return = tensorify(builder_observations, )

		if len(forbidder_rewards) != 0:
			print(len(forbidder_rewards) == len(forbidder_observations))
			forbidder_rewards = torch.tensor(list(forbidder_rewards))
			
			forbidder_discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(forbidder_rewards[k:])) for k in range(len(forbidder_rewards))])
			forbidder_discounted_returns = forbidder_discounted_returns.repeat_interleave(game.N) #TODO: Possibly move before the discounting? Might keep it the way it is b/c good heuristic for larger M, N
			#print("N = ", game.N)

			forbidder_actions_probs = torch.cat(list(forbidder_actions_probs))
			forbidder_actions_chosen = torch.cat(list(forbidder_actions_chosen))

			forbidder_actions = forbidder_actions_probs[torch.arange(len(forbidder_actions_probs)), forbidder_actions_chosen]

			batch_forbidder_actions.append(forbidder_actions)
			batch_forbidder_returns.append(forbidder_discounted_returns)
			batch_forbidder_observations.append(forbidder_observations)

	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])
	
	return batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats


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
		return builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, builder_observations, forbidder_observations, winner


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

		print('builder_loss',builder_loss.item())

		builder_loss.backward()
		builder_optimizer.step()
		builder_optimizer.zero_grad()
	if len(forbidder_batch) != 0:
		FX = forbidder_batch[:-1]
		FY = forbidder_batch[1:]

		indices = torch.arange(N_TOKENS).to(device)[None, None, :] == FY[:, :, None]
		forbidder_loss = -torch.mean(torch.log(forbidder_policy(FX)[indices]))

		print('forbidder_loss', forbidder_loss.item())

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