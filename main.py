from competiclique_the_game import CompetiClique
from simple_decoder_transformer import SimpleDecoderTransformer

from copy import deepcopy

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from collections import deque

from tqdm import tqdm

from config import *

def edge_condition(x : torch.Tensor):
	assert len(x) % 2 == 0
	adjacency_list = dict()

	for k in range(len(x)//2):
		u = x[2*k].item()
		v = x[2*k+1].item()

		if u == v:
			return False
		if len(adjacency_list) == 0:
			adjacency_list[u] = {v}
			adjacency_list[v] = {u}
		else:
			if u not in adjacency_list.keys() and v not in adjacency_list.keys():
				return False
			if u in adjacency_list.keys():
				if v in adjacency_list[u]:
					return False
			if v in adjacency_list.keys():
				if u in adjacency_list[v]:
					return False
		adjacency_list[u] = {v}
		adjacency_list[v] = {u}
	return True

def get_forbidden_mask(size):
	assert size % 2 == 0

	k = 0
	forbidden_mask = []
	while k < size//2:
		pair = torch.randint(low=AVAILABLE_TOKEN, high=FORBIDDEN_TOKEN+1, size=(2,)) # note these two tokens are consecutive.
		if pair[0].item() != FORBIDDEN_TOKEN or pair[1].item() != FORBIDDEN_TOKEN:
			k += 1
			forbidden_mask.append(pair)
	forbidden_mask = torch.cat(forbidden_mask, dim = 0)
	return forbidden_mask

def generate_batch(batch_size : int, episode_length : int):
	"""
	generates data for a subproblem, namely listing edges in an order so that at least one vertex of each edge is already present in the graph.
	"""

	batch = []

	curr_episode = 0
	while curr_episode < batch_size:
		x = torch.randint(low=0,high=VERTEX_VOCABULARY,size=(2*episode_length,))
		if edge_condition(x):
			for end_observation_index in range(1,episode_length):
				forbidden_mask = get_forbidden_mask(size=2*end_observation_index)
				xp = torch.cat((
						torch.flatten(torch.stack((x[:2*end_observation_index],forbidden_mask), dim=0).transpose(-1,-2)),
						torch.tensor([END_OBSERVATION_TOKEN]),
						x[2*end_observation_index:]), dim=0)
				batch.append(xp)
			curr_episode += 1
	return torch.nested.nested_tensor(batch).to_padded_tensor(padding=PAD_TOKEN)

"""
pre_training_optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)


Pretraining like this seems to actually make the RL training harder...

batch_size = 1000
episode_length = 10

epochs = 100
loss = torch.tensor(float('inf'))

PATH = "supervised_pretrained_model.pt"

prevloss = None

for epoch in range(epochs):
	prevloss = loss.item()
	batch = generate_batch(batch_size=batch_size, episode_length=episode_length).to(device)

	Y = batch[:-1]
	X = batch[1:]

	indices = torch.arange(N_TOKENS).to(device)[None, None, :] == Y[:, :, None]
	loss = -torch.mean(torch.log(model(X)[indices]))

	loss.backward()
	pre_training_optimizer.step()
	pre_training_optimizer.zero_grad()

	stop = prevloss < loss.item()
	if epoch % 5 == 0 or stop or epoch == epochs-1:
		torch.save({
			'epoch': epochs,
			'loss': loss.item(),
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': pre_training_optimizer.state_dict()
			}, PATH)
		if stop:
			break
			 
	print(epoch, loss.item())

model.load_state_dict(torch.load(PATH)['model_state_dict'])
#model.eval()

if eval_only:
		model.eval()
else:
		model.train() 
"""
embedding_dim = 64
mlp_dim = 96

device = torch.device("cpu")

model = SimpleDecoderTransformer(L=6, H=4, d_e=embedding_dim, d_mlp = mlp_dim).to(device)

model.train()

eval_only = False

num_batches = 100
batch_size = 1000
lr = 0.002
discount_factor = 0.9

BESTBUILDERPOLICYOPTPATH = "best_builder_policy_opt.pt"
BESTFORBIDDERPOLICYOPTPATH = "best_forbidder_policy_opt.pt"
BUILDERPOLICYOPTPATH = "builder_policy_opt.pt"
FORBIDDERPOLICYOPTPATH = "forbidder_policy_opt.pt"

LOAD_SAVED_WEIGHTS = True

game = CompetiClique(clique_size = 3,
							edges_per_builder_turn=1,
						vertices_per_forbidder_turn=1
						)

"""
Vanilla policy gradient implementation.
"""
builder_policy = deepcopy(model).to(device)#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)
forbidder_policy = deepcopy(model).to(device)#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)

if LOAD_SAVED_WEIGHTS:
	builder_policy.load_state_dict(torch.load(BUILDERPOLICYOPTPATH)['builder_policy_state_dict'])
	forbidder_policy.load_state_dict(torch.load(FORBIDDERPOLICYOPTPATH)['forbidder_policy_state_dict'])

builder_optimizer = torch.optim.Adam(builder_policy.parameters(), lr=lr)
forbidder_optimizer = torch.optim.Adam(forbidder_policy.parameters(), lr=lr)

builder_policy.train()
forbidder_policy.train()

training_stats = {}

best_so_far = {"builder" : float('-inf'), 
								 "forbidder" : float('-inf')
								}

for batch in range(num_batches):
	#batch_builder_observations = deque()
	batch_builder_actions = deque()
	batch_builder_returns = deque()

	#batch_forbidder_observations = deque()
	batch_forbidder_actions = deque()
	batch_forbidder_returns = deque()

	batch_stats = {'average_game_length' : deque(),
						'max_game_length' : 0}

	for episode in tqdm(range(batch_size)):
		#builder_observations = deque()
		builder_actions_probs = deque()
		builder_actions_chosen = deque()
		builder_rewards = deque()
		#forbidder_observations = deque()
		forbidder_actions_probs = deque()
		forbidder_actions_chosen = deque()
		forbidder_rewards = deque()

		observation = game.reset()
		turn_number = 0

		prevobs = None

		while observation is not None:
			#prevobs = observation
			if turn_number % 2 == 0:
				#builder_observations.append(observation)

				formatted_action_probs = []
				formatted_action_chosen = []
				for k in range(2*game.M):
					observation = observation.to(device)
					action_probs = builder_policy(observation)[0][-1]
					action_chosen = Categorical(probs = action_probs).sample()
					formatted_action_probs.append(action_probs)
					formatted_action_chosen.append(action_chosen)
					observation = torch.cat((observation,torch.unsqueeze(torch.unsqueeze(action_chosen,dim=0),dim=0)), dim=1)
				formatted_action_probs = torch.stack(formatted_action_probs, dim=0)
				formatted_action_chosen = torch.stack(formatted_action_chosen, dim=0)

				#print("before\n", observation)

				observation, reward = game.step(formatted_action_chosen, 'builder')

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
					action_chosen = Categorical(probs = action_probs).sample()
					formatted_action_probs.append(action_probs)
					formatted_action_chosen.append(action_chosen)
					observation = torch.cat((observation,torch.unsqueeze(torch.unsqueeze(action_chosen,dim=0),dim=0)), dim=1)
				formatted_action_probs = torch.stack(formatted_action_probs, dim=0)
				formatted_action_chosen = torch.stack(formatted_action_chosen, dim=0)

				observation, reward = game.step(formatted_action_chosen, 'forbidder')

				forbidder_actions_probs.append(formatted_action_probs)
				forbidder_actions_chosen.append(formatted_action_chosen)
				forbidder_rewards.append(reward)
			turn_number += 1

		#print(f"previous observation on turn {turn_number}:", prevobs)
		#print(f"previous action on turn {turn_number}", builder_actions_chosen[-1] if turn_number%2==1 else forbidder_actions_chosen[-1] if len(forbidder_actions_chosen)!=0 else "empty")
		batch_stats['average_game_length']
		batch_stats['average_game_length'].append(turn_number)
		batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_number)

		if len(builder_rewards) != 0:
			builder_rewards = torch.tensor(list(builder_rewards))

			#print("builder_rewards:", builder_rewards)

			builder_discounted_returns = torch.tensor([sum(discount_factor**i*reward for i, reward in enumerate(builder_rewards[k:])) for k in range(len(builder_rewards))])
			builder_discounted_returns = builder_discounted_returns.repeat_interleave(2*game.M)

			builder_actions_probs = torch.cat(list(builder_actions_probs))
			builder_actions_chosen = torch.cat(list(builder_actions_chosen))

			builder_actions = builder_actions_probs[torch.arange(len(builder_actions_probs)), builder_actions_chosen]

			#print("builder_discounted_returns:", builder_discounted_returns)

			batch_builder_actions.append(builder_actions)
			batch_builder_returns.append(builder_discounted_returns)


		if len(forbidder_rewards) != 0:
			forbidder_rewards = torch.tensor(list(forbidder_rewards))

			#print("forbidder_rewards:", forbidder_rewards)

			forbidder_discounted_returns = torch.tensor([sum(discount_factor**i*reward for i, reward in enumerate(forbidder_rewards[k:])) for k in range(len(forbidder_rewards))])
			forbidder_discounted_returns = forbidder_discounted_returns.repeat_interleave(game.N)

			forbidder_actions_probs = torch.cat(list(forbidder_actions_probs))
			forbidder_actions_chosen = torch.cat(list(forbidder_actions_chosen))

			forbidder_actions = forbidder_actions_probs[torch.arange(len(forbidder_actions_probs)), forbidder_actions_chosen]

			batch_forbidder_actions.append(forbidder_actions)
			batch_forbidder_returns.append(forbidder_discounted_returns)

			#print("forbidder_discounted_returns:", forbidder_discounted_returns)
			#print("forbidden_token = ", FORBIDDEN_TOKEN)
			#print("available_token = ", AVAILABLE_TOKEN)
			#return

	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])

	if len(batch_builder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_builder_actions)

		batch_builder_actions = list(F.pad(actions, (0,longest_trajectory_len - len(actions)), value = 1.0) for actions in batch_builder_actions)
		batch_builder_actions = torch.stack(batch_builder_actions, dim = 0)

		#print(batch_builder_actions.isnan().any())

		batch_builder_returns = (list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value = 0.0) for returns in batch_builder_returns))
		batch_builder_returns = torch.stack(batch_builder_returns, dim = 0).to(device)

		#print(batch_builder_returns.isnan().any())

		#print("batch_builder_actions = ", batch_builder_actions)
		#print("batch_builder_returns = ", batch_builder_returns)

		if not eval_only:
			intermediate = (torch.log(batch_builder_actions)*batch_builder_returns).sum(dim=-1)

			builder_loss = -torch.mean(intermediate)

			builder_loss.backward()
			builder_optimizer.step()
			builder_optimizer.zero_grad()

			batch_stats['builder_loss'] = builder_loss.cpu().item()

		batch_stats['average_builder_return'] = torch.mean(batch_builder_returns[:,0]).cpu().item()
	if len(batch_forbidder_actions) != 0:
		longest_trajectory_len = max(len(trajectory) for trajectory in batch_forbidder_actions)

		batch_forbidder_actions = list(F.pad(actions, (0, longest_trajectory_len - len(actions)), value= 1.0) for actions in batch_forbidder_actions)
		batch_forbidder_actions = torch.stack(batch_forbidder_actions, dim = 0)
		
		batch_forbidder_returns = list(F.pad(returns, (0, longest_trajectory_len - len(returns)), value= 0.0) for returns in batch_forbidder_returns)
		batch_forbidder_returns = torch.stack(batch_forbidder_returns, dim = 0).to(device)

		#print("batch_forbidder_actions = ", batch_forbidder_actions)
		#print("batch_forbidder_returns = ", batch_forbidder_returns)

		if not eval_only:
			intermediate = (torch.log(batch_forbidder_actions)*batch_forbidder_returns).sum(dim=-1)

			forbidder_loss = -torch.mean(intermediate)

			forbidder_loss.backward()
			forbidder_optimizer.step()
			forbidder_optimizer.zero_grad()

			batch_stats['forbidder_loss'] = forbidder_loss.cpu().item()

		batch_stats['average_forbidder_return'] = torch.mean(batch_forbidder_returns[:,0]).cpu().item()
	training_stats[batch] = batch_stats
	
	torch.save(
			{
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BUILDERPOLICYOPTPATH
		)
	torch.save(
			{
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, FORBIDDERPOLICYOPTPATH
		)
	if best_so_far['builder'] < batch_stats['average_builder_return']:
		torch.save(
			{
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BESTBUILDERPOLICYOPTPATH
		)
		best_so_far['builder'] = batch_stats['average_builder_return']
	if best_so_far['forbidder'] < batch_stats['average_forbidder_return']:
		torch.save(
			{
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, BESTFORBIDDERPOLICYOPTPATH
		)
		best_so_far['forbidder'] = batch_stats['average_forbidder_return']
	print(f"batch {batch}:", batch_stats)
	print()
	print()