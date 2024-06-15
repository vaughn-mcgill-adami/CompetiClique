from competiclique_the_game import CompetiClique
from simple_decoder_transformer import SimpleDecoderTransformer
from config import *

from copy import deepcopy
from collections import deque

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from rich.progress import track

def load_training_history(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, device):
	builder_state = torch.load(BUILDERPOLICYOPTPATH, map_location=device)
	forbidder_state = torch.load(FORBIDDERPOLICYOPTPATH, map_location=device)

	training_stats = builder_state['training_stats']

	builder_policy.load_state_dict(builder_state['builder_policy_state_dict'])
	print('loaded builder policy')
	forbidder_policy.load_state_dict(forbidder_state['forbidder_policy_state_dict'])
	print('loaded builder optimizer')

	builder_optimizer.load_state_dict(builder_state['builder_optimizer_state_dict'])
	print('loaded forbidder policy')
	forbidder_optimizer.load_state_dict(forbidder_state['forbidder_optimizer_state_dict'])
	print('loaded forbidder optimizer')

	assert len(training_stats) != 0
	print(len(training_stats))

	print(training_stats)

	best_so_far = {"builder" : max(batch_stats['average_builder_return'] for batch_stats in training_stats),
								"forbidder" : max(batch_stats['average_forbidder_return'] for batch_stats in training_stats)
								}
	print('best builder average return :', best_so_far['builder'])
	print('best forbidder average return :', best_so_far['forbidder'])
	print('latest builder average return :', training_stats[len(training_stats) - 1]['average_builder_return'])
	print('latest forbidder average return :', training_stats[len(training_stats) - 1]['average_forbidder_return'])

	return training_stats, best_so_far


def collect_batch_of_trajectories(game, batch : int, builder_policy, forbidder_policy, device, evalu = False):
	#batch_builder_observations = deque()
	batch_builder_actions = deque()
	batch_builder_returns = deque()

	#batch_forbidder_observations = deque()
	batch_forbidder_actions = deque()
	batch_forbidder_returns = deque()

	batch_stats = {'average_game_length' : deque(),
								'max_game_length' : 0}

	for episode in track(range(BATCH_SIZE), description = f'Batch: {batch}/{NUM_BATCHES} : playing game : '):
		builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number = run_trajectory(game, builder_policy, forbidder_policy, device, evalu = evalu)
		#print(f"previous observation on turn {turn_number}:", prevobs)
		#print(f"previous action on turn {turn_number}", builder_actions_chosen[-1] if turn_number%2==1 else forbidder_actions_chosen[-1] if len(forbidder_actions_chosen)!=0 else "empty")
		batch_stats['average_game_length']
		batch_stats['average_game_length'].append(turn_number)
		batch_stats['max_game_length'] = max(batch_stats['max_game_length'], turn_number)

		if len(builder_rewards) != 0:
			builder_rewards = torch.tensor(list(builder_rewards))

			#print("builder_rewards:", builder_rewards)

			builder_discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(builder_rewards[k:])) for k in range(len(builder_rewards))])
			builder_discounted_returns = builder_discounted_returns.repeat_interleave(2*game.M)

			builder_actions_probs = torch.cat(list(builder_actions_probs))
			builder_actions_chosen = torch.cat(list(builder_actions_chosen))

			builder_actions = builder_actions_probs[torch.arange(len(builder_actions_probs)), builder_actions_chosen]

			batch_builder_actions.append(builder_actions)
			batch_builder_returns.append(builder_discounted_returns)


		if len(forbidder_rewards) != 0:
			forbidder_rewards = torch.tensor(list(forbidder_rewards))
			
			forbidder_discounted_returns = torch.tensor([sum(DISCOUNT_FACTOR**i*reward for i, reward in enumerate(forbidder_rewards[k:])) for k in range(len(forbidder_rewards))])
			forbidder_discounted_returns = forbidder_discounted_returns.repeat_interleave(game.N)

			forbidder_actions_probs = torch.cat(list(forbidder_actions_probs))
			forbidder_actions_chosen = torch.cat(list(forbidder_actions_chosen))

			forbidder_actions = forbidder_actions_probs[torch.arange(len(forbidder_actions_probs)), forbidder_actions_chosen]

			batch_forbidder_actions.append(forbidder_actions)
			batch_forbidder_returns.append(forbidder_discounted_returns)

	batch_stats['average_game_length'] = sum(batch_stats['average_game_length'])/len(batch_stats['average_game_length'])
	
	return batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats


def run_trajectory(game, builder_policy, forbidder_policy, device, evalu=False):
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

	if evalu:
		builder_policy.eval()
		forbidder_policy.eval()

	builder_continue = True
	forbidder_continue = True

	while observation is not None and (builder_continue or forbidder_continue):
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

	if evalu:
		return builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number, builder_observations, forbidder_observations, graphs_each_turn
	else:
		return builder_actions_probs, builder_actions_chosen, builder_rewards, forbidder_actions_probs, forbidder_actions_chosen, forbidder_rewards, turn_number


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


def checkpoint(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, training_stats, batch_stats, best_so_far):
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

def main():
	device = torch.device(DEVICE)

	model = SimpleDecoderTransformer(L=LAYERS, H=HEADS, d_e=EMBEDDING_DIM, d_mlp = MLP_DIM).to(device)

	game = CompetiClique(clique_size = CLIQUE_SIZE,
								edges_per_builder_turn=EDGES_PER_BUILDER_TURN,
							vertices_per_forbidder_turn=VERTICES_PER_FORBIDDER_TURN
							)

	builder_policy = deepcopy(model)#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)
	forbidder_policy = deepcopy(model)#SimpleDecoderTransformer(L = 2, H = 4, d_e = 32, d_mlp = 48)

	builder_optimizer = torch.optim.Adam(builder_policy.parameters(), lr=LEARNING_RATE)
	forbidder_optimizer = torch.optim.Adam(forbidder_policy.parameters(), lr=LEARNING_RATE)

	training_stats = []

	best_so_far = {"builder" : float('-inf'), 
									"forbidder" : float('-inf')
									}

	if LOAD_SAVED_WEIGHTS:
		training_stats, best_so_far = load_training_history(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, device)
	else:
		print('WARNING: training will overwrite existing weights because LOAD_SAVED_WEIGHTS = False in config.py')
	
	print('start training? (y to continue / n to cancel): ')
	inp = None
	while inp != 'y':
		inp = input()
		if inp == 'n':
			print('training cancelled.')
			return

	builder_policy.to(device)
	forbidder_policy.to(device)

	builder_policy.train()
	forbidder_policy.train()

	for batch in range(NUM_BATCHES):
		batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats = collect_batch_of_trajectories(game, 
																																																																								batch, 
																																																																								builder_policy, 
																																																																								forbidder_policy, 
																																																																								device)

		print(f"Batch {batch} Statistics:")
		for key, value in batch_stats.items():
			print(key, value)
		
		update_policies(builder_optimizer, 
									 forbidder_optimizer, 
									 batch_builder_actions, 
									 batch_forbidder_actions, 
									 batch_builder_returns, 
									 batch_forbidder_returns, 
									 device, 
									 batch_stats)
		
		training_stats.append(batch_stats)

		checkpoint(builder_policy, 
						 forbidder_policy, 
						 builder_optimizer, 
						 forbidder_optimizer, 
						 training_stats, 
						 batch_stats, 
						 best_so_far)

		if SAVE_A_TRAJECTORY_PATH:
			torch.save(run_trajectory(
				game, builder_policy, forbidder_policy, device, evalu=True
			), SAVE_A_TRAJECTORY_PATH)
		
		print()
		print()

if __name__ == '__main__':
	main()