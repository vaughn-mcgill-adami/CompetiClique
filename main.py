from competiclique_the_game import CompetiClique
from simple_decoder_transformer import SimpleDecoderTransformer#, set_batch_norm_momentum
from config import *
from training import *
from agents import ActorCriticAgent, Deterministic


import time
import argparse

import torch

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
	
	builder = ActorCriticAgent(agent_file=BUILDERPOLICYOPTPATH,
							   player_name='builder',
							   policy_architecture_args=NN_ARCH_ARGS,
							   critic_architecture_args=None,
							   policy_training_args=TRAINING_PARAMS,
							   critic_training_args=None,
							   action_noise=Deterministic(N_TOKENS, device),
							   device=device)
	
	forbidder = ActorCriticAgent(agent_file=FORBIDDERPOLICYOPTPATH,
								 player_name='forbidder',
							   	 policy_architecture_args=NN_ARCH_ARGS,
							   	 critic_architecture_args=None,
							   	 policy_training_args=TRAINING_PARAMS,
								 critic_training_args=None,
								 action_noise=Deterministic(N_TOKENS, device),
							   	 device=device)
	
	training_stats = builder.training_stats

	if BUILDERPOLICYOPTPATH is not None or FORBIDDERPOLICYOPTPATH is not None:
		print(f'WARNING: training will overwrite existing weights of {'builder'}')
	
	for batch in range(NUM_BATCHES):
		builder.train()
		forbidder.train()
		
		
		action_noise = Deterministic(size=N_TOKENS,device = device)

		start_collect = time.time()
		batch_builder_actions, batch_builder_returns, batch_forbidder_actions, batch_forbidder_returns, batch_stats = collect_batch_of_trajectories(game, 
																																					BATCH_SIZE,
																																					batch, 
																																					builder.policy, 
																																					forbidder.policy, 
																																					action_noise,
																																					device)
		print(f"Collecting batch took {time.time() - start_collect} secs")
		print(f"Batch {batch} (RL) Statistics:")
		for key, value in batch_stats.items():
			print(key, value)
		
		start_backprop = time.time()
		update_policies(builder.policy_optimizer, 
						forbidder.policy_optimizer, 
						batch_builder_actions, 
						batch_forbidder_actions, 
						batch_builder_returns, 
						batch_forbidder_returns, 
						device, 
						batch_stats)
		print(f"Backpropagation took: {time.time() - start_backprop} secs")

		start_eval = time.time()
		eval_stats = evaluate(game, batch, builder.policy, forbidder.policy, device)
		print(f"Eval took : {time.time() - start_eval} secs")

		print(f"Eval Statistics:")
		for key, value in eval_stats.items():
			print(key, value)
		
		training_stats.append((batch_stats, eval_stats))
		
		builder.checkpoint(BUILDERPOLICYOPTPATH, training_stats)
		forbidder.checkpoint(FORBIDDERPOLICYOPTPATH, training_stats)

		print()
		print()


if __name__ == '__main__':
	main()