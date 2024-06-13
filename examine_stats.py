import torch

BUILDERPOLICYOPTPATH = "builder_policy_opt.pt"
FORBIDDERPOLICYOPTPATH = "forbidder_policy_opt.pt"

builder_training_stats = torch.load(BUILDERPOLICYOPTPATH)['training_stats']
forbidder_training_stats = torch.load(FORBIDDERPOLICYOPTPATH)['training_stats']

for k, batch_stats in builder_training_stats.items():
	print(k, batch_stats['average_builder_return'], batch_stats['average_forbidder_return'], batch_stats['average_game_length'])