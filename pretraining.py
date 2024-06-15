import torch

from simple_decoder_transformer import SimpleDecoderTransformer
from config import *
from main import run_trajectory
from competiclique_the_game import CompetiClique

from rich.progress import track

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
	#generates data for a subproblem, namely listing edges in an order so that at least one vertex of each edge is already present in the graph.
	
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

def main():
	device = torch.device(DEVICE)

	model = SimpleDecoderTransformer(L=LAYERS, H=HEADS, d_e=EMBEDDING_DIM, d_mlp = MLP_DIM).to(device)
	pre_training_optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

	batch_size = BATCH_SIZE
	episode_length = 10

	epochs = 100
	loss = torch.tensor(float('inf'))

	PATH = "supervised_pretrained_model.pt"

	prevloss = None

	game = CompetiClique(3, 1, 1)

	for epoch in range(epochs):
		prevloss = loss.item()
		print('generating batch...')
		batch = generate_batch(batch_size=batch_size, episode_length=episode_length).to(device)

		Y = batch[:-1]
		X = batch[1:]

		indices = torch.arange(N_TOKENS).to(device)[None, None, :] == Y[:, :, None]
		loss = -torch.mean(torch.log(model(X)[indices]))

		print('updating parameters with batch...')
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

		if SAVE_A_PRETRAIN_TRAJECTORY_PATH:
			torch.save(run_trajectory(
				game, model, model, device, evalu=True
			), SAVE_A_PRETRAIN_TRAJECTORY_PATH)
				
		print(epoch, loss.item())

if __name__ == "__main__":
	main()