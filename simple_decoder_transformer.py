import torch
from torch import nn
from torch.nn.functional import softmax

from config import *
"""
def set_batch_norm_momentum(policy, momentum):
	for block in policy.trunk:
		block[0].momentum = momentum
		block[2].momentum = momentum
"""

class SimpleDecoderTransformer(nn.Module):
	def __init__(self, L : int, H : int, d_e : int, d_mlp : int, n_tokens : int, n_positions : int, n_out : int):
		super().__init__()
		self.vertex_embedding = nn.Embedding(num_embeddings = n_tokens,
										 		 embedding_dim = d_e
											)
		self.position_embedding = nn.Embedding(num_embeddings = n_positions,
										 		 embedding_dim = d_e
										 		)
		self.trunk = nn.ModuleList(
			[ nn.ModuleList([
				nn.LayerNorm(d_e),
				nn.MultiheadAttention(d_e, H, dropout=0.0, batch_first=True),
				nn.LayerNorm(d_e),
				nn.Linear(in_features=d_e, 
								out_features=d_mlp),
				nn.GELU(),
				nn.Linear(d_mlp, d_e)]
		 	) for layer in range(L)]
		)

		self.final_layer_norm = nn.LayerNorm(d_e)
		self.final_linear = nn.Linear(d_e, n_out)

		self.my_device_for_mask = torch.device(DEVICE)

	def get_causal_mask(self, timesteps):
		mask = torch.tensor([[source_time_step > target_time_step for source_time_step in range(timesteps)] for target_time_step in range(timesteps)]).to(self.my_device_for_mask)
		return mask

	def forward(self, X):
		mask = self.get_causal_mask(timesteps=X.shape[-1])
		#print(mask)
		#print(X.shape)
		X = self.vertex_embedding(X) + self.position_embedding(X)

		#print(X.shape)

		for block in self.trunk:
		#	X = X.transpose(-1,-2)
		#	print(X.shape)
			X = block[0](X) # layer norm
		#	print(X.shape)
		#	X = X.transpose(-1,-2)
		#	print(X.shape)
			X = X + block[1](X, X, X, need_weights=False, attn_mask=mask, is_causal=True)[0] # multihead attention
		#	print(X.shape)
		#	X = X.transpose(-1,-2)
		#	print(X.shape)
			X = block[2](X) # layer norm again
		#	print(X.shape)
			X = X + block[5](block[4](block[3](X))) # 
		#	print(X.shape)
		#X = X.transpose(-1,-2)
		#print(X.shape)
		X = self.final_layer_norm(X) # layer norm
		#print(X.shape)
		#X = X.transpose(-1,-2)
		#print(X.shape)
		X = softmax(self.final_linear(X), dim=-1)
		#print(X.shape)
		return X