import torch
from torch import nn
from torch.nn.functional import softmax

from config import *

class SimpleDecoderTransformer(nn.Module):
	def __init__(self, L : int, H : int, d_e : int, d_mlp : int):
		super().__init__()
		self.vertex_embedding = nn.Embedding(num_embeddings = N_TOKENS,
									   		 embedding_dim = d_e
											)
		self.position_embedding = nn.Embedding(num_embeddings = POSITIONS,
										 	   embedding_dim = d_e
										 	  )
		self.trunk = nn.ModuleList(
			[ nn.ModuleList([
				nn.BatchNorm1d(d_e),
				nn.MultiheadAttention(d_e, H, dropout=0.0, batch_first=True),
				nn.BatchNorm1d(d_e),
				nn.Linear(in_features=d_e, 
			  			  out_features=d_mlp),
				nn.GELU(),
				nn.Linear(d_mlp, d_e)]
		 	) for layer in range(L)]
		)

		self.final_batch_norm = nn.BatchNorm1d(d_e)
		self.final_linear = nn.Linear(d_e, N_TOKENS)

	def get_causal_mask(self, timesteps):
		mask = torch.tensor([[source_time_step > target_time_step for source_time_step in range(timesteps)] for target_time_step in range(timesteps)])
		return mask

	def forward(self, X):
		mask = self.get_causal_mask(timesteps=X.shape[-1])
		#print(mask)
		
		X = self.vertex_embedding(X) + self.position_embedding(X)

		#print(X.shape)

		for block in self.trunk:
			X = X.transpose(-1,-2)
		#	print(X.shape)
			X = block[0](X) # batch norm
		#	print(X.shape)
			X = X.transpose(-1,-2)
		#	print(X.shape)
			X = X + block[1](X, X, X, need_weights=False, attn_mask=mask, is_causal=True)[0] # multihead attention
		#	print(X.shape)
			X = X.transpose(-1,-2)
		#	print(X.shape)
			X = block[2](X) # batch norm again
		#	print(X.shape)
			X = X.transpose(-1,-2)
		#	print(X.shape)
			X = X + block[5](block[4](block[3](X))) # 
		#	print(X.shape)
		X = X.transpose(-1,-2)
		#print(X.shape)
		X = self.final_batch_norm(X) # batch norm
		#print(X.shape)
		X = X.transpose(-1,-2)
		#print(X.shape)
		X = softmax(self.final_linear(X), dim=-1)
		#print(X.shape)
		return X