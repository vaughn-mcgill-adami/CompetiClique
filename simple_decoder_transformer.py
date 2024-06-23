import torch
from torch import nn
from torch.nn import functional as F

from config import *

class SimpleDecoderTransformer(nn.Module):
	def __init__(self, L : int, H : int, d_e : int, d_mlp : int, n_tokens : int, n_positions : int, n_out : int, activation = nn.Softmax(dim=-1)):
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
		self.activation = activation

		self.my_device = None# = torch.device(DEVICE) #change this.
	
	def update_embedding_sizes(self, vocab_size : int = 0, num_positions : int = 0, out_size : int = 0):
		with torch.no_grad():
			if vocab_size > 0 and out_size > 0:
				self.vertex_embedding.weight = nn.Parameter(torch.cat((self.vertex_embedding.weight, torch.randn(vocab_size - self.vertex_embedding.weight.shape[0], EMBEDDING_DIM).to(self.my_device) ), dim=0))
				self.final_linear.weight = nn.Parameter(torch.cat((self.final_linear.weight, torch.randn(out_size - self.final_linear.weight.shape[0], EMBEDDING_DIM).to(self.my_device)), dim=0))
				self.final_linear.bias = nn.Parameter(torch.cat((self.final_linear.bias, torch.randn(out_size - self.final_linear.bias.shape[0]).to(self.my_device) ), dim=0))
			if num_positions > 0:
				self.position_embedding.weight = nn.Parameter(torch.cat((self.position_embedding.weight, torch.randn(num_positions - self.position_embedding.weight.shape[0], EMBEDDING_DIM).to(self.my_device) ), dim=0))

	def get_causal_mask(self, timesteps):
		mask = torch.tensor([[source_time_step > target_time_step for source_time_step in range(timesteps)] for target_time_step in range(timesteps)]).to(self.my_device)
		return mask

	def forward(self, X):
		mask = self.get_causal_mask(timesteps=X.shape[-1])

		X = self.vertex_embedding(X) + self.position_embedding(X)

		for block in self.trunk:
			X = block[0](X)
			X = X + block[1](X, X, X, need_weights=False, attn_mask=mask, is_causal=True)[0] # multihead attention
			X = block[2](X)
			X = X + block[5](block[4](block[3](X))) # 

		X = self.final_layer_norm(X) # layer norm
		X = self.activation(self.final_linear(X))
		
		return X