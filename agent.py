import torch
from torch import nn
from torch import optim

from simple_decoder_transformer import SimpleDecoderTransformer

class ActorCriticAgent():
	def __init__(self, policy_architecture_args, critic_architecture_args, policy_training_args, critic_training_args, device):
		self.policy = SimpleDecoderTransformer(**policy_architecture_args)
		self.critic = SimpleDecoderTransformer(**critic_architecture_args, n_out = 1, activation=nn.Identity)

		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = policy_training_args['learning_rate'])
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_training_args['learning_rate'])
		
		self.device = device

		self.policy.to(device)
		self.critic.to(device)

	def load(self, path):
		agent_state = torch.load(path, map_location=self.device)

		training_stats = agent_stats['training_stats']

	def train()

	def forward(self): #TODO: make functional notation
		pass