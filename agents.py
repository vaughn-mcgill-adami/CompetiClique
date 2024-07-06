import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from simple_decoder_transformer import SimpleDecoderTransformer
from config import *
from training_primitives import *

class ActorCriticAgent():
	def __init__(self, agent_file = None,
							 player_name = None, 
							 policy_architecture_args = None, 
						 critic_architecture_args = None, 
						 policy_training_args = None, 
						 critic_training_args = None, 
						 action_noise = None, 
						 device = None):
		
		self.player_name = player_name

		self.policy_architecture_args = policy_architecture_args
		self.critic_architecture_args = critic_architecture_args
		self.policy_training_args = policy_training_args
		self.critic_training_args = critic_training_args
		
		self.device = device

		if agent_file is not None:
			self.load(agent_file)
		else:
			self.policy = SimpleDecoderTransformer(**policy_architecture_args)
			self.critic = SimpleDecoderTransformer(**critic_architecture_args, activation=nn.Identity())

			self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = policy_training_args['learning_rate'])
			self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_training_args['learning_rate'])
			
			self.device = device

			self.policy.to(device)
			self.critic.to(device)
			self.policy.my_device = device
			self.critic.my_device = device
			
			self.action_noise = action_noise
		
			self.player_name = player_name
			
			self.training_stats = []

			#self.last_critic_loss = 1.0

	def to(self, device):
		if type(device) == str:
			device = torch.device(device)
		self.policy.to(device)
		self.critic.to(device)
		self.policy.my_device = device
		self.critic.my_device = device

	def load(self, path):
		agent_state = torch.load(path, map_location=self.device)
		
		old_policy_n_tokens = agent_state['policy_state_dict']['vertex_embedding.weight'].shape[0]
		old_policy_n_positions = agent_state['policy_state_dict']['position_embedding.weight'].shape[0]
		old_policy_n_out = agent_state['policy_state_dict']['final_linear.bias'].shape[0]

		self.action_noise = agent_state['action_noise']

		self.policy = SimpleDecoderTransformer( L=LAYERS, 
												H=HEADS, 
												d_e=EMBEDDING_DIM,
												d_mlp=MLP_DIM,
												n_tokens=old_policy_n_tokens,
												n_positions=old_policy_n_positions,
												n_out=old_policy_n_out ).to(self.device)
		self.policy.my_device = self.device

		self.critic = SimpleDecoderTransformer( L=LAYERS, 
												H=HEADS, 
												d_e=EMBEDDING_DIM,
												d_mlp=MLP_DIM,
												n_tokens=old_policy_n_tokens,
												n_positions=old_policy_n_positions,
												n_out=1,
												activation=nn.Identity()).to(self.device)
		self.critic.my_device = self.device
	
		self.training_stats = agent_state['agent_training_stats']

		self.policy.load_state_dict(agent_state['policy_state_dict'])
		
		if self.policy_architecture_args is not None:
			self.policy.update_embedding_sizes(self.policy_architecture_args['n_tokens'],
										 		 self.policy_architecture_args['n_positions'],
												 self.policy_architecture_args['n_out'])
		if self.critic_architecture_args is not None:
			self.critic.update_embedding_sizes(self.critic_architecture_args['n_tokens'],
													 self.critic_architecture_args['n_positions'],
												 1)

		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = self.policy_training_args['learning_rate'])
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.critic_training_args['learning_rate'])

		if self.policy_architecture_args['n_tokens'] == old_policy_n_tokens and \
			 self.policy_architecture_args['n_positions'] == old_policy_n_positions and \
			 self.policy_architecture_args['n_out'] == old_policy_n_out:
			self.policy_optimizer.load_state_dict(agent_state['policy_optimizer_state_dict'])
			self.critic_optimizer.load_state_dict(agent_state['critic_optimizer_state_dict'])
				
	def values(self, batch_observations, no_grad = True):

		#batch   obs.shape = num_traj, jagged_traj_len, personal_max_observation_len
		#		    		 deque     tensor

		#desire: values.shape = num_traj, max_traj_len
		#						tensor

		# might want to test with/without padding, see what happens.
		# does it impact performance?
		
		max_obs_len = batch_observations.shape[-1]
		max_traj_len = batch_observations.shape[-2]
		
		mask = batch_observations == torch.tensor(END_OBSERVATION_TOKEN).to(self.device)
		
		batch_observations = batch_observations.reshape((-1, max_obs_len))
		
		#print("in values : batch_observations.requires_grad = ", batch_observations.requires_grad)

		if no_grad:
			with torch.no_grad():
				value = self.critic(batch_observations).reshape(-1, max_traj_len, max_obs_len)
		else:
			value = self.critic(batch_observations).reshape(-1, max_traj_len, max_obs_len)

		
		indices = torch.masked_fill(torch.cumsum(mask.int(), dim=-1), ~mask, value=0)
		value = torch.scatter(input=torch.zeros_like(value), dim=-1, index=indices, src=value)[...,1]
		#print('after value:', value.shape)
		batch_observations = batch_observations.reshape(-1, max_traj_len, max_obs_len)
		#print('batch_observations:', batch_observations.shape)
		#0 value for padding observations so we don't learn anything from them.
		zeromask = torch.tensor([[(batch_observations[traj][obs] == PAD_TOKEN).all() for obs in range(batch_observations.shape[1]) ]for traj in range(batch_observations.shape[0])]).to(self.device)
		value = torch.masked_fill(value, zeromask, value=0)
		#print('after overwriting padding:', value)
		#print("in values : value.requires_grad = ", value.requires_grad)
		return value

	def update_policy(self, in_batch_observations, in_batch_actions, in_batch_returns, batch_stats, in_actions_per_turn):
		batch_observations = in_batch_observations
		max_obs_len = max(len(obs) for trajectory in batch_observations for obs in trajectory)
		batch_observations = [pad_jagged_batch(trajectory, PAD_TOKEN, self.device, pad_to=max_obs_len) for trajectory in batch_observations]
		batch_observations = pad_jagged_batch(batch_observations, PAD_TOKEN, self.device, dim=-2)
		
		batch_actions = pad_jagged_batch(in_batch_actions, 1.0, self.device)
		batch_returns =	pad_jagged_batch(in_batch_returns, 0.0, self.device)
		
		value = self.values(batch_observations)
		
		batch_returns = batch_returns - value

		#print("in update_policy : batch_returns.requires_grad = ", batch_returns.requires_grad)

		loss_per_trajectory = (torch.log(batch_actions)*batch_returns).sum(dim=-1)
		loss = -torch.mean(loss_per_trajectory)#*0.05/self.last_critic_loss #0.05 is a factor representing the "ideal" loss.

		loss.backward()
		self.policy_optimizer.step()
		self.policy_optimizer.zero_grad()

		#logging
		batch_stats[f'average_{self.player_name}_return'] = torch.mean(batch_returns[:,0]).cpu().item()
		batch_stats[f'{self.player_name}_policy_loss'] = loss.cpu().item()

		del loss
		del loss_per_trajectory
	
	def update_critic(self, in_batch_observations, in_batch_returns, batch_stats, in_actions_per_turn):
		batch_observations = in_batch_observations
		max_obs_len = max(len(obs) for trajectory in batch_observations for obs in trajectory)
		batch_observations = [pad_jagged_batch(trajectory, PAD_TOKEN, self.device, pad_to=max_obs_len) for trajectory in batch_observations]
		batch_observations = pad_jagged_batch(batch_observations, PAD_TOKEN, self.device, dim=-2)
		value = self.values(batch_observations, no_grad=False)
		
		batch_returns =	pad_jagged_batch(in_batch_returns, 0.0, self.device)
		
		loss = torch.mean(torch.pow(value - batch_returns, 2)) #quadratic loss, for simplicity r.n.

		loss.backward()
		self.critic_optimizer.step()
		self.critic_optimizer.zero_grad()
		
		#self.last_critic_loss = loss.item()

		batch_stats[f'{self.player_name}_critic_loss'] = loss.cpu().item()

		del loss

	"""
	def pretrain_policies(self, batch_observations, batch_actions, batch_stats):
		#I don't think this works...
		batch = [torch.cat((observation, action)) for observation, action in zip(batch_observations, batch_actions)]
		batch = pad_jagged_batch(batch, PAD_TOKEN).to(self.device)
		
		assert len(batch) != 0
		X = batch[...,:-1,:]
		Y = batch[...,1:,:]

		indices = torch.arange(N_TOKENS).to(self.device)[None, None, :] == Y[:, :, None]
		loss = -torch.mean(torch.log(self.policy(X)[indices]))

		loss.backward()
		self.policy_optimizer.step()
		self.policy_optimizer.zero_grad()
	"""
	def checkpoint(self, path, training_stats):
		torch.save({
			"policy_state_dict" : self.policy.state_dict(),
			"critic_state_dict" : self.critic.state_dict(),
			"policy_optimizer_state_dict" : self.policy_optimizer.state_dict(),
			"critic_optimizer_state_dict" : self.critic_optimizer.state_dict(),
			"agent_training_stats" : training_stats,
			"device" : self.device,
			"action_noise" : self.action_noise, #TODO : make these be strings.
			"player_name" : self.player_name
		}, path)
		
	def train(self):
		self.policy.train()
		self.critic.train()
		
	def eval(self):
		self.policy.eval()
		self.critic.eval()