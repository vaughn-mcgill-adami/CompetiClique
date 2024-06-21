import torch
from torch import nn
from torch import optim

from simple_decoder_transformer import SimpleDecoderTransformer
from config import *

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
			#self.critic = SimpleDecoderTransformer(**critic_architecture_args, n_out = 1, activation=nn.Identity)

			self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = policy_training_args['learning_rate'])
			#self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_training_args['learning_rate'])
			
			self.device = device

			self.policy.to(device)
			#self.critic.to(device)
			
			self.action_noise = action_noise
		
			self.player_name = player_name

			self.training_stats

	def load(self, path):
		agent_state = torch.load(path, map_location=self.device)

		print(agent_state.keys())
		
		old_policy_n_tokens = agent_state[f'{self.player_name}_policy_state_dict']['vertex_embedding.weight'].shape[0]
		old_policy_n_positions = agent_state[f'{self.player_name}_policy_state_dict']['position_embedding.weight'].shape[0]
		old_policy_n_out = agent_state[f'{self.player_name}_policy_state_dict']['final_linear.bias'].shape[0]

		self.policy = SimpleDecoderTransformer( L=LAYERS, 
												H=HEADS, 
												d_e=EMBEDDING_DIM,
												d_mlp=MLP_DIM,
												n_tokens=old_policy_n_tokens,
												n_positions=old_policy_n_positions,
												n_out=old_policy_n_out ).to(self.device)
	
		self.training_stats = agent_state['training_stats']

		self.policy.load_state_dict(agent_state[f'{self.player_name}_policy_state_dict'])
		
		if self.policy_architecture_args is not None:
			self.policy.update_embedding_sizes(self.policy_architecture_args['n_tokens'],
										 	   self.policy_architecture_args['n_positions'],
											   self.policy_architecture_args['n_out'])

		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = self.policy_training_args['learning_rate'])

		if self.policy_architecture_args['n_tokens'] == old_policy_n_tokens and \
		   self.policy_architecture_args['n_positions'] == old_policy_n_positions and \
		   self.policy_architecture_args['n_out'] == old_policy_n_out:
			self.policy_optimizer.load_state_dict(agent_state[f'{self.player_name}_optimizer_state_dict'])
	
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
		#self.critic.train()
		
	def eval(self):
		self.policy.eval()
		#self.critic.eval()

		
class Deterministic():
	"""
	A dummy class that is used in place of a distribution over non-negative "noise vectors" of shape size
	"""
	def __init__(self, size, device):
		self.size = size
		self.zeros = torch.zeros(self.size).to(device)
		return
	def sample(self):
		return self.zeros

"""
def load_training_history(device):
	for path in [BUILDERPOLICYOPTPATH, FORBIDDERPOLICYOPTPATH]
	builder_state = torch.load(BUILDERPOLICYOPTPATH, map_location=device)
	forbidder_state = torch.load(FORBIDDERPOLICYOPTPATH, map_location=device)

	training_stats = builder_state['training_stats']
	
	print('builder_policy_state_dict : ')
	for name, val in builder_state['builder_policy_state_dict'].items():
		print(name, val.shape)
	print()

	print('forbidder_policy_state_dict : ')
	for name, val in forbidder_state['forbidder_policy_state_dict'].items():
		print(name, val.shape)
	print()

	builder_old_n_tokens = builder_state['builder_policy_state_dict']['vertex_embedding.weight'].shape[0]
	builder_old_n_positions = builder_state['builder_policy_state_dict']['position_embedding.weight'].shape[0]
	builder_old_n_out = builder_state['builder_policy_state_dict']['final_linear.bias'].shape[0]

	builder_policy = SimpleDecoderTransformer(L = LAYERS, 
											 H=HEADS, 
											 d_e=EMBEDDING_DIM,
											 d_mlp=MLP_DIM,
											 n_tokens=builder_old_n_tokens,
											 n_positions=builder_old_n_positions,
											 n_out=builder_old_n_out).to(device)

	forbidder_old_n_tokens = forbidder_state['forbidder_policy_state_dict']['vertex_embedding.weight'].shape[0]
	forbidder_old_n_positions = forbidder_state['forbidder_policy_state_dict']['position_embedding.weight'].shape[0]
	forbidder_old_n_out = forbidder_state['forbidder_policy_state_dict']['final_linear.bias'].shape[0]

	forbidder_policy = SimpleDecoderTransformer(L = LAYERS, 
											 H=HEADS, 
											 d_e=EMBEDDING_DIM,
											 d_mlp=MLP_DIM,
											 n_tokens=forbidder_old_n_tokens,
											 n_positions=forbidder_old_n_positions,
											 n_out=forbidder_old_n_out).to(device)

	builder_policy.load_state_dict(builder_state['builder_policy_state_dict'])
	print('loaded builder policy')
	forbidder_policy.load_state_dict(forbidder_state['forbidder_policy_state_dict'])
	print('loaded forbidder policy')

	assert len(training_stats) != 0
	print(len(training_stats))

	best_so_far = {"builder" : float('-inf'), 
				   "forbidder" : float('-inf')
				  }

	if not LOAD_PRETRAINED:
		best_so_far = {"builder" : max(batch_stats['average_builder_return'] for batch_stats, eval_stats in training_stats),
										"forbidder" : max(batch_stats['average_forbidder_return'] for batch_stats, eval_stats in training_stats)}
		print('best builder average return :', best_so_far['builder'])
		print('best forbidder average return :', best_so_far['forbidder'])
		print('second latest builder average return :', training_stats[len(training_stats) - 1][0]['average_builder_return'])
		print('second latest forbidder average return :', training_stats[len(training_stats) - 1][0]['average_forbidder_return'])
	
	for policy in [builder_policy, forbidder_policy]:
		policy.update_embedding_sizes(N_TOKENS, POSITIONS)

	print('builder_policy_state_dict (2): ')
	for name, val in builder_policy.named_parameters():
		print(name, val.shape)
	print()

	print('forbidder_policy_state_dict (2): ')
	for name, val in forbidder_policy.named_parameters():
		print(name, val.shape)
	print()
	
	print('building optimizers')
	builder_optimizer = torch.optim.Adam(builder_policy.parameters(), lr=LEARNING_RATE)
	forbidder_optimizer = torch.optim.Adam(forbidder_policy.parameters(), lr=LEARNING_RATE)
	if builder_old_n_tokens == N_TOKENS and builder_old_n_positions == POSITIONS and builder_old_n_out == N_OUT:
		builder_optimizer.load_state_dict(builder_state['builder_optimizer_state_dict'])
	if forbidder_old_n_tokens == N_TOKENS and forbidder_old_n_positions == POSITIONS and forbidder_old_n_out == N_OUT:
		forbidder_optimizer.load_state_dict(forbidder_state['forbidder_optimizer_state_dict'])
	print('loaded optimizers')

	return training_stats, best_so_far, builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer


def checkpoint(builder_policy, forbidder_policy, builder_optimizer, forbidder_optimizer, training_stats, batch_stats, best_so_far):
	torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BUILDERPOLICYOPTPATH
		)
	torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, FORBIDDERPOLICYOPTPATH
		)
	if best_so_far['builder'] < batch_stats['average_builder_return']:
		torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'builder_policy_state_dict' : builder_policy.state_dict(),
				'builder_optimizer_state_dict' : builder_optimizer.state_dict(),
			}, BESTBUILDERPOLICYOPTPATH
		)
		best_so_far['builder'] = batch_stats['average_builder_return']
	if best_so_far['forbidder'] < batch_stats['average_forbidder_return']:
		torch.save(
			{
				'vertex_vocabulary' : VERTEX_VOCABULARY,
				'training_stats' : training_stats,
				'forbidder_policy_state_dict' : forbidder_policy.state_dict(),
				'forbidder_optimizer_state_dict' : forbidder_optimizer.state_dict(),
			}, BESTFORBIDDERPOLICYOPTPATH
		)
		best_so_far['forbidder'] = batch_stats['average_forbidder_return']
"""