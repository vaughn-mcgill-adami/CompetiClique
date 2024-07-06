#Game Parameters.
from torch import cuda

AVAILABLE_TOKEN = 0
FORBIDDEN_TOKEN = 1
PAD_TOKEN = 2
END_OBSERVATION_TOKEN = 3
END_GAME_DESCRIPTION_TOKEN = 4

VERTEX_VOCAB_STARTS_AT = 5

N_TOKENS = VERTEX_VOCABULARY = 10
POSITIONS = 2*N_TOKENS*(N_TOKENS-1)
N_TOKENS = N_TOKENS + VERTEX_VOCAB_STARTS_AT
#N_OUT = N_TOKENS

MIN_CLIQUE_SIZE = 3
MIN_EDGES_PER_BUILDER_TURN = 1
MIN_VERTICES_PER_FORBIDDER_TURN = 1

MAX_CLIQUE_SIZE = 4
MAX_EDGES_PER_BUILDER_TURN = 5
MAX_VERTICES_PER_FORBIDDER_TURN = 3

#Model Parameters.
LAYERS = 10
HEADS = 6
EMBEDDING_DIM = 96
MLP_DIM = 128

POLICY_ARCH_ARGS = {"L" : LAYERS, 
                    "H" : HEADS, 
                    "d_e" : EMBEDDING_DIM, 
                    "d_mlp" : MLP_DIM, 
                    "n_tokens" : N_TOKENS, 
                    "n_positions" : POSITIONS,
                    "n_out" : N_TOKENS
                }

CRITIC_ARCH_ARGS = {"L" : LAYERS, 
                    "H" : HEADS, 
                    "d_e" : EMBEDDING_DIM, 
                    "d_mlp" : MLP_DIM, 
                    "n_tokens" : N_TOKENS, 
                    "n_positions" : POSITIONS,
                    "n_out" : 1
                }

#Logging Parameters
#BESTBUILDERPOLICYOPTPATH = #"best_builder_policy_opt.pt"
#BESTFORBIDDERPOLICYOPTPATH = #"best_forbidder_policy_opt.pt"
BUILDERLOADPATH = "models/L10_0_builder_agent_c4_m5_n3.pt"
FORBIDDERLOADPATH = "models/L10_0_forbidder_agent_c4_m5_n3.pt"

BUILDERSAVEPATH = "models/L10_0_builder_agent_c4_m5_n3.pt"
FORBIDDERSAVEPATH = "models/L10_0_forbidder_agent_c4_m5_n3.pt"

#SAVE_A_TRAJECTORY_PATH = "trajectory.pt"
#SAVE_A_PRETRAIN_TRAJECTORY_PATH = "pretrain_trajectory.pt"

#Training Parameters
DEVICE = 'cuda:0' if cuda.is_available() else 'cpu'

EVAL_ONLY = False #not sure if this is even used anywhere.

NUM_BATCHES = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.000128

#RL specific parameters
DISCOUNT_FACTOR = 0.9

TRAINING_PARAMS = {"learning_rate": LEARNING_RATE}

#Pretraining specific parameters
PRETRAIN_BATCH_SIZE = BATCH_SIZE
WIDTH = 2
MAXLOOKAHEAD = 16

#Eval parameters
NUM_EVAL_SAMPLES = 2
