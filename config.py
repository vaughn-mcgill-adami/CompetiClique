#Game Parameters.
AVAILABLE_TOKEN = 0
FORBIDDEN_TOKEN = 1
PAD_TOKEN = 2
END_OBSERVATION_TOKEN = 3
END_GAME_DESCRIPTION_TOKEN = 4

VERTEX_VOCAB_STARTS_AT = 5

N_TOKENS = VERTEX_VOCABULARY = 10
POSITIONS = 2*N_TOKENS*(N_TOKENS-1)
N_TOKENS = N_TOKENS + 5
N_OUT = N_TOKENS

MIN_CLIQUE_SIZE = 3
MIN_EDGES_PER_BUILDER_TURN = 1
MIN_VERTICES_PER_FORBIDDER_TURN = 1

MAX_CLIQUE_SIZE = 3
MAX_EDGES_PER_BUILDER_TURN = 1
MAX_VERTICES_PER_FORBIDDER_TURN = 1

#Model Parameters.
LAYERS = 6
HEADS = 4
EMBEDDING_DIM = 64
MLP_DIM = 96

LOAD_SAVED_WEIGHTS = False

#Logging Parameters
BESTBUILDERPOLICYOPTPATH = "best_builder_policy_opt.pt"
BESTFORBIDDERPOLICYOPTPATH = "best_forbidder_policy_opt.pt"
BUILDERPOLICYOPTPATH = "builder_policy_opt.pt"
FORBIDDERPOLICYOPTPATH = "forbidder_policy_opt.pt"

SAVE_A_TRAJECTORY_PATH = "trajectory.pt"
SAVE_A_PRETRAIN_TRAJECTORY_PATH = "pretrain_trajectory.pt"

#Training Parameters
DEVICE = 'cpu'

EVAL_ONLY = False #not sure if this is even used anywhere.

NUM_BATCHES = 1000
BATCH_SIZE = 1000
LEARNING_RATE = 0.0005

#RL specific parameters
DISCOUNT_FACTOR = 0.9

#Pretraining specific parameters
PRETRAIN_BATCH_SIZE = 5000
PRETRAIN_LEARNING_RATE = 0.001
PRETRAIN_DISCOUNT_FACTOR = 1.0

WIDTH = 2
MAXLOOKAHEAD = 16

#Eval parameters
NUM_EVAL_SAMPLES = 200
