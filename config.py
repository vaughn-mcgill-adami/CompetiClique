#Game Parameters.
AVAILABLE_TOKEN = 0
FORBIDDEN_TOKEN = 1
PAD_TOKEN = 2
END_OBSERVATION_TOKEN = 3
END_GAME_DESCRIPTION_TOKEN = 4

VERTEX_VOCAB_STARTS_AT = 5

#OLD_N_TOKENS = OLD_VERTEX_VOCABULARY = 100
#OLD_POSITIONS = 2*OLD_N_TOKENS*(OLD_N_TOKENS - 1)
#OLD_N_TOKENS = OLD_N_TOKENS + 5
#OLD_N_OUT = OLD_N_TOKENS

N_TOKENS = VERTEX_VOCABULARY = 5
POSITIONS = 2*N_TOKENS*(N_TOKENS-1)
N_TOKENS = N_TOKENS + 5
N_OUT = N_TOKENS

MIN_CLIQUE_SIZE = 3
MIN_EDGES_PER_BUILDER_TURN = 1
MIN_VERTICES_PER_FORBIDDER_TURN = 1

MAX_CLIQUE_SIZE = 3
MAX_EDGES_PER_BUILDER_TURN = 2
MAX_VERTICES_PER_FORBIDDER_TURN = 2

#Model Parameters.
LAYERS = 6
HEADS = 4
EMBEDDING_DIM = 64
MLP_DIM = 96

#BATCH_NORM_MOMENTUM = None

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

EVAL_ONLY = False

NUM_BATCHES = 100
BATCH_SIZE = 1000
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.9

#Eval parameters
NUM_EVAL_SAMPLES = 200
