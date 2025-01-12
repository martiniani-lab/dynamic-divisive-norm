MODEL_NAME = "sMNIST"
FOLDER_NAME = "/vast/sr6364/dynamic-divisive-norm/tb_logs"
VERSION = 0
PERMUTED = False
SEED = 73
RESIZE = 1.0

# Training hyperparameters
INPUT_SIZE = 1
SEQUENCE_LENGTH = 784
HIDDEN_SIZE = 128

# # ORGaNICs cell parameters
# Wr_identity = False
# learn_tau = True
# dt_tau_max_y = 0.01
# dt_tau_max_a = 0.01
# dt_tau_max_b = 0.1

NUM_CLASSES = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 256
NUM_EPOCHS = 1
SCHEDULER_CHANGE_STEP = 20
SCHEDULER_GAMMA = 0.7
CHECKPOINT_EVERY_N_EPOCH = 1

# Dataset
DATA_DIR = "/home/sr6364/python_scripts/dynamic-divisive-norm/data/mnist"
NUM_WORKERS = 2

# Compute related
ACCELERATOR = "gpu"
DEVICES = "auto"
PRECISION = 32
