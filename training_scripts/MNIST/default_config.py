MODEL_NAME = "MNIST"
FOLDER_NAME = "../../tb_logs"
VERSION = 0
PERMUTED = False
SEED = 73
RESIZE = 1.0

# Training hyperparameters
# Take the image dimensions and number of channels from the dataloader
INPUT_SIZE = 28 * 28
FF_INPUT_SIZE = 40
HIDDEN_SIZES = [128, 64]

NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 1
SCHEDULER_CHANGE_STEP = 100
SCHEDULER_GAMMA = 0.8
CHECKPOINT_EVERY_N_EPOCH = 1

# Dataset
DATA_DIR = "../../data/mnist"
NUM_WORKERS = 2

# Compute related
ACCELERATOR = "gpu"
DEVICES = "auto"
PRECISION = 32
