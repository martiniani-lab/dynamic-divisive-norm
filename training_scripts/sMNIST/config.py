import argparse
import default_config

def get_config():
    parser = argparse.ArgumentParser(description="Training Configuration")
    
    parser.add_argument('--MODEL_NAME', type=str, default=default_config.MODEL_NAME, help='Model name')
    parser.add_argument('--FOLDER_NAME', type=str, default=default_config.FOLDER_NAME, help='Folder name')
    parser.add_argument('--PERMUTED', action='store_true', help='Permute the input')
    parser.add_argument('--CHECKPOINT', action='store_true', help='Start from a checkpoint')
    parser.add_argument('--VERSION', type=int, default=default_config.VERSION, help='Checkpoint version')
    parser.add_argument('--SEED', type=int, default=default_config.SEED, help='Permuted seed')
    parser.add_argument('--RESIZE', type=float, default=default_config.RESIZE, help='Fraction of the original input size')

    # Training hyperparameters
    parser.add_argument('--INPUT_SIZE', type=int, default=default_config.INPUT_SIZE, help='Input size')
    parser.add_argument('--SEQUENCE_LENGTH', type=int, default=default_config.SEQUENCE_LENGTH, help='Sequence length')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=default_config.HIDDEN_SIZE, help='Hidden size')
    parser.add_argument('--NUM_CLASSES', type=int, default=default_config.NUM_CLASSES, help='Number of classes')
    parser.add_argument('--LEARNING_RATE', type=float, default=default_config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, default=default_config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--NUM_EPOCHS', type=int, default=default_config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--SCHEDULER_CHANGE_STEP', type=int, default=default_config.SCHEDULER_CHANGE_STEP, help='Scheduler change step')
    parser.add_argument('--SCHEDULER_GAMMA', type=float, default=default_config.SCHEDULER_GAMMA, help='Scheduler gamma')
    parser.add_argument('--CHECKPOINT_EVERY_N_EPOCH', type=int, default=default_config.CHECKPOINT_EVERY_N_EPOCH, help='Checkpoint every n epoch')

    # Dataset
    parser.add_argument('--DATA_DIR', type=str, default=default_config.DATA_DIR, help='Data directory')
    parser.add_argument('--NUM_WORKERS', type=int, default=default_config.NUM_WORKERS, help='Number of workers')

    # Compute related
    parser.add_argument('--ACCELERATOR', type=str, default=default_config.ACCELERATOR, help='Accelerator')
    parser.add_argument('--DEVICES', nargs='+', default=default_config.DEVICES, help='Devices')
    parser.add_argument('--PRECISION', type=int, default=default_config.PRECISION, help='Precision')

    args, unknown = parser.parse_known_args()

    return args, unknown

if __name__ == "__main__":
    config = get_config()
    print(config)
