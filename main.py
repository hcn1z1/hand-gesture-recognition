import os
import sys
import argparse
import configparser
import logging
from logging.handlers import RotatingFileHandler
from src import train

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Extract configuration values
device_config = config['device']
model_config = config['model']
training_config = config['training']
dataset_config = config['dataset']
logging_config = config['logging']

# Set up device
use_cuda = config.getboolean('device', 'cuda')

# Set up logging
log_dir = logging_config['log_dir']
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Add console handler for stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    # Extract hyperparameters
    batch_size = int(training_config['batch_size'])
    num_epochs = int(training_config['num_epochs'])
    learning_rate = float(training_config['learning_rate'])

    if args.mode == 'train':
        model = train(num_epochs, batch_size, learning_rate)
    elif args.mode == 'test':
        pass

if __name__ == "__main__":
    main()