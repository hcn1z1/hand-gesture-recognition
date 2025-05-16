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


import os
import pandas as pd
from PIL import Image

def preprocess_split(split, input_dir, output_dir, csv_file):
    df = pd.read_csv(csv_file)
    has_labels = 'label_id' in df.columns and not df['label_id'].isnull().all()

    for _, row in df.iterrows():
        video_id = str(row['id'])
        frames = row['frames']
        middle_frame = (frames + 1) // 2
        frame_path = os.path.join(input_dir, video_id, f'{middle_frame:05d}.jpg')

        try:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue

        if has_labels:
            label_id = row['label_id']
            save_dir = os.path.join(output_dir, split, str(label_id))
        else:
            save_dir = os.path.join(output_dir, split)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{video_id}.jpg')
        img.save(save_path)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument('mode', choices=['train', 'test','preprocess'], help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    # Extract hyperparameters
    batch_size = int(training_config['batch_size'])
    num_epochs = int(training_config['num_epochs'])
    learning_rate = float(training_config['learning_rate'])

    if args.mode == 'train':
        model = train(num_epochs, batch_size, learning_rate)
    elif args.mode == 'test':
        pass
    elif args.mode == 'preprocess':
        input_dir = 'data'
        output_dir = 'data/jester'
        splits = ['train', 'validation', 'test']

        for split in splits:
            csv_file = f'annotations/{split}.csv'
            if not os.path.exists(csv_file):
                print(f"CSV file {csv_file} not found")
                continue
            preprocess_split(split, input_dir, output_dir, csv_file)

if __name__ == "__main__":
    main()