import os
import sys
import argparse
import configparser
import logging
from logging.handlers import RotatingFileHandler
from src import train
import shutil

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
    
    # Get the list of valid video_ids from CSV

    # Iterate over folders in the input directory
    for _,row in df.iterrows():
        folder_name = str(row['video_id'])
        folder_path = os.path.join(input_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue  # Skip files

        video_id = str(row.get('label_id'))

        if video_id == 'nan':
            output_video_dir = os.path.join(output_dir, split)
        else : output_video_dir = os.path.join(output_dir, split,video_id)
        os.makedirs(output_video_dir, exist_ok=True)

        # Copy contents (not the folder itself)
        for item in os.listdir(folder_path):
            src_path = os.path.join(folder_path, item)
            new_name = f"{folder_name}_{item}"
            dst_item = os.path.join(output_video_dir, new_name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_item)

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
        input_dir = '20bn-jester-v1'
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