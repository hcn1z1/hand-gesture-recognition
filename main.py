import os
import sys
import argparse
import configparser
import logging
from logging.handlers import RotatingFileHandler
from src import train, train_3
import shutil
from multiprocessing import Pool
from functools import partial
import pandas as pd
from src.utils import  preprocess_jester_multiproc
from PIL import Image

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


def create_output_directories(df, output_dir, split):
    """Pre-create all output directories."""
    output_dirs = set()
    for row in df.itertuples():
        video_id = str(row.label_id) if hasattr(row, 'label_id') else 'nan'
        output_video_dir = os.path.join(output_dir, split) if video_id == 'nan' else os.path.join(output_dir, split, video_id)
        output_dirs.add(output_video_dir)
    
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created {len(output_dirs)} output directories for split {split}")

def process_image(image_path, output_path):
    """Process a .jpg frame with MediaPipe (pose estimation)."""
    try:
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        ) as pose:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return
            
            # Downscale image
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(image_rgb)
            
            # Save landmarks as JSON
            if results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                json_path = f"{output_path}.json"
                with open(json_path, 'w') as f:
                    json.dump(landmarks, f)
            
            # Save processed image
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            cv2.imwrite(output_path, image)
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")

def process_chunk(chunk, input_dir, output_dir, split):
    """Process a chunk of DataFrame rows."""
    for _, row in chunk.iterrows():
        folder_name = str(row['video_id'])
        folder_path = os.path.join(input_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            logger.warning(f"Skipping non-directory: {folder_path}")
            continue

        video_id = str(row.get('label_id', 'nan'))
        output_video_dir = os.path.join(output_dir, split) if video_id == 'nan' else os.path.join(output_dir, split, video_id)

        try:
            for item in os.listdir(folder_path):
                src_path = os.path.join(folder_path, item)
                new_name = f"{folder_name}_{item}"
                dst_item = os.path.join(output_video_dir, new_name)
                
                if os.path.isfile(src_path) and item.lower().endswith('.jpg'):
                    process_image(src_path, dst_item)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_item)
        except Exception as e:
            logger.error(f"Error processing {folder_path}: {e}")

def preprocess_split(split, input_dir, output_dir, csv_file, num_processes=8):
    """Preprocess a dataset split using multiprocessing."""
    df = pd.read_csv(csv_file)
    
    # Pre-create output directories
    create_output_directories(df, output_dir, split)
    
    chunk_size = max(1, len(df) // (num_processes * 4))
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    logger.info(f"Processing {len(df)} rows in {len(chunks)} chunks with {num_processes} processes")
    
    process_func = partial(process_chunk, input_dir=input_dir, output_dir=output_dir, split=split)
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_func, chunks)
    
    logger.info(f"Completed preprocessing for split {split}")
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument('mode', choices=['train', 'test','preprocess','train_3'], help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    # Extract hyperparameters
    batch_size = int(training_config['batch_size'])
    num_epochs = int(training_config['num_epochs'])
    learning_rate = float(training_config['learning_rate'])

    if args.mode == 'train':
        model = train(num_epochs, batch_size, learning_rate)

    if args.mode == 'train_3':
        model = train_3(num_epochs, batch_size, learning_rate)

    elif args.mode == 'test':
        pass
    elif args.mode == 'preprocess':
        input_dir = "20bn-jester-v1"
        output_dir = "data/jester_processed"
        for split, csv in [("train", "annotations/train.csv"), ("val", "annotations/validation.csv")]:
            preprocess_jester_multiproc(input_dir, output_dir, csv, split)

if __name__ == "__main__":
    main()