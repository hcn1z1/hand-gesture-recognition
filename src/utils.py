import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd


mp_holistic = mp.solutions.holistic


actions = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]
label2id = {action: idx for idx, action in enumerate(actions)}

def extract_keypoints(results):
    keypoints = []
    # Pose landmarks (indices: 11, 12, 13, 14, 15, 16)
    pose_indices = [11, 12, 13, 14, 15, 16]  # Left/right shoulder, elbow, wrist
    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 3 * len(pose_indices))

    # Hand landmarks (21 per hand)
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 3 * 21)
    return np.array(keypoints).reshape(48, 3)  # 6 pose + 42 hand = 48 points

def define_connections():
    # Pose connections
    pose_connections = [(3, 0), (3, 4), (4, 5), (0, 1), (1, 2)]  # Adjusted indices
    # Hand connections (MediaPipe’s predefined connections)
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    return pose_connections, hand_connections

def preprocess_jester(input_dir, output_dir, csv_path, split, sequence_length=37):
    df = pd.read_csv(csv_path)
    df = df[df['label'].isin(actions)]
    os.makedirs(output_dir, exist_ok=True)

    pose_conn, hand_conn = define_connections()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {split}"):
            video_id = str(row['video_id'])
            label = label2id[row['label']]
            src_dir = Path(input_dir) / video_id
            if not src_dir.is_dir():
                continue

            frames = []
            for i in range(1, sequence_length + 1):
                frame_path = src_dir / f"{i:05d}.jpg"
                if not frame_path.exists():
                    break
                frame = cv2.imread(str(frame_path))
                _, results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                keypoints = extract_keypoints(results)
                frames.append(keypoints)

            if len(frames) < sequence_length:
                continue  # Skip incomplete sequences

            frames = np.stack(frames)  # [T, 48, 3]
            joint_stream = frames
            # Bone stream
            bone_pose = np.stack([frames[:, b] - frames[:, a] for a, b in pose_conn], axis=1)
            bone_left = np.stack([frames[:, b + 6] - frames[:, a + 6] for a, b in hand_conn], axis=1)
            bone_right = np.stack([frames[:, b + 27] - frames[:, a + 27] for a, b in hand_conn], axis=1)
            bone_stream = np.concatenate([bone_pose, bone_left, bone_right], axis=1)
            # Motion streams
            joint_motion = joint_stream[1:] - joint_stream[:-1]
            bone_motion = bone_stream[1:] - bone_stream[:-1]

            split_dir = Path(output_dir) / split / str(label)
            os.makedirs(split_dir, exist_ok=True)
            np.save(split_dir / f"{video_id}_joint.npy", joint_stream)
            np.save(split_dir / f"{video_id}_bone.npy", bone_stream)
            np.save(split_dir / f"{video_id}_joint_motion.npy", joint_motion)
            np.save(split_dir / f"{video_id}_bone_motion.npy", bone_motion)
            # Copy original frames
            frame_dir = split_dir / video_id
            os.makedirs(frame_dir, exist_ok=True)
            for i, frame_path in enumerate(sorted(src_dir.glob("*.jpg"))):
                cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), cv2.imread(str(frame_path)))

# Update main() to include preprocessing
input_dir = "20bn-jester-v1"
output_dir = "data/jester_processed"
for split, csv in [("train", "annotations/train.csv"), ("val", "annotations/validation.csv")]:
    preprocess_jester(input_dir, output_dir, csv, split)