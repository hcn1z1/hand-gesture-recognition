import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import logging

# Suppress logs
logging.getLogger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Action labels
actions = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]
label2id = {action: idx for idx, action in enumerate(actions)}

def extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """Extract keypoints from pose and hand landmarks."""
    keypoints = []
    pose_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    if pose_landmarks:
        for idx in pose_indices:
            lm = pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 3 * len(pose_indices))

    for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 3 * 21)
    return np.array(keypoints).reshape(48, 3)

def define_connections():
    """Define connections for bones in pose and hands."""
    pose_connections = [(3, 0), (3, 4), (4, 5), (0, 1), (1, 2)]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    return pose_connections, hand_connections

def process_video(args):
    """Process a single video and extract keypoints."""
    row, input_dir, output_dir, sequence_length, split = args
    video_id = str(row['video_id'])
    label_name = row['label']
    if label_name not in label2id:
        return
    label = label2id[label_name]
    src_dir = Path(input_dir) / video_id
    if not src_dir.is_dir():
        return

    pose_conn, hand_conn = define_connections()
    frames = []

    # Use context managers for resource management
    with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for i in range(1, sequence_length + 1):
            frame_path = src_dir / f"{i:05d}.jpg"
            if not frame_path.exists():
                break
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with Pose model
            pose_results = pose.process(frame_rgb)
            # Process with Hands model
            hands_results = hands.process(frame_rgb)

            # Extract landmarks
            pose_landmarks = pose_results.pose_landmarks if pose_results else None
            left_hand_landmarks = None
            right_hand_landmarks = None
            if hands_results.multi_hand_landmarks:
                for landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        left_hand_landmarks = landmarks
                    elif handedness.classification[0].label == 'Right':
                        right_hand_landmarks = landmarks

            keypoints = extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks)
            frames.append(keypoints)

    if len(frames) < sequence_length:
        return

    # Compute streams and motions
    frames = np.stack(frames)
    joint_stream = frames
    bone_pose = np.stack([frames[:, b] - frames[:, a] for a, b in pose_conn], axis=1)
    bone_left = np.stack([frames[:, b + 6] - frames[:, a + 6] for a, b in hand_conn], axis=1)
    bone_right = np.stack([frames[:, b + 27] - frames[:, a + 27] for a, b in hand_conn], axis=1)
    bone_stream = np.concatenate([bone_pose, bone_left, bone_right], axis=1)
    joint_motion = joint_stream[1:] - joint_stream[:-1]
    bone_motion = bone_stream[1:] - bone_stream[:-1]

    # Save results
    split_dir = Path(output_dir) / split / str(label)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{video_id}_joint.npy", joint_stream)
    np.save(split_dir / f"{video_id}_bone.npy", bone_stream)
    np.save(split_dir / f"{video_id}_joint_motion.npy", joint_motion)
    np.save(split_dir / f"{video_id}_bone_motion.npy", bone_motion)

    # Save original frames
    frame_dir = split_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_path in enumerate(sorted(src_dir.glob("*.jpg"))):
        img = cv2.imread(str(frame_path))
        cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), img)


def preprocess_jester_multiproc(input_dir, output_dir, csv_path, split, sequence_length=37):
    df = pd.read_csv(csv_path)
    df = df[df['label'].isin(actions)]
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (row, input_dir, output_dir, sequence_length, split)
        for _, row in df.iterrows()
    ]

    with Pool(processes=32) as pool:
        for _ in pool.imap_unordered(process_video, tasks):
            pass


if __name__ == "__main__":
    input_dir = "20bn-jester-v1"
    output_dir = "data/jester_processed"
    for split, csv in [("train", "annotations/train.csv"), ("val", "annotations/validation.csv")]:
        preprocess_jester_multiproc(input_dir, output_dir, csv, split)
