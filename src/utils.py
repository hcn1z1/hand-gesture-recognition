import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count

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
    pose_indices = [11, 12, 13, 14, 15, 16]
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
    return np.array(keypoints).reshape(48, 3)


def define_connections():
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
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in range(1, sequence_length + 1):
            frame_path = src_dir / f"{i:05d}.jpg"
            if not frame_path.exists():
                break
            frame = cv2.imread(str(frame_path))
            _, results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = extract_keypoints(results)
            frames.append(keypoints)

    if len(frames) < sequence_length:
        return

    frames = np.stack(frames)
    joint_stream = frames
    bone_pose = np.stack([frames[:, b] - frames[:, a] for a, b in pose_conn], axis=1)
    bone_left = np.stack([frames[:, b + 6] - frames[:, a + 6] for a, b in hand_conn], axis=1)
    bone_right = np.stack([frames[:, b + 27] - frames[:, a + 27] for a, b in hand_conn], axis=1)
    bone_stream = np.concatenate([bone_pose, bone_left, bone_right], axis=1)
    joint_motion = joint_stream[1:] - joint_stream[:-1]
    bone_motion = bone_stream[1:] - bone_stream[:-1]

    split_dir = Path(output_dir) / split / str(label)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{video_id}_joint.npy", joint_stream)
    np.save(split_dir / f"{video_id}_bone.npy", bone_stream)
    np.save(split_dir / f"{video_id}_joint_motion.npy", joint_motion)
    np.save(split_dir / f"{video_id}_bone_motion.npy", bone_motion)

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

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks), desc=f"Processing {split}"))


if __name__ == "__main__":
    input_dir = "20bn-jester-v1"
    output_dir = "data/jester_processed"
    for split, csv in [("train", "annotations/train.csv"), ("val", "annotations/validation.csv")]:
        preprocess_jester_multiproc(input_dir, output_dir, csv, split)
