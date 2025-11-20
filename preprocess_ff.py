import os
import cv2
import numpy as np
from mtcnn import MTCNN

# Path to your dataset folders
BASE_PATH = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\FF++"

# Output folders to save face crops
OUTPUT_BASE = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\processed_faces"

detector = MTCNN()

def extract_frames(video_path, max_frames=64, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

def crop_faces(frames):
    face_crops = []
    for frame in frames:
        detections = detector.detect_faces(frame)
        if detections:
            x, y, w, h = detections[0]['box']
            x, y = max(0, x), max(0, y)
            crop = frame[y:y+h, x:x+w]
            crop = cv2.resize(crop, (128, 128))
            face_crops.append(crop)
    return np.array(face_crops)

def process_video_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for video_file in os.listdir(input_folder):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing {video_path} ...")

        frames = extract_frames(video_path)
        faces = crop_faces(frames)

        if len(faces) == 0:
            print(f"No faces detected in {video_file}, skipping!")
            continue

        save_path = os.path.join(output_folder, video_file.replace('.mp4', '.npy'))
        np.save(save_path, faces)
        print(f"Saved cropped faces to {save_path}")

if __name__ == "__main__":
    for folder_type in ['fake', 'real']:
        input_folder = os.path.join(BASE_PATH, folder_type)
        output_folder = os.path.join(OUTPUT_BASE, folder_type)
        process_video_folder(input_folder, output_folder)
