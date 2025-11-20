import cv2
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from src.models import get_feature_extractor
import numpy as np

# Paths - UPDATED VIDEO PATH
video_path = r"C:\Users\Akshanth Chouhan\Downloads\invideo-ai-1080 Watch the Face Morph Mid-Sentence (10s T 2025-10-17.mp4"
output_folder = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\extracted_frames_morph"
ckpt_path = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\checkpoints\best_resnet18_1760334297.pth"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Step 1: Extract frames
print("Extracting frames from face morph video...")
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_frames = []

while frame_count < 24:  # Extract 24 frames
    ret, frame = cap.read()
    if not ret:
        print(f"Could only extract {frame_count} frames")
        break
    
    # Save frame
    frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    saved_frames.append(frame_path)
    frame_count += 1

cap.release()
print(f"✓ Extracted {frame_count} frames to {output_folder}")

if frame_count == 0:
    print("ERROR: Could not extract any frames. Video codec not supported.")
    exit()

# Step 2: Load frames and run inference
print("\nRunning inference on face morph video...")

# Load frames
frames = []
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for frame_path in saved_frames:
    img = Image.open(frame_path).convert('RGB')
    frames.append(transform(img))

# If we have fewer than 24 frames, repeat
while len(frames) < 24:
    frames.extend(frames[:min(24 - len(frames), len(frames))])

input_tensor = torch.stack(frames[:24]).unsqueeze(0)  # (1, 24, 3, 224, 224)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(ckpt_path, map_location=device)

cnn_output = get_feature_extractor('resnet18')
if isinstance(cnn_output, tuple):
    cnn = cnn_output[0]
    feature_dim = cnn_output[1]
else:
    cnn = cnn_output
    feature_dim = 512

class FullModel(nn.Module):
    def __init__(self, cnn, hidden_size=128, num_classes=2):
        super().__init__()
        self.cnn = cnn
        self.rnn = nn.GRU(feature_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        features = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t])
            features.append(feat)
        features = torch.stack(features, dim=1)
        _, hidden = self.rnn(features)
        out = self.fc(hidden.squeeze(0))
        return out

model = FullModel(cnn, hidden_size=128, num_classes=2).to(device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Run inference
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    label = np.argmax(probs)
    confidence = probs[0, label]

label_name = "Fake" if label == 1 else "Real"

# Print results
print("\n" + "="*60)
print("INFERENCE RESULTS - FACE MORPH VIDEO")
print("="*60)
print(f"Video: Face Morph Mid-Sentence (AI Generated)")
print(f"Frames Extracted: {frame_count}")
print(f"Prediction: {label_name}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
print(f"Real Probability: {probs[0, 0]:.4f} ({probs[0, 0]*100:.2f}%)")
print(f"Fake Probability: {probs[0, 1]:.4f} ({probs[0, 1]*100:.2f}%)")
print("="*60)

# Analysis
if label_name == "Fake":
    print("\n✓ SUCCESS: Model correctly detected the face morph as FAKE!")
    print("The temporal inconsistencies from face morphing were detected.")
else:
    print("\n✗ Model classified as REAL despite face morphing present.")
    print("The morphing may be too smooth for the model to detect.")
