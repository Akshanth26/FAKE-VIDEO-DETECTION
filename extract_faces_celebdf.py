import os, cv2, numpy as np, torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ====== CONFIGURE PATHS ======
# Raw videos live here (as in your screenshot):
# data/archive/Celeb-real, data/archive/YouTube-real, data/archive/Celeb-synthesis
ARCHIVE_BASE = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\archive"

SRC_REAL_DIRS = [
    os.path.join(ARCHIVE_BASE, "Celeb-real"),
    os.path.join(ARCHIVE_BASE, "YouTube-real"),
]
SRC_FAKE_DIRS = [
    os.path.join(ARCHIVE_BASE, "Celeb-synthesis"),
]

# Processed output (npy sequences) will be written here:
OUT_BASE = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\CelebDFv2\processed"
DST_REAL = os.path.join(OUT_BASE, "real")
DST_FAKE = os.path.join(OUT_BASE, "fake")
os.makedirs(DST_REAL, exist_ok=True)
os.makedirs(DST_FAKE, exist_ok=True)

# ====== EXTRACTION SETTINGS ======
IMG_SIZE = 128
SAMPLE_EVERY = 1  # seconds between sampled frames
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=IMG_SIZE, margin=0, select_largest=True, post_process=False, device=device)

def process_video(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * SAMPLE_EVERY), 1)
    frames, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                frames.append(face.permute(1,2,0).byte().cpu().numpy())  # (128,128,3)
        idx += 1
    cap.release()
    if frames:
        np.save(out_path, np.stack(frames))

def run_many(src_dirs, dst_dir, label):
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            print(f"[skip] missing: {src_dir}")
            continue
        for fn in tqdm(sorted(os.listdir(src_dir)), desc=f"Processing {label} ({os.path.basename(src_dir)})"):
            if not fn.lower().endswith((".mp4",".mov",".avi",".mkv")):
                continue
            in_path = os.path.join(src_dir, fn)
            out_path = os.path.join(dst_dir, os.path.splitext(fn)[0] + ".npy")
            if os.path.exists(out_path):
                continue
            process_video(in_path, out_path)

# ====== RUN EXTRACTION ======
run_many(SRC_REAL_DIRS, DST_REAL, "real")
run_many(SRC_FAKE_DIRS, DST_FAKE, "fake")
print("Done CelebDFv2 face extraction.")
