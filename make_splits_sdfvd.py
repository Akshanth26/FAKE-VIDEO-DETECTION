import os, json, random

root = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\SDFVD\processed"
out_json = r"C:\Users\Akshanth Chouhan\OneDrive\Documents\Fake_detection\data\SDFVD\splits\sdfvd_70_30.json"
os.makedirs(os.path.dirname(out_json), exist_ok=True)

splits = {"real":{"train":[],"test":[]}, "fake":{"train":[],"test":[]}}
for cls in ["real","fake"]:
    p = os.path.join(root, cls)
    files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".npy")]
    random.shuffle(files)
    n = int(0.7 * len(files))
    splits[cls]["train"] = files[:n]
    splits[cls]["test"]  = files[n:]

with open(out_json, "w") as f:
    json.dump(splits, f, indent=2)

print("Saved splits to:", out_json)
