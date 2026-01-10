import os, numpy as np

root = r"Hashmani's Dataset/FusionCache_1024x576"
for split in ["train", "val", "test"]:
    d = os.path.join(root, split)
    files = [f for f in os.listdir(d) if f.endswith(".npy")]
    print(split, "n_files=", len(files), "example=", files[0] if files else None)
    if files:
        obj = np.load(os.path.join(d, files[0]), allow_pickle=True).item()
        print("  input shape:", obj["input"].shape, "label:", obj["label"])
