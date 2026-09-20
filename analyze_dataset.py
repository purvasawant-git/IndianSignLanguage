import os
import cv2

dataset_path = 'dataset_ISL'
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()

classes = os.listdir(dataset_path)
stats = {}
for c in classes:
    class_path = os.path.join(dataset_path, c)
    if os.path.isdir(class_path):
        files = os.listdir(class_path)
        if len(files) > 0:
            sample_img_path = os.path.join(class_path, files[0])
            img = cv2.imread(sample_img_path)
            if img is not None:
                shape = img.shape
            else:
                shape = "Unknown (failed to load)"
        else:
            shape = "Empty"
        stats[c] = {"count": len(files), "shape": shape}

for c, data in stats.items():
    print(f"Class: {c}, Count: {data['count']}, Shape: {data['shape']}")
print(f"Total Classes: {len(stats)}")
