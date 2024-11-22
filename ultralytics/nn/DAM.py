# A Dynamic Context-aware Aggregation Strategy (DCAS) for Small Object Detection, GPL-3.0 license
"""
Dynamic Aggregation Mechanism (DAM)
"""

import os
import cv2
import numpy as np
import math
from tqdm import tqdm

class YOLODataSet:
    def __init__(self, yolo_root, image_folder="images/train", label_folder="labels/train"):
        self.root = yolo_root
        self.image_root = os.path.join(self.root, image_folder)
        self.label_root = os.path.join(self.root, label_folder)
        self.image_files = [os.path.join(self.image_root, filename)
                            for filename in os.listdir(self.image_root) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def parse_yolo_label(self, label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        return [(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                for line in lines if (parts := line.strip().split()) and len(parts) == 5]

    def get_info(self):
        im_wh_list, boxes_wh_list = [], []

        for image_path in tqdm(self.image_files, desc="Reading data info"):
            label_path = os.path.join(self.label_root, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
            if not os.path.exists(label_path):
                continue

            image = cv2.imread(image_path)
            im_height, im_width, _ = image.shape
            label_data = self.parse_yolo_label(label_path)

            wh = [(width, height) for _, _, _, width, height in label_data if 0 < width < 1 and 0 < height < 1]
            if wh:
                im_wh_list.append([im_width, im_height])
                boxes_wh_list.append(wh)

        return im_wh_list, boxes_wh_list

def calculate_label_counts(min_wh, bin_size=8):
    num_bins = 640 // bin_size
    bin_counts, _ = np.histogram(min_wh, bins=num_bins, range=(0, 640))
    return bin_counts

def label_counts(img_size=640):
    dataset = YOLODataSet(yolo_root='/home/datasets/VisDrone') # dataset root dir
    im_wh, boxes_wh = dataset.get_info()

    im_wh = np.array(im_wh, dtype=np.float32)
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True)
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])
    
    small_objects_count = (wh0 < 1.0).any(1).sum()
    if small_objects_count:
        print(f'WARNING: Extremely small objects found. {small_objects_count} of {len(wh0)} labels are < 1 pixels in size.')
    wh = wh0[(wh0 >= 1.0).any(1)]
    
    min_wh = np.minimum.reduce(wh, axis=1)
    label_counts = calculate_label_counts(min_wh)

    total_anchors = len(min_wh)
    results = [(i * 8, (i + 1) * 8, count, count / total_anchors) for i, count in enumerate(label_counts) if count >= 0]

    print(f'Anchors in range {0}-{8}: {results[0][2]}, Proportion: {results[0][3]}')
    print(f'Anchors in range {8}-{16}: {results[1][2]}, Proportion: {results[1][3]}')
    print(f'Anchors in range {16}-{24}: {results[2][2]}, Proportion: {results[2][3]}')
    print(f'Anchors in range {24}-{32}: {results[3][2]}, Proportion: {results[3][3]}')
    
    return results[0][3]

def num_DAM():
    def function(J):
        R = 0.15
        k = 30
        return math.floor(k * (J - R)) + 3

    J = label_counts()
    N = function(J)

    num_layers = max(min(N, 20), 3)
    print(f'Number of max-pooling layers in the DAM module: {num_layers}')
    return num_layers + 1
