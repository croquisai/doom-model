import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

class ViZDoomFrameGenDataset(Dataset):
    def __init__(self, frames_dir, actions_csv, img_size=32):
        self.frames_dir = frames_dir
        self.data = pd.read_csv(actions_csv)
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def _load_frame(self, frame_name):
        frame_name = frame_name.replace(".png", ".jpg")  # for some reason
        path = os.path.join(self.frames_dir, frame_name)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        return img

    def __getitem__(self, idx):
        # Current frame and action
        row = self.data.iloc[idx]
        curr_frame = self._load_frame(row['frame'])
        action = row[1:].values.astype(np.float32)

        # previous frame
        if idx == 0:
            prev_frame = np.zeros_like(curr_frame)
        else:
            prev_row = self.data.iloc[idx - 1]
            prev_frame = self._load_frame(prev_row['frame'])

        return (
            torch.tensor(prev_frame, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(curr_frame, dtype=torch.float32),
        )
