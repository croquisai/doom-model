import torch
import numpy as np
from model import ConditionedFrameGen
import matplotlib.pyplot as plt
import time
import imageio
from config import SMALL, LARGE, TEST

model_config = TEST

image_size = model_config["model"]["input_size"]
context = model_config["model"]["context"]
emb_dim = model_config["model"]["emb_dim"]

def infer(prev_frame, action, model):
    device = next(model.parameters()).device
    prev_frame = torch.tensor(prev_frame, dtype=torch.float32).unsqueeze(0).to(device)
    action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(prev_frame, action)
    out_img = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return out_img

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = ConditionedFrameGen(img_channels=3, action_dim=3, emb_dim=emb_dim, input_size=image_size).to(device)
    model.load_state_dict(torch.load("checkpoints/ckfpt200.pth", map_location=device))
    model.eval()

    # Initial random frame
    prev_frame = np.random.rand(3, image_size, image_size)
    frames = []
    num_frames = 100
    for i in range(num_frames):
        action = np.random.randint(0, 2, size=(3,)).astype(np.float32)
        out_img = infer(prev_frame, action, model)
        frames.append((np.clip(out_img * 255, 0, 255)).astype(np.uint8))
        prev_frame = out_img.transpose(2, 0, 1)

    gif_path = "generated_frames.gif"
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"GIF saved to {gif_path}")

    mp4_path = "generated_frames.mp4"
    imageio.mimsave(mp4_path, frames, fps=10)
    print(f"MP4 saved to {mp4_path}")

    plt.imshow(frames[0])
    plt.title("First Frame of Generated GIF")
    plt.show()
