import torch
import numpy as np
import pygame
from model import ConditionedFrameGen
from config import SMALL, LARGE, TEST

model_config = TEST

image_size = model_config["model"]["input_size"]
context = model_config["model"]["context"]
emb_dim = model_config["model"]["emb_dim"]

def get_action(keys):
    action = np.zeros(3, dtype=np.float32)
    if keys[pygame.K_a]:
        action[0] = 1.0
    if keys[pygame.K_d]:
        action[1] = 1.0
    if keys[pygame.K_SPACE]:
        action[2] = 1.0
    return action

initial_frame = "dataset/frames/frame_000000.jpg"

def infer(prev_frames, action, model):
    device = next(model.parameters()).device
    prev_frames = torch.tensor(prev_frames, dtype=torch.float32).unsqueeze(0).to(device)  # (1, context, C, H, W)
    action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(prev_frames, action)
    out_img = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return out_img

def main():
    pygame.init()
    screen = pygame.display.set_mode((image_size, image_size))
    pygame.display.set_caption('frame gen')
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = ConditionedFrameGen(img_channels=3, action_dim=3, emb_dim=emb_dim, input_size=image_size, context=context).to(device)
    model.load_state_dict(torch.load("framegen.pth", map_location=device))
    model.eval()

    import time
    from PIL import Image

    img = Image.open(initial_frame).resize((image_size, image_size)).convert("RGB")
    first_frame = np.array(img).astype(np.float32) / 255.0
    first_frame = first_frame.transpose(2, 0, 1)  # (C, H, W)

    prev_frames = [first_frame.copy() for _ in range(context)]  # list of (C, H, W)

    import os
    import psutil
    process = psutil.Process(os.getpid())
    last_log = time.time()
    frame_count = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = get_action(keys)
        # Stack context frames to shape (context, C, H, W)
        prev_frames_np = np.stack(prev_frames, axis=0)
        out_img = infer(prev_frames_np, action, model)
        # Update context buffer: drop oldest, append newest
        prev_frames.pop(0)
        prev_frames.append(out_img.transpose(2, 0, 1))

        frame_disp = np.clip(out_img * 255, 0, 255).astype(np.uint8)
        frame_disp = np.rot90(frame_disp, k=1)
        frame_disp = np.flipud(frame_disp)

        surf = pygame.surfarray.make_surface(frame_disp)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(10)  # 10 FPS

        frame_count += 1
        now = time.time()
        if now - last_log >= 1.0:
            fps = frame_count / (now - last_log)
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"FPS: {fps:.2f} | Memory: {mem_mb:.2f} MB")
            frame_count = 0
            last_log = now

    pygame.quit()

if __name__ == "__main__":
    main()