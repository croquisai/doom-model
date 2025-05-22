import os
import random
import numpy as np
import cv2
from vizdoom import DoomGame, Mode
import csv
import threading
from queue import Queue
import random
from tqdm import tqdm

SCENARIO_PATH = "doom.wad"
OUTPUT_DIR = "dataset"
N_EPISODES = 500
STEPS_PER_EPISODE = 50
SAVE_QUEUE_SIZE = 100
NUM_WORKERS = 4

def save_frame_worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        frame, frame_path = item
        cv2.imwrite(frame_path, frame)
        q.task_done()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)

    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path(SCENARIO_PATH)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_doom_skill(4)
    game.init()

    buttons = game.get_available_buttons()
    n_buttons = len(buttons)

    frame_queue = Queue(maxsize=SAVE_QUEUE_SIZE)
    actions_list = []
    frame_idx = 0

    workers = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=save_frame_worker, args=(frame_queue,))
        t.daemon = True
        t.start()
        workers.append(t)

    try:
        for ep in tqdm(range(N_EPISODES)):
            game.new_episode()
            for step in range(STEPS_PER_EPISODE):
                action = [random.choice([0, 1]) for _ in range(n_buttons)]
                reward = game.make_action(action)
                state = game.get_state()
                if state is None:
                    break
                if frame_idx % random.randint(1, 5) == 0:
                    frame = state.screen_buffer
                    frame = np.transpose(frame, (1, 2, 0))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_path = os.path.join(OUTPUT_DIR, "frames", f"frame_{frame_idx:06d}.jpg")
                    frame_queue.put((frame_bgr, frame_path))
                    actions_list.append({
                        "frame": f"frame_{frame_idx:06d}.png",
                        "action": action
                    })
                frame_idx += 1

        frame_queue.join()
        for _ in workers:
            frame_queue.put(None)
        for t in workers:
            t.join()

        with open(os.path.join(OUTPUT_DIR, "actions.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame"] + [btn.name for btn in buttons])
            for entry in actions_list:
                row = [entry["frame"]] + [str(x) for x in entry["action"]]
                writer.writerow(row)

    finally:
        game.close()

    print(f"Dataset generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()