import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConditionedFrameGen
from dataset import ViZDoomFrameGenDataset
import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = ConditionedFrameGen(img_channels=3, action_dim=3, emb_dim=512, input_size=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    dataset = ViZDoomFrameGenDataset(
        frames_dir="dataset/frames",
        actions_csv="dataset/actions.csv",
        img_size=512,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    global_step = 0
    for epoch in range(25):
        total_loss = 0
        for prev_frame, action, curr_frame in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}"):
            prev_frame = prev_frame.to(device)
            action = action.to(device)
            curr_frame = curr_frame.to(device)

            pred = model(prev_frame, action)
            loss = criterion(pred, curr_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * prev_frame.size(0)

            if global_step % 100 == 0:
                torch.save(model.state_dict(), f"checkpoints/ckpt{global_step}.pth")
            global_step += 1

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "framegen.pth")

if __name__ == "__main__":
    train()
