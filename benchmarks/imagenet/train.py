import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import make_model
from data import load_imagenet_1k

def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    print(f"Training on {device}")
    # train mode
    model.train()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # lr scheduler -- reduce learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

    for epoch in range(100):
        total_loss = 0.0
        for idx, batch in enumerate(train_loader):
            opt.zero_grad()
            x, y = batch["image"], batch["label"]
            x = x.to(device)
            y = y.to(device).reshape(-1)
            y = F.one_hot(y, num_classes=1000).float()
            with torch.autocast("cuda"):
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            if idx % 20 == 0:
                print(f"Epoch {epoch}, batch {idx} avg loss: {total_loss / (idx + 1)}")
            loss.backward()
            opt.step()
        scheduler.step(total_loss)

if __name__ == "__main__":
    model = make_model(out_classes=1000)
    print("loading data")
    dl = load_imagenet_1k()
    train(model, dl, torch.device("cuda:0"))
    torch.save(model, "model.pth")
