import argparse
import torch
from torch import nn
from torch.optim import AdamW
from .model import build_model
from .datasets import build_dataloaders

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total)

def train(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
    model = build_model(args).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader = build_dataloaders(args)

    best_state = None
    best_metric = float("-inf")
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        metric = validate(model, val_loader, device)
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_state if best_state is not None else model.state_dict(), args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--output", type=str, default="/app/model_store/model.pt")
    args = p.parse_args()
    train(args)
