from torch.utils.data import DataLoader, TensorDataset
import torch

def build_dataloaders(args):
    x = torch.randn(1024, 128)
    y = torch.randint(0, 2, (1024,))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=getattr(args, "batch_size", 64), shuffle=True)
    return dl, dl
