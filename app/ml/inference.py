import torch
from .model import build_model

_model = None

def get_model(device="cpu", weights_path="/app/model_store/model.pt"):
    global _model
    if _model is None:
        m = build_model({})
        state = torch.load(weights_path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _model = m.to(device)
    return _model
