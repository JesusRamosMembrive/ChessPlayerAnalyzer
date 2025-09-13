from app.celery_app import celery_app
import torch
from app.ml.inference import get_model

@celery_app.task(name="app.ml_tasks.infer_model", queue="torch")
def infer_model(vector):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device=device)
    import torch as _t
    x = _t.tensor(vector, dtype=_t.float32, device=device).view(1, -1)
    with _t.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1).cpu().numpy().tolist()[0]
    return {"probs": probs}
