import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """Simple LSTM autoencoder for 1D sequences."""

    def __init__(self, seq_len: int, n_features: int = 1, embedding_dim: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, _ = self.encoder(x)
        context = enc[:, -1, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec, _ = self.decoder(context)
        out = self.output_layer(dec)
        return out


def load_training_data(path: Path, seq_len: int) -> tuple[torch.Tensor, float, float]:
    """Load move_time sequences and normalise."""
    data = json.loads(Path(path).read_text())
    all_times: list[float] = []
    for game in data:
        all_times.extend(game["move_times"])
    arr = np.asarray(all_times, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std() if arr.std() > 0 else 1.0)
    norm = (arr - mean) / std
    samples = []
    for i in range(len(norm) - seq_len + 1):
        samples.append(norm[i : i + seq_len])
    dataset = torch.tensor(samples, dtype=torch.float32).unsqueeze(-1)
    return dataset, mean, std


def main() -> None:
    seq_len = 10
    dataset, mean, std = load_training_data(Path("test/data/input.json"), seq_len)
    loader = DataLoader(TensorDataset(dataset, dataset), batch_size=32, shuffle=True)
    model = LSTMAutoencoder(seq_len)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(5):
        for batch, _ in loader:
            optim.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optim.step()

    save_path = Path("models/lstm_autoencoder.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "seq_len": seq_len,
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
