import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from linkpred.metrics import evaluate


class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=32, model_type="gcn", dropout=0.3):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "model_type": model_type,
            "dropout": dropout,
        }
        self.dropout = dropout
        if model_type == "gcn":
            self.conv1, self.conv2 = GCNConv(in_channels, hidden_channels), GCNConv(hidden_channels, out_channels)
        elif model_type == "sage":
            self.conv1, self.conv2 = SAGEConv(in_channels, hidden_channels), SAGEConv(hidden_channels, out_channels)
        elif model_type == "gat":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=8)
            self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1)
        else:
            raise ValueError(f"Unknown GNN model: {model_type}")
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def encode(self, features, edge_index):
        hidden = F.relu(self.conv1(features, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index)

    def decode(self, embeddings, pairs):
        left, right = embeddings[pairs[0]], embeddings[pairs[1]]
        symmetric = torch.cat((left * right, torch.abs(left - right)), dim=1)
        return self.decoder(symmetric).reshape(-1)


def training_edge_index(split, device="cpu"):
    pairs = torch.tensor(split.train_pos, dtype=torch.long, device=device).t().contiguous()
    return torch.cat((pairs, pairs.flip(0)), dim=1)


class FittedGNN:
    def __init__(self, model, features, edge_index, history, best_epoch, validation):
        self.model, self.features, self.edge_index = model, features, edge_index
        self.history, self.best_epoch, self.validation = history, best_epoch, validation

    @torch.inference_mode()
    def scores(self, pairs):
        self.model.eval()
        pairs = torch.as_tensor(np.ascontiguousarray(pairs), dtype=torch.long, device=self.features.device).t()
        embeddings = self.model.encode(self.features, self.edge_index)
        return self.model.decode(embeddings, pairs).sigmoid().cpu().numpy()

    def save(self, path, split_id):
        torch.save(
            {
                "task": "link_prediction",
                "config": self.model.config,
                "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "best_epoch": self.best_epoch,
                "split_id": split_id,
                "torch_version": str(torch.__version__),
            },
            path,
        )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "link_prediction":
        raise ValueError("Expected a link-prediction checkpoint.")
    model = GNNLinkPredictor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def fit_gnn(
    features, split, model_type, epochs=100, dimensions=32, hidden_channels=32, seed=42, device="cpu", log_every=0
):
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    torch.manual_seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    if features.ndim != 2 or features.shape[0] != split.num_nodes or not torch.isfinite(features).all():
        raise ValueError("Node features must be a finite matrix aligned with every node.")
    edge_index = training_edge_index(split, device)
    model = GNNLinkPredictor(features.shape[1], hidden_channels, dimensions, model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_pairs, train_labels = split.examples("train")
    train_pairs = torch.as_tensor(train_pairs, dtype=torch.long, device=device).t()
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    val_pairs, val_labels = split.examples("val")
    val_pairs = torch.as_tensor(val_pairs, dtype=torch.long, device=device).t()
    best_key, best_state, best_epoch, best_validation = (-1.0, -1.0), None, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(features, edge_index)
        loss = F.binary_cross_entropy_with_logits(model.decode(embeddings, train_pairs), train_labels)
        if not torch.isfinite(loss):
            raise ValueError("GNN training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            embeddings = model.encode(features, edge_index)
            scores = model.decode(embeddings, val_pairs).sigmoid().cpu().numpy()
        validation = evaluate(val_labels, scores)
        key = (validation["auc_roc"], validation["average_precision"])
        if key > best_key:
            best_key, best_epoch, best_validation = key, epoch, validation
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "validation": validation})
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                f"{model_type}: epoch {epoch}/{epochs}, loss={history[-1]['loss']:.5f}, validation AUC={validation['auc_roc']:.5f}, best epoch={best_epoch}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return FittedGNN(model, features, edge_index, history, best_epoch, best_validation)
