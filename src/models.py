from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_mean_pool,
)

class PhysicsInformedLoss(nn.Module):
    """
    Weighted MSE for band gap + formation energy, with an additional penalty
    whenever the model's predicted formation energy crosses the thermodynamic
    stability boundary (sign changes relative to the true value)

    The penalty discourages mislabelling stable materials as unstable and
    vice-versa
    """

    def __init__(self, stability_penalty = 5.0, bandgap_weight = 1.0, formation_weight = 1.0,):
        super().__init__()
        self.stability_penalty = stability_penalty
        self.w_bg = bandgap_weight
        self.w_fe = formation_weight

    def forward(self, pred, target):
        loss_bg = F.mse_loss(pred[:, 0], target[:, 0])
        loss_fe = F.mse_loss(pred[:, 1], target[:, 1])

        boundary_crossed = (torch.sign(pred[:, 1]) != torch.sign(target[:, 1])).float()
        penalty = (boundary_crossed * self.stability_penalty * (pred[:, 1] - target[:, 1]) ** 2).mean()

        total = self.w_bg * loss_bg + self.w_fe * loss_fe + penalty
        return total, {
            "loss_bandgap": loss_bg.item(),
            "loss_formation": loss_fe.item(),
            "physics_penalty": penalty.item(),
            "total_loss": total.item(),
        }



class GCN(nn.Module):
    def __init__(
        self,
        node_features,
        hidden_dim = 128,
        num_layers = 3,
        dropout = 0.1,
    ):
        super().__init__()
        dims = [node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = _regression_head(hidden_dim)

    def forward(self, data_or_x, edge_index=None, batch=None):
        if isinstance(data_or_x, Data):
            x, edge_index, batch = data_or_x.x, data_or_x.edge_index, data_or_x.batch
        else:
            x = data_or_x
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch))

class GAT(nn.Module):
    def __init__(
        self,
        node_features,
        hidden_dim = 64,
        num_layers = 3,
        heads = 4,
        dropout = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        in_dim = node_features
        for i in range(num_layers):
            last = i == num_layers - 1
            if last:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                in_dim = hidden_dim
            else:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
                in_dim = hidden_dim * heads

        self.head = _regression_head(hidden_dim)

    def forward(self, data_or_x, edge_index=None, batch=None) -> torch.Tensor:
        if isinstance(data_or_x, Data):
            x, edge_index, batch = data_or_x.x, data_or_x.edge_index, data_or_x.batch
        else:
            x = data_or_x
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch))


class GIN(nn.Module):
    def __init__(
        self,
        node_features,
        hidden_dim = 128,
        num_layers = 3,
        dropout = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        def _mlp(in_d: int, out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
            )

        in_dim = node_features
        for _ in range(num_layers):
            self.convs.append(GINConv(_mlp(in_dim, hidden_dim), train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data_or_x, edge_index=None, batch=None) -> torch.Tensor:
        if isinstance(data_or_x, Data):
            x, edge_index, batch = data_or_x.x, data_or_x.edge_index, data_or_x.batch
        else:
            x = data_or_x
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(global_add_pool(x, batch))
        return self.head(torch.cat(layer_outs, dim=1))


def _regression_head(hidden_dim):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim // 2, 2),
    )

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y.view(-1, 2)
        loss, _ = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, all_preds, all_targets = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y.view(-1, 2)
        loss, _ = loss_fn(pred, target)
        total += loss.item() * batch.num_graphs
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return {
        "loss": total / len(loader.dataset),
        "mae_bandgap": F.l1_loss(preds[:, 0], targets[:, 0]).item(),
        "mae_formation": F.l1_loss(preds[:, 1], targets[:, 1]).item(),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    epochs = 100,
    lr = 1e-3,
    weight_decay = 1e-5,
    device = "cpu",
    model_name = "gnn",
    use_wandb = False,
    patience = 15,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = PhysicsInformedLoss()

    best_val, best_state, no_improve = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])

        row = {"epoch": epoch, "train_loss": tr_loss, **val_metrics}
        history.append(row)

        if use_wandb:
            try:
                import wandb
                wandb.log(row)
            except ImportError:
                pass

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{model_name}] Early stop at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f"[{model_name}] epoch {epoch:3d} | "
                f"train {tr_loss:.4f} | "
                f"val MAE bg {val_metrics['mae_bandgap']:.4f} | "
                f"val MAE fe {val_metrics['mae_formation']:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)
