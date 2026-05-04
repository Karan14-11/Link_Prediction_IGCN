"""
Incremental GCN for Link Prediction.
Adapted from DynamicGNN-main: first layer uses cached AX, second uses GCNConv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class IncrementalGCNLink(nn.Module):
    """
    First hop: Linear(AX) where AX = precomputed A_hat @ X
    Second hop: GCNConv on hidden features
    Decoder: dot product for link prediction
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, ax, edge_index):
        """Encode using cached AX for first layer."""
        x = self.lin1(ax)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=1)

    def forward(self, ax, edge_index, edge_label_index):
        z = self.encode(ax, edge_index)
        return self.decode(z, edge_label_index)


def train_incremental_link(model, ax, edge_index, train_pos_edge, train_neg_edge,
                           optimizer, device='cpu'):
    """One training step for IncrementalGCNLink."""
    model.train()
    optimizer.zero_grad()
    ax = ax.to(device)
    edge_index = edge_index.to(device)
    z = model.encode(ax, edge_index)
    pos_score = model.decode(z, train_pos_edge.to(device))
    neg_score = model.decode(z, train_neg_edge.to(device))
    pos_label = torch.ones(pos_score.shape[0], device=device)
    neg_label = torch.zeros(neg_score.shape[0], device=device)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([pos_label, neg_label])
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_incremental_link(model, ax, edge_index, pos_edge, neg_edge, device='cpu'):
    """Evaluate IncrementalGCNLink."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()
    ax = ax.to(device)
    edge_index = edge_index.to(device)
    z = model.encode(ax, edge_index)
    pos_score = model.decode(z, pos_edge.to(device)).cpu()
    neg_score = model.decode(z, neg_edge.to(device)).cpu()
    scores = torch.cat([pos_score, neg_score]).sigmoid().numpy()
    labels = torch.cat([
        torch.ones(pos_score.shape[0]),
        torch.zeros(neg_score.shape[0])
    ]).numpy()
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    ap = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.5
    return {'auc': auc, 'ap': ap}


def fine_tune_incremental(model, ax, edge_index, train_pos_edge, train_neg_edge,
                          epochs=50, lr=5e-4, device='cpu'):
    """Fine-tune model after incremental AX update."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        train_incremental_link(model, ax, edge_index, train_pos_edge, train_neg_edge,
                               optimizer, device)
    return model


def fine_tune_svd_selective(model, ax, edge_index, train_pos_edge, train_neg_edge,
                            epochs=50, lr=5e-4, svd_k=5, svd_top_k=5, device='cpu'):
    """Fine-tune with SVD-based row-selective gradient masking."""
    from .incremental_utils import compute_important_rows, mask_gradients

    # Compute important rows before training
    imp_W1 = compute_important_rows(model.lin1.weight, k=svd_k, top_k=svd_top_k)
    imp_W2 = compute_important_rows(model.conv2.lin.weight, k=svd_k, top_k=svd_top_k)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        ax_d = ax.to(device)
        ei_d = edge_index.to(device)
        z = model.encode(ax_d, ei_d)
        pos_score = model.decode(z, train_pos_edge.to(device))
        neg_score = model.decode(z, train_neg_edge.to(device))
        pos_label = torch.ones(pos_score.shape[0], device=device)
        neg_label = torch.zeros(neg_score.shape[0], device=device)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        # Mask gradients — only update important rows
        mask_gradients(model.lin1.weight, imp_W1)
        mask_gradients(model.conv2.lin.weight, imp_W2)
        optimizer.step()

    return model
