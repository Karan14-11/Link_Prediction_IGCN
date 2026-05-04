"""
Main Experiment: Snapshot-based Dynamic Link Prediction (GPU-ready)

Compares 6 strategies:
  1. GCN Static       — baseline GCN trained once, no updates
  2. GCN Retrain      — baseline GCN retrained from scratch each snapshot
  3. Inc-GCN NoUpdate — IncrementalGCN, no update (stale model)
  4. Inc-GCN CachedAX — IncrementalGCN with incremental AX cache updates + fine-tune
  5. Inc-GCN Subgraph — IncrementalGCN with subgraph-local fine-tuning
  6. Inc-GCN SVD      — IncrementalGCN with SVD row-selective gradient masking

Usage:
    python train_snapshot.py                    # defaults (CPU)
    python train_snapshot.py --device cuda      # GPU
    python train_snapshot.py --epochs 200       # more epochs
"""
import os, sys, copy, time, json, argparse
import torch
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.snapshot_builder import build_snapshots, compute_node_features
from model.gcn_link_predictor import GCNLinkPredictor, train_gcn_link, eval_gcn_link
from model.incremental_gcn_link import (
    IncrementalGCNLink, train_incremental_link, eval_incremental_link,
    fine_tune_incremental, fine_tune_svd_selective,
)
from model.incremental_utils import (
    compute_AX_sparse, build_adj_structures, update_AX_rows,
    nodes_for_AX_update, compute_edge_diff, compute_weight_svd, compare_svd,
    build_k_hop_subgraph_from_edge_index,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def negative_sample(num_nodes, num_neg):
    src = torch.randint(0, num_nodes, (num_neg,))
    dst = torch.randint(0, num_nodes, (num_neg,))
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def train_model_full(model, data, num_nodes, epochs, lr, device, model_type, desc="Training"):
    """Train a model from scratch with tqdm progress bar."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Sample training edges (cap at 50k for speed)
    max_train = min(data.edge_index.shape[1], 50000)
    perm = torch.randperm(data.edge_index.shape[1])[:max_train]
    pos_sub = data.edge_index[:, perm]
    neg_sub = negative_sample(num_nodes, max_train)

    pbar = tqdm(range(epochs), desc=desc, leave=False)
    for epoch in pbar:
        if model_type == 'gcn':
            loss = train_gcn_link(model, data, pos_sub, neg_sub, optimizer, device)
        else:
            loss = train_incremental_link(
                model, data.ax, data.edge_index, pos_sub, neg_sub, optimizer, device
            )
        pbar.set_postfix(loss=f"{loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(config, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    num_total  = config['snapshot']['num_total']
    num_train  = config['snapshot']['num_train']
    feature_dim = config['model_gcn']['feature_dim']
    gcn_cfg    = config['model_gcn']
    inc_cfg    = config['incremental']
    epochs     = args.epochs or config['training']['epochs']
    lr         = config['training']['lr']

    # ── 1. Build snapshots ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Building snapshots")
    print("=" * 60)

    snapshots, meta = build_snapshots(
        config['data_file'], num_snapshots=num_total, feature_dim=feature_dim
    )
    num_nodes = meta['num_nodes']

    print("Computing node features...")
    for s in tqdm(snapshots, desc="Features", leave=False):
        s.x = compute_node_features(s, num_nodes, feature_dim)

    # ── 2. Initial training on snapshot[num_train-1] ───────────────
    train_snap = snapshots[num_train - 1]
    train_snap.x = compute_node_features(train_snap, num_nodes, feature_dim)
    train_snap.ax = compute_AX_sparse(train_snap.edge_index, num_nodes, train_snap.x)

    print("\n" + "=" * 60)
    print(f"STEP 2: Initial training (snapshot {num_train-1})")
    print(f"  {num_nodes:,} nodes | {train_snap.edge_index.shape[1]//2:,} edges")
    print("=" * 60)

    # Baseline GCN
    t0 = time.time()
    gcn_model = GCNLinkPredictor(feature_dim, gcn_cfg['hidden_dim'],
                                  gcn_cfg['embed_dim'], gcn_cfg['dropout'])
    gcn_model = train_model_full(gcn_model, train_snap, num_nodes, epochs, lr,
                                  device, 'gcn', desc="GCN initial train")
    t_gcn_init = time.time() - t0
    print(f"  GCN trained in {t_gcn_init:.2f}s")

    # IncrementalGCN
    t0 = time.time()
    inc_model = IncrementalGCNLink(feature_dim, gcn_cfg['hidden_dim'],
                                    gcn_cfg['embed_dim'], gcn_cfg['dropout'])
    inc_model = train_model_full(inc_model, train_snap, num_nodes, epochs, lr,
                                  device, 'incremental', desc="Inc-GCN initial train")
    t_inc_init = time.time() - t0
    print(f"  Inc-GCN trained in {t_inc_init:.2f}s")

    svd_init = compute_weight_svd(inc_model, "init")

    # ── 3. Evaluate on test snapshots ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: Evaluating snapshots {num_train}–{num_total-1}")
    print("=" * 60)

    STRATEGIES = ['gcn_static', 'gcn_retrain',
                  'inc_no_update', 'inc_cached_ax', 'inc_subgraph', 'inc_svd']
    results = {s: [] for s in STRATEGIES}
    times   = {s: [] for s in STRATEGIES}
    svd_drifts = {s: [] for s in STRATEGIES}

    # Running copies for incremental strategies
    inc_cached   = copy.deepcopy(inc_model)
    inc_subgraph = copy.deepcopy(inc_model)
    inc_svd_m    = copy.deepcopy(inc_model)

    prev_edge_index = train_snap.edge_index.clone()
    prev_ax         = train_snap.ax.clone()

    ft_epochs = inc_cfg['finetune_epochs']
    ft_lr     = inc_cfg['finetune_lr']

    for snap_idx in tqdm(range(num_train, num_total), desc="Test snapshots"):
        snap = snapshots[snap_idx]
        snap.x = compute_node_features(snap, num_nodes, feature_dim)

        # Test edges = new edges in this snapshot
        test_pos = snap.new_edges
        if test_pos.shape[1] == 0:
            for s in STRATEGIES:
                results[s].append({'auc': float('nan'), 'ap': float('nan')})
                times[s].append(0.0)
            continue

        test_neg = negative_sample(num_nodes, test_pos.shape[1])

        # Fine-tuning edges (subsample for speed)
        max_ft = min(snap.edge_index.shape[1], 20000)
        perm_ft = torch.randperm(snap.edge_index.shape[1])[:max_ft]
        ft_pos = snap.edge_index[:, perm_ft]
        ft_neg = negative_sample(num_nodes, max_ft)

        # Edge diff for incremental strategies
        added, removed, affected = compute_edge_diff(prev_edge_index, snap.edge_index)

        # ── Strategy 1: GCN Static ──
        t0 = time.time()
        r = eval_gcn_link(gcn_model, snap, test_pos, test_neg, device)
        times['gcn_static'].append(time.time() - t0)
        results['gcn_static'].append(r)

        # ── Strategy 2: GCN Retrain ──
        t0 = time.time()
        gcn_re = GCNLinkPredictor(feature_dim, gcn_cfg['hidden_dim'],
                                   gcn_cfg['embed_dim'], gcn_cfg['dropout'])
        gcn_re = train_model_full(gcn_re, snap, num_nodes,
                                   max(epochs // 2, 30), lr, device, 'gcn',
                                   desc=f"GCN retrain s{snap_idx}")
        r = eval_gcn_link(gcn_re, snap, test_pos, test_neg, device)
        times['gcn_retrain'].append(time.time() - t0)
        results['gcn_retrain'].append(r)

        # ── Strategy 3: Inc-GCN No Update ──
        t0 = time.time()
        snap_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x)
        r = eval_incremental_link(inc_model, snap_ax, snap.edge_index,
                                   test_pos, test_neg, device)
        times['inc_no_update'].append(time.time() - t0)
        results['inc_no_update'].append(r)

        # ── Strategy 4: Cached AX ──
        t0 = time.time()
        if 0 < len(affected) < num_nodes // 2:
            adj, deg, ns, n2i = build_adj_structures(snap.edge_index, num_nodes)
            rows = nodes_for_AX_update(adj, affected)
            cached_ax = update_AX_rows(adj, deg, ns, n2i, snap.x.cpu(), prev_ax, rows)
        else:
            cached_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x)
        m_c = copy.deepcopy(inc_cached)
        m_c = fine_tune_incremental(m_c, cached_ax, snap.edge_index,
                                     ft_pos, ft_neg, ft_epochs, ft_lr, device)
        r = eval_incremental_link(m_c, cached_ax, snap.edge_index,
                                   test_pos, test_neg, device)
        times['inc_cached_ax'].append(time.time() - t0)
        results['inc_cached_ax'].append(r)
        svd_c = compute_weight_svd(m_c)
        svd_drifts['inc_cached_ax'].append(compare_svd(svd_init, svd_c))
        inc_cached = m_c

        # ── Strategy 5: Subgraph ──
        t0 = time.time()
        sub_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x)
        if len(affected) > 0:
            sub_nodes, sub_ei, _, g2l = build_k_hop_subgraph_from_edge_index(
                snap.edge_index, affected, num_nodes, k=inc_cfg['k_hop']
            )
            sub_set = set(sub_nodes)
            ss, sd = [], []
            for i in range(ft_pos.shape[1]):
                s, d = ft_pos[0, i].item(), ft_pos[1, i].item()
                if s in sub_set and d in sub_set:
                    ss.append(s); sd.append(d)
            if ss:
                sub_ft_pos = torch.tensor([ss, sd], dtype=torch.long)
                sub_ft_neg = negative_sample(num_nodes, len(ss))
            else:
                sub_ft_pos, sub_ft_neg = ft_pos, ft_neg
        else:
            sub_ft_pos, sub_ft_neg = ft_pos, ft_neg

        m_s = copy.deepcopy(inc_subgraph)
        m_s = fine_tune_incremental(m_s, sub_ax, snap.edge_index,
                                     sub_ft_pos, sub_ft_neg, ft_epochs, ft_lr, device)
        r = eval_incremental_link(m_s, sub_ax, snap.edge_index,
                                   test_pos, test_neg, device)
        times['inc_subgraph'].append(time.time() - t0)
        results['inc_subgraph'].append(r)
        inc_subgraph = m_s

        # ── Strategy 6: SVD Selective ──
        t0 = time.time()
        svd_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x)
        m_v = copy.deepcopy(inc_svd_m)
        m_v = fine_tune_svd_selective(
            m_v, svd_ax, snap.edge_index, ft_pos, ft_neg,
            ft_epochs, ft_lr, inc_cfg['svd_k'], inc_cfg['svd_top_k'], device
        )
        r = eval_incremental_link(m_v, svd_ax, snap.edge_index,
                                   test_pos, test_neg, device)
        times['inc_svd'].append(time.time() - t0)
        results['inc_svd'].append(r)
        svd_v = compute_weight_svd(m_v)
        svd_drifts['inc_svd'].append(compare_svd(svd_init, svd_v))
        inc_svd_m = m_v

        # Update state
        prev_edge_index = snap.edge_index.clone()
        prev_ax = snap_ax.clone()

        # Print snapshot summary
        tqdm.write(
            f"  Snap {snap_idx} | "
            f"GCN-S:{results['gcn_static'][-1]['auc']:.4f} "
            f"GCN-R:{results['gcn_retrain'][-1]['auc']:.4f} "
            f"Inc-No:{results['inc_no_update'][-1]['auc']:.4f} "
            f"Inc-AX:{results['inc_cached_ax'][-1]['auc']:.4f} "
            f"Inc-Sub:{results['inc_subgraph'][-1]['auc']:.4f} "
            f"Inc-SVD:{results['inc_svd'][-1]['auc']:.4f}"
        )

    # ── 4. Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    LABELS = {
        'gcn_static':    'GCN (Static)',
        'gcn_retrain':   'GCN (Retrain)',
        'inc_no_update': 'Inc-GCN (NoUpdate)',
        'inc_cached_ax': 'Inc-GCN (CachedAX)',
        'inc_subgraph':  'Inc-GCN (Subgraph)',
        'inc_svd':       'Inc-GCN (SVD)',
    }

    summary = {}
    print(f"\n{'Strategy':<25s} {'Mean AUC':>10s} {'Std AUC':>10s} "
          f"{'Mean AP':>10s} {'Std AP':>10s} {'Total Time':>12s} {'Avg Time':>10s}")
    print("-" * 90)

    for s in STRATEGIES:
        aucs = [r['auc'] for r in results[s] if not np.isnan(r.get('auc', np.nan))]
        aps  = [r['ap']  for r in results[s] if not np.isnan(r.get('ap', np.nan))]
        tt = sum(times[s])
        at = np.mean(times[s]) if times[s] else 0
        summary[s] = {
            'mean_auc': float(np.mean(aucs)) if aucs else 0,
            'std_auc':  float(np.std(aucs))  if aucs else 0,
            'mean_ap':  float(np.mean(aps))  if aps else 0,
            'std_ap':   float(np.std(aps))   if aps else 0,
            'total_time': tt,
            'avg_time': at,
        }
        d = summary[s]
        print(f"  {LABELS[s]:<23s} {d['mean_auc']:10.4f} {d['std_auc']:10.4f} "
              f"{d['mean_ap']:10.4f} {d['std_ap']:10.4f} {tt:10.2f}s {at:10.2f}s")

    # ── 5. Save ────────────────────────────────────────────────────
    output = {
        'results':  {k: [dict(r) for r in v] for k, v in results.items()},
        'times':    times,
        'summary':  summary,
        'svd_drifts': {k: [{kk: float(vv) for kk, vv in d.items()} if d else {}
                        for d in v] for k, v in svd_drifts.items()},
        'metadata': {
            'num_nodes': num_nodes,
            'num_train': num_train,
            'num_test':  num_total - num_train,
            'init_time_gcn': t_gcn_init,
            'init_time_inc': t_inc_init,
            'device': str(device),
            'epochs': epochs,
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → results/experiment_results.json")

    # Generate plots
    print("\nGenerating plots...")
    from visualize_results import generate_all_plots
    generate_all_plots('results/experiment_results.json')

    return output


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic GNN Link Prediction Experiment")
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: from config)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path')
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = args.config or os.path.join(project_root, 'configs', 'default.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve data_file
    data_file = config.get('data_file', 'processed/askubuntu_compact.pt')
    if not os.path.isabs(data_file):
        config['data_file'] = os.path.join(project_root, data_file.lstrip('../'))

    # Override device from CLI
    if args.device:
        config['training']['device'] = args.device
    if not args.device:
        args.device = config['training'].get('device', 'cpu')

    os.chdir(project_root)
    run_experiment(config, args)
