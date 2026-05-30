from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.compat.proto import event_pb2, summary_pb2

# Resolve Chinese-username path issues: allow overriding ROOT via env var,
# force UTF-8 I/O, and fix stdout encoding on Windows.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
if sys.platform == "win32":
    for attr in ("stdout", "stderr"):
        stream = getattr(sys, attr)
        try:
            buf = stream.buffer
        except AttributeError:
            pass
        else:
            setattr(sys, attr, io.TextIOWrapper(buf, encoding="utf-8", errors="replace"))

ROOT = Path(os.environ.get("TUGRAPH3_ROOT", Path(__file__).resolve().parents[1]))
DATASET = Path.home() / "tugraph_homework_submission_03" / "data" / "raw" / "Dataset-Unicauca-Version2-87Atts.csv"
IMPORT_DIR = ROOT / "tugraph_import"
OUTPUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"
REPORT_DIR = ROOT / "reports"


CRC32C_POLY = 0x82F63B78


def _crc32c_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ CRC32C_POLY if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return table


CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class SimpleTensorBoardWriter:
    """Small TFRecord event writer that avoids TensorFlow gfile on CJK Windows paths."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        host = os.environ.get("COMPUTERNAME", "windows")
        filename = f"events.out.tfevents.{int(time.time())}.{host}"
        self.path = self.log_dir / filename
        self.file = self.path.open("wb")
        self._write_event(event_pb2.Event(wall_time=time.time(), file_version="brain.Event:2"))

    def _write_record(self, data: bytes) -> None:
        length = struct.pack("<Q", len(data))
        self.file.write(length)
        self.file.write(struct.pack("<I", masked_crc32c(length)))
        self.file.write(data)
        self.file.write(struct.pack("<I", masked_crc32c(data)))

    def _write_event(self, event: event_pb2.Event) -> None:
        self._write_record(event.SerializeToString())

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))])
        self._write_event(event_pb2.Event(wall_time=time.time(), step=int(step), summary=summary))

    def close(self) -> None:
        self.file.flush()
        self.file.close()


@dataclass
class ExperimentConfig:
    scan_rows: int
    top_classes: int
    samples_per_class: int
    embedding_dim: int
    test_size: float
    random_state: int
    epochs: int
    batch_size: int
    causal_window_seconds: int


def endpoint(ip: object, port: object) -> str:
    try:
        port_text = str(int(float(port)))
    except Exception:
        port_text = str(port)
    return f"{ip}:{port_text}"


def safe_numeric_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    excluded = {
        "Flow.ID",
        "Source.IP",
        "Destination.IP",
        "Timestamp",
        "Label",
        "ProtocolName",
        "L7Protocol",
        target_col,
    }
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    x = df[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True).fillna(0)
    return x.fillna(medians)


def load_balanced_sample(cfg: ExperimentConfig) -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print(f"Loading first {cfg.scan_rows:,} rows from {DATASET.name} ...")
    df = pd.read_csv(DATASET, nrows=cfg.scan_rows, low_memory=False, encoding="utf-8")
    df = df.dropna(subset=["Source.IP", "Destination.IP", "Source.Port", "Destination.Port", "ProtocolName"])
    top = df["ProtocolName"].value_counts().head(cfg.top_classes).index.tolist()
    df = df[df["ProtocolName"].isin(top)].copy()
    sampled = (
        df.groupby("ProtocolName", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cfg.samples_per_class), random_state=cfg.random_state))
        .sample(frac=1.0, random_state=cfg.random_state)
        .reset_index(drop=True)
    )
    sampled.insert(0, "row_id", [f"flow_{i:07d}" for i in range(len(sampled))])
    sampled["src_endpoint"] = [endpoint(ip, port) for ip, port in zip(sampled["Source.IP"], sampled["Source.Port"])]
    sampled["dst_endpoint"] = [endpoint(ip, port) for ip, port in zip(sampled["Destination.IP"], sampled["Destination.Port"])]
    sampled["timestamp_dt"] = pd.to_datetime(sampled["Timestamp"], format="%d/%m/%Y%H:%M:%S", errors="coerce")
    print("Class distribution:")
    print(sampled["ProtocolName"].value_counts().to_string())
    return sampled


def export_hcg(df: pd.DataFrame) -> None:
    endpoints = pd.concat(
        [
            df[["src_endpoint", "Source.IP", "Source.Port"]].rename(
                columns={"src_endpoint": "endpoint_id", "Source.IP": "ip", "Source.Port": "port"}
            ),
            df[["dst_endpoint", "Destination.IP", "Destination.Port"]].rename(
                columns={"dst_endpoint": "endpoint_id", "Destination.IP": "ip", "Destination.Port": "port"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("endpoint_id")
    endpoints["port"] = pd.to_numeric(endpoints["port"], errors="coerce").fillna(-1).astype(int)

    edges = pd.DataFrame(
        {
            "src_endpoint": df["src_endpoint"],
            "dst_endpoint": df["dst_endpoint"],
            "flow_id": df["row_id"],
            "protocol": pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(int),
            "protocol_name": df["ProtocolName"],
            "l7_protocol": pd.to_numeric(df["L7Protocol"], errors="coerce").fillna(-1).astype(int),
            "flow_duration": pd.to_numeric(df["Flow.Duration"], errors="coerce").fillna(0).astype(float),
            "total_packets": (
                pd.to_numeric(df["Total.Fwd.Packets"], errors="coerce").fillna(0)
                + pd.to_numeric(df["Total.Backward.Packets"], errors="coerce").fillna(0)
            ).astype(float),
            "total_bytes": (
                pd.to_numeric(df["Total.Length.of.Fwd.Packets"], errors="coerce").fillna(0)
                + pd.to_numeric(df["Total.Length.of.Bwd.Packets"], errors="coerce").fillna(0)
            ).astype(float),
            "timestamp": df["Timestamp"],
        }
    )
    endpoints.to_csv(IMPORT_DIR / "hcg_vertices_endpoint.csv", index=False, encoding="utf-8")
    edges.to_csv(IMPORT_DIR / "hcg_edges_communicates.csv", index=False, encoding="utf-8")


def export_tcg(df: pd.DataFrame, causal_window_seconds: int) -> None:
    """Export TCG with four edge types: CR, PR, DHR, SHR (Liu Zhen paper)."""
    # Pre-compute numeric ports and ensure correct types
    df = df.copy()
    df["_src_port"] = pd.to_numeric(df["Source.Port"], errors="coerce").fillna(-1).astype(int)
    df["_dst_port"] = pd.to_numeric(df["Destination.Port"], errors="coerce").fillna(-1).astype(int)
    df["_protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(int)

    # Flow vertices
    vertices = pd.DataFrame(
        {
            "flow_id": df["row_id"],
            "src_endpoint": df["src_endpoint"],
            "dst_endpoint": df["dst_endpoint"],
            "protocol_name": df["ProtocolName"],
            "l7_protocol": pd.to_numeric(df["L7Protocol"], errors="coerce").fillna(-1).astype(int),
            "flow_duration": pd.to_numeric(df["Flow.Duration"], errors="coerce").fillna(0).astype(float),
            "total_packets": (
                pd.to_numeric(df["Total.Fwd.Packets"], errors="coerce").fillna(0)
                + pd.to_numeric(df["Total.Backward.Packets"], errors="coerce").fillna(0)
            ).astype(float),
            "total_bytes": (
                pd.to_numeric(df["Total.Length.of.Fwd.Packets"], errors="coerce").fillna(0)
                + pd.to_numeric(df["Total.Length.of.Bwd.Packets"], errors="coerce").fillna(0)
            ).astype(float),
            "timestamp": df["Timestamp"],
        }
    )

    rows = df[["row_id", "Source.IP", "Destination.IP", "_src_port", "_dst_port", "_protocol", "timestamp_dt"]].copy()
    rows.columns = ["row_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp_dt"]
    recs = rows.to_dict("records")

    cr_edges = _build_cr_edges(recs, causal_window_seconds)
    pr_edges = _build_pr_edges(recs, causal_window_seconds)
    dhr_edges = _build_dhr_edges(recs, causal_window_seconds)
    shr_edges = _build_shr_edges(recs, causal_window_seconds)

    total = len(cr_edges) + len(pr_edges) + len(dhr_edges) + len(shr_edges)
    print(f"TCG edges — CR:{len(cr_edges)} PR:{len(pr_edges)} DHR:{len(dhr_edges)} SHR:{len(shr_edges)} total:{total}")

    vertices.to_csv(IMPORT_DIR / "tcg_vertices_flow.csv", index=False, encoding="utf-8")

    _save_edges(cr_edges, "tcg_edges_CR.csv", ("src_flow", "dst_flow", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "delta_seconds"))
    _save_edges(pr_edges, "tcg_edges_PR.csv", ("src_flow", "dst_flow", "shared_ip", "delta_seconds"))
    _save_edges(dhr_edges, "tcg_edges_DHR.csv", ("src_flow", "dst_flow", "shared_ip", "src_port_f1", "src_port_f2", "delta_seconds"))
    _save_edges(shr_edges, "tcg_edges_SHR.csv", ("src_flow", "dst_flow", "shared_ip", "shared_port", "delta_seconds"))


def _save_edges(edges: list[dict[str, object]], filename: str, columns: tuple[str, ...]) -> None:
    df_out = pd.DataFrame(edges, columns=list(columns)) if edges else pd.DataFrame(columns=list(columns))
    df_out.to_csv(IMPORT_DIR / filename, index=False, encoding="utf-8")


def _build_cr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    """CR: Communication Relationship — bidirectional counterpart flows.

    protocol(f1)=protocol(f2), srcIp(f1)=dstIp(f2), srcPort(f1)=dstPort(f2),
    dstIp(f1)=srcIp(f2), dstPort(f1)=srcPort(f2)
    """
    from collections import defaultdict
    index: dict[tuple, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        key = (r["protocol"], str(r["src_ip"]), str(r["dst_ip"]), int(r["src_port"]), int(r["dst_port"]))
        rev_key = (r["protocol"], str(r["dst_ip"]), str(r["src_ip"]), int(r["dst_port"]), int(r["src_port"]))
        for other in index.get(rev_key, []):
            delta = abs((ts - other["timestamp_dt"]).total_seconds())
            if delta <= window_sec and r["row_id"] != other["row_id"]:
                edges.append({
                    "src_flow": other["row_id"], "dst_flow": r["row_id"],
                    "src_ip": str(r["src_ip"]), "src_port": int(r["src_port"]),
                    "dst_ip": str(r["dst_ip"]), "dst_port": int(r["dst_port"]),
                    "protocol": r["protocol"], "delta_seconds": float(delta),
                })
                break
        index[key].append(r)
    return edges


def _build_pr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    """PR: Propagation Relationship — dstIp(f1) = srcIp(f2)."""
    from collections import defaultdict
    index: dict[str, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        for prev in index.get(str(r["src_ip"]), []):
            delta = (ts - prev["timestamp_dt"]).total_seconds()
            if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                edges.append({
                    "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                    "shared_ip": str(r["src_ip"]), "delta_seconds": float(delta),
                })
                break
        index[str(r["dst_ip"])].append(r)
    return edges


def _build_dhr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    """DHR: Dynamic-Port Host Relationship — srcIp(f1)=srcIp(f2), srcPort(f1)≠srcPort(f2)."""
    from collections import defaultdict
    index: dict[str, list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        src_ip = str(r["src_ip"])
        src_port = int(r["src_port"])
        for prev in index[src_ip]:
            prev_port = int(prev["src_port"])
            if prev_port != src_port:
                delta = (ts - prev["timestamp_dt"]).total_seconds()
                if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                    edges.append({
                        "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                        "shared_ip": src_ip, "src_port_f1": prev_port,
                        "src_port_f2": src_port, "delta_seconds": float(delta),
                    })
                    break
        index[src_ip].append(r)
    return edges


def _build_shr_edges(recs: list[dict], window_sec: int) -> list[dict[str, object]]:
    """SHR: Static-Port Host Relationship — srcIp(f1)=srcIp(f2), srcPort(f1)=srcPort(f2)."""
    from collections import defaultdict
    index: dict[tuple[str, int], list[dict]] = defaultdict(list)
    edges: list[dict[str, object]] = []
    for r in recs:
        ts = r["timestamp_dt"]
        if ts is None or pd.isna(ts):
            continue
        key = (str(r["src_ip"]), int(r["src_port"]))
        for prev in index[key]:
            delta = (ts - prev["timestamp_dt"]).total_seconds()
            if 0 <= delta <= window_sec and r["row_id"] != prev["row_id"]:
                edges.append({
                    "src_flow": prev["row_id"], "dst_flow": r["row_id"],
                    "shared_ip": key[0], "shared_port": key[1],
                    "delta_seconds": float(delta),
                })
                break
        index[key].append(r)
    return edges


def build_nx_graph(left: Iterable[str], right: Iterable[str], directed: bool = False):
    """Build a networkx graph from edge lists."""
    import networkx as nx

    g = nx.DiGraph() if directed else nx.Graph()
    for u, v in zip(left, right):
        if u != v:
            g.add_edge(u, v)
    return g


def node2vec_random_walks(g, walk_length: int, num_walks: int, p: float, q: float, seed: int = 42) -> list[list[str]]:
    """Generate biased random walks à la Node2Vec.

    p = return parameter  (small → avoid backtracking, larger → encourage)
    q = in-out parameter   (<1 → BFS-like, >1 → DFS-like)
    """
    import networkx as nx

    rng = np.random.RandomState(seed)
    nodes = list(g.nodes())
    walks: list[list[str]] = []

    # Precompute alias tables for each node to avoid O(degree²) per step
    alias_nodes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for node in nodes:
        neighbors = list(g.neighbors(node))
        if not neighbors:
            alias_nodes[node] = (np.array([0.0]), np.array([0], dtype=np.int32))
            continue
        # Unnormalised transition probs (uniform for the first step)
        probs = np.ones(len(neighbors), dtype=np.float64) / len(neighbors)
        alias_nodes[node] = _alias_setup(probs)

    alias_edges: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _get_alias_edge(t: str, v: str) -> tuple[np.ndarray, np.ndarray]:
        key = (t, v)
        if key in alias_edges:
            return alias_edges[key]
        neighbors = list(g.neighbors(v))
        if not neighbors:
            probs = np.array([1.0])
            j_arr = np.array([0], dtype=np.int32)
            alias_edges[key] = (probs, j_arr)
            return alias_edges[key]

        probs = np.zeros(len(neighbors), dtype=np.float64)
        for i, x in enumerate(neighbors):
            if x == t:
                probs[i] = 1.0 / p
            elif g.has_edge(x, t) or (isinstance(g, nx.Graph) and g.has_edge(t, x)):
                probs[i] = 1.0
            else:
                probs[i] = 1.0 / q
        probs /= probs.sum()
        alias_edges[key] = _alias_setup(probs)
        return alias_edges[key]

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            if g.degree(start) == 0:
                continue
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_neighbors = list(g.neighbors(cur))
                if not cur_neighbors:
                    break
                if len(walk) == 1:
                    probs, j_arr = alias_nodes[cur]
                else:
                    probs, j_arr = _get_alias_edge(walk[-2], cur)
                # Sample using alias method
                idx = _alias_draw(probs, j_arr, rng)
                if idx < len(cur_neighbors):
                    walk.append(cur_neighbors[idx])
                else:
                    break
            if len(walk) >= 2:
                walks.append(walk)
    return walks


def _alias_setup(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build Vose alias table. Returns (prob, alias) arrays."""
    n = len(probs)
    prob = probs * n
    alias = np.zeros(n, dtype=np.int32)
    small = []
    large = []
    for i, p in enumerate(prob):
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        s = small.pop()
        l = large.pop()
        alias[s] = l
        prob[l] = prob[l] + prob[s] - 1.0
        if prob[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
    while large:
        prob[large.pop()] = 1.0
    while small:
        prob[small.pop()] = 1.0
    return prob.astype(np.float32), alias


def _alias_draw(prob: np.ndarray, alias: np.ndarray, rng: np.random.RandomState) -> int:
    """Draw a sample from the alias table."""
    n = len(prob)
    col = rng.randint(0, n)
    if rng.rand() < prob[col]:
        return col
    return alias[col]


def node2vec_embeddings(
    left: Iterable[str],
    right: Iterable[str],
    dim: int,
    walk_length: int = 7,
    num_walks: int = 10,
    p: float = 0.3,
    q: float = 0.7,
    directed: bool = False,
    epochs: int = 5,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Node2Vec graph embeddings via Skip-gram trained with PyTorch.

    Parameters match Su Yongcai paper: p=0.3, q=0.7, walk_length=7.
    Default embedding dim = 18 (as in the paper).
    """
    g = build_nx_graph(left, right, directed=directed)
    nodes = sorted(g.nodes())
    if len(nodes) < 2:
        return {n: np.zeros(dim, dtype=np.float32) for n in nodes}, np.zeros((len(nodes), dim), dtype=np.float32)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    walks = node2vec_random_walks(g, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    if len(walks) < 10:
        # Fallback: if graph is too sparse, use simple adjacency-based embeddings
        adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for u, v in zip(left, right):
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                adj[i, j] = 1.0
                if not directed:
                    adj[j, i] = 1.0
        n_components = max(1, min(dim, min(adj.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = svd.fit_transform(adj).astype(np.float32)
        if n_components < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - n_components)))
        return {node: emb[i] for node, i in node_to_idx.items()}, emb

    # Train Skip-gram on random walks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_walks = [[node_to_idx[n] for n in w] for w in walks]

    # Build training pairs (center, context) with window size 5
    window = 5
    pairs: list[tuple[int, int]] = []
    for walk in idx_walks:
        for i, center in enumerate(walk):
            start = max(0, i - window)
            end = min(len(walk), i + window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, walk[j]))

    if len(pairs) == 0:
        emb = np.random.randn(len(nodes), dim).astype(np.float32) * 0.01
        return {node: emb[i] for node, i in node_to_idx.items()}, emb

    # Simple Skip-gram with negative sampling
    n_vocab = len(nodes)
    emb_in = nn.Embedding(n_vocab, dim, sparse=False)
    emb_out = nn.Embedding(n_vocab, dim, sparse=False)
    emb_in.weight.data.normal_(0, 0.01)
    emb_out.weight.data.normal_(0, 0.01)

    # Noise distribution for negative sampling (unigram ^ 0.75)
    node_freq = np.zeros(n_vocab, dtype=np.float64)
    for w in idx_walks:
        for ni in w:
            node_freq[ni] += 1
    node_freq = node_freq ** 0.75
    node_freq /= node_freq.sum()

    neg_samples = 5
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.01)

    pair_arr = np.array(pairs, dtype=np.int64)
    batch_size = 2048
    n_batches = max(1, len(pair_arr) // batch_size)

    for epoch in range(epochs):
        np.random.shuffle(pair_arr)
        total_loss = 0.0
        for b in range(n_batches):
            batch = pair_arr[b * batch_size : (b + 1) * batch_size]
            centers = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            neg = torch.multinomial(
                torch.tensor(node_freq, dtype=torch.float32, device=device),
                len(centers) * neg_samples,
                replacement=True,
            ).view(len(centers), neg_samples)

            emb_c = emb_in(centers)     # (B, dim)
            emb_pos = emb_out(contexts)  # (B, dim)
            emb_neg = emb_out(neg)       # (B, neg, dim)

            pos_score = (emb_c * emb_pos).sum(dim=1).sigmoid().log()  # (B,)
            neg_score = (-(emb_c.unsqueeze(1) * emb_neg).sum(dim=2)).sigmoid().log().sum(dim=1)  # (B,)

            loss = -(pos_score + neg_score).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

    emb = emb_in.weight.detach().cpu().numpy().astype(np.float32)
    return {node: emb[i] for node, i in node_to_idx.items()}, emb


def build_graph_features(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    # HCG node embeddings (Node2Vec on communication graph)
    endpoint_emb, _ = node2vec_embeddings(
        df["src_endpoint"], df["dst_endpoint"],
        dim=cfg.embedding_dim, walk_length=7, num_walks=10,
        p=0.3, q=0.7, directed=False,
    )
    src_hcg = np.vstack([endpoint_emb[e] for e in df["src_endpoint"]])
    dst_hcg = np.vstack([endpoint_emb[e] for e in df["dst_endpoint"]])
    # HCG edge embeddings: concat + diff + product
    hcg_edge_emb = np.hstack([src_hcg, dst_hcg, np.abs(src_hcg - dst_hcg), src_hcg * dst_hcg])

    # TCG flow embeddings from 4 edge types
    tcg_embeddings = []
    edge_files = {
        "CR": ("tcg_edges_CR.csv", ("src_flow", "dst_flow")),
        "PR": ("tcg_edges_PR.csv", ("src_flow", "dst_flow")),
        "DHR": ("tcg_edges_DHR.csv", ("src_flow", "dst_flow")),
        "SHR": ("tcg_edges_SHR.csv", ("src_flow", "dst_flow")),
    }
    for label, (filename, (src_col, dst_col)) in edge_files.items():
        edge_path = IMPORT_DIR / filename
        if edge_path.exists():
            edge_df = pd.read_csv(edge_path, encoding="utf-8")
            if len(edge_df) > 5:
                emb, _ = node2vec_embeddings(
                    edge_df[src_col].astype(str), edge_df[dst_col].astype(str),
                    dim=max(4, cfg.embedding_dim // 2), walk_length=7, num_walks=10,
                    p=0.3, q=0.7, directed=True,
                )
            else:
                emb = {fid: np.zeros(max(4, cfg.embedding_dim // 2), dtype=np.float32) for fid in df["row_id"]}
        else:
            emb = {fid: np.zeros(max(4, cfg.embedding_dim // 2), dtype=np.float32) for fid in df["row_id"]}
        flow_emb = np.vstack([emb.get(fid, np.zeros(max(4, cfg.embedding_dim // 2), dtype=np.float32)) for fid in df["row_id"]])
        tcg_embeddings.append(flow_emb)

    tcg_combined = np.hstack(tcg_embeddings)

    # Structural features
    degree_counts = pd.concat([df["src_endpoint"], df["dst_endpoint"]]).value_counts()
    out_counts = df["src_endpoint"].value_counts()
    in_counts = df["dst_endpoint"].value_counts()
    structural = pd.DataFrame(
        {
            "src_degree": df["src_endpoint"].map(degree_counts).fillna(0).to_numpy(),
            "dst_degree": df["dst_endpoint"].map(degree_counts).fillna(0).to_numpy(),
            "src_out_degree": df["src_endpoint"].map(out_counts).fillna(0).to_numpy(),
            "dst_in_degree": df["dst_endpoint"].map(in_counts).fillna(0).to_numpy(),
        }
    )

    # Also add weighted degree features
    flow_bytes = pd.to_numeric(df["Total.Length.of.Fwd.Packets"], errors="coerce").fillna(0) + \
                 pd.to_numeric(df["Total.Length.of.Bwd.Packets"], errors="coerce").fillna(0)
    src_byte_deg = flow_bytes.groupby(df["src_endpoint"]).sum()
    dst_byte_deg = flow_bytes.groupby(df["dst_endpoint"]).sum()
    structural["src_byte_degree"] = df["src_endpoint"].map(src_byte_deg).fillna(0).to_numpy()
    structural["dst_byte_degree"] = df["dst_endpoint"].map(dst_byte_deg).fillna(0).to_numpy()

    # Assemble all graph features
    graph = pd.DataFrame(hcg_edge_emb, columns=[f"hcg_e{i}" for i in range(hcg_edge_emb.shape[1])])
    for ci, emb_arr in enumerate(tcg_embeddings):
        for j in range(emb_arr.shape[1]):
            graph[f"tcg_{['CR','PR','DHR','SHR'][ci]}_{j}"] = emb_arr[:, j]
    graph = pd.concat([graph, structural.reset_index(drop=True)], axis=1)
    return graph


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden = min(512, max(128, in_dim * 3))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    cfg: ExperimentConfig,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SimpleTensorBoardWriter(RUNS_DIR / "network_traffic_mlp")
    (RUNS_DIR / "network_traffic_mlp" / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    model = MLP(x_train.shape[1], len(class_names)).to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1), dtype=torch.float32, device=device
    )
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-5)

    best_f1 = -math.inf
    best_state = None
    patience = max(5, cfg.epochs // 4)
    no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)

        scheduler.step()

        metrics, _ = eval_torch(model, x_test, y_test, device)
        train_loss = total_loss / len(train_ds)
        lr = opt.param_groups[0]["lr"]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metrics/test_accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("metrics/test_f1_weighted", metrics["f1_weighted"], epoch)
        writer.add_scalar("metrics/learning_rate", lr, epoch)
        print(
            f"epoch {epoch:02d}/{cfg.epochs} "
            f"loss={train_loss:.4f} acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f} lr={lr:.2e}"
        )
        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, y_pred = eval_torch(model, x_test, y_test, device)
    writer.close()
    torch.save(
        {"state_dict": model.state_dict(), "class_names": class_names, "input_dim": x_train.shape[1]},
        OUTPUT_DIR / "mlp_model.pt",
    )
    write_classification_artifacts("MLP_Torch", y_test, y_pred, class_names)
    return metrics


def eval_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return metric_dict(y, pred), pred


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
    }


def write_classification_artifacts(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / f"{name}_classification_report.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def train_sklearn_models(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, class_names: list[str]
) -> dict[str, dict[str, float]]:
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=3, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=160, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics = metric_dict(y_test, pred)
        results[name] = metrics
        joblib.dump(model, OUTPUT_DIR / f"{name}.joblib")
        write_classification_artifacts(name, y_test, pred, class_names)
        print(f"{name}: {metrics}")
    return results


def write_tugraph_config() -> None:
    import_config = {
        "schema": [
            {
                "label": "Endpoint",
                "type": "VERTEX",
                "primary": "endpoint_id",
                "properties": [
                    {"name": "endpoint_id", "type": "STRING"},
                    {"name": "ip", "type": "STRING"},
                    {"name": "port", "type": "INT32"},
                ],
            },
            {
                "label": "Flow",
                "type": "VERTEX",
                "primary": "flow_id",
                "properties": [
                    {"name": "flow_id", "type": "STRING"},
                    {"name": "src_endpoint", "type": "STRING"},
                    {"name": "dst_endpoint", "type": "STRING"},
                    {"name": "protocol_name", "type": "STRING"},
                    {"name": "l7_protocol", "type": "INT32"},
                    {"name": "flow_duration", "type": "DOUBLE"},
                    {"name": "total_packets", "type": "DOUBLE"},
                    {"name": "total_bytes", "type": "DOUBLE"},
                    {"name": "timestamp", "type": "STRING"},
                ],
            },
            {
                "label": "COMMUNICATES",
                "type": "EDGE",
                "constraints": [["Endpoint", "Endpoint"]],
                "properties": [
                    {"name": "flow_id", "type": "STRING"},
                    {"name": "protocol", "type": "INT32"},
                    {"name": "protocol_name", "type": "STRING"},
                    {"name": "l7_protocol", "type": "INT32"},
                    {"name": "flow_duration", "type": "DOUBLE"},
                    {"name": "total_packets", "type": "DOUBLE"},
                    {"name": "total_bytes", "type": "DOUBLE"},
                    {"name": "timestamp", "type": "STRING"},
                ],
            },
            {
                "label": "CAUSAL_CR",
                "type": "EDGE",
                "constraints": [["Flow", "Flow"]],
                "properties": [
                    {"name": "src_ip", "type": "STRING"},
                    {"name": "src_port", "type": "INT32"},
                    {"name": "dst_ip", "type": "STRING"},
                    {"name": "dst_port", "type": "INT32"},
                    {"name": "protocol", "type": "INT32"},
                    {"name": "delta_seconds", "type": "DOUBLE"},
                ],
            },
            {
                "label": "CAUSAL_PR",
                "type": "EDGE",
                "constraints": [["Flow", "Flow"]],
                "properties": [
                    {"name": "shared_ip", "type": "STRING"},
                    {"name": "delta_seconds", "type": "DOUBLE"},
                ],
            },
            {
                "label": "CAUSAL_DHR",
                "type": "EDGE",
                "constraints": [["Flow", "Flow"]],
                "properties": [
                    {"name": "shared_ip", "type": "STRING"},
                    {"name": "src_port_f1", "type": "INT32"},
                    {"name": "src_port_f2", "type": "INT32"},
                    {"name": "delta_seconds", "type": "DOUBLE"},
                ],
            },
            {
                "label": "CAUSAL_SHR",
                "type": "EDGE",
                "constraints": [["Flow", "Flow"]],
                "properties": [
                    {"name": "shared_ip", "type": "STRING"},
                    {"name": "shared_port", "type": "INT32"},
                    {"name": "delta_seconds", "type": "DOUBLE"},
                ],
            },
        ],
        "files": [
            {
                "path": "/data/traffic_import/hcg_vertices_endpoint.csv",
                "format": "CSV",
                "label": "Endpoint",
                "header": 1,
                "columns": ["endpoint_id", "ip", "port"],
            },
            {
                "path": "/data/traffic_import/hcg_edges_communicates.csv",
                "format": "CSV",
                "label": "COMMUNICATES",
                "header": 1,
                "SRC_ID": "Endpoint",
                "DST_ID": "Endpoint",
                "columns": [
                    "SRC_ID",
                    "DST_ID",
                    "flow_id",
                    "protocol",
                    "protocol_name",
                    "l7_protocol",
                    "flow_duration",
                    "total_packets",
                    "total_bytes",
                    "timestamp",
                ],
            },
            {
                "path": "/data/traffic_import/tcg_vertices_flow.csv",
                "format": "CSV",
                "label": "Flow",
                "header": 1,
                "columns": [
                    "flow_id",
                    "src_endpoint",
                    "dst_endpoint",
                    "protocol_name",
                    "l7_protocol",
                    "flow_duration",
                    "total_packets",
                    "total_bytes",
                    "timestamp",
                ],
            },
            {
                "path": "/data/traffic_import/tcg_edges_CR.csv",
                "format": "CSV",
                "label": "CAUSAL_CR",
                "header": 1,
                "SRC_ID": "Flow",
                "DST_ID": "Flow",
                "columns": ["SRC_ID", "DST_ID", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "delta_seconds"],
            },
            {
                "path": "/data/traffic_import/tcg_edges_PR.csv",
                "format": "CSV",
                "label": "CAUSAL_PR",
                "header": 1,
                "SRC_ID": "Flow",
                "DST_ID": "Flow",
                "columns": ["SRC_ID", "DST_ID", "shared_ip", "delta_seconds"],
            },
            {
                "path": "/data/traffic_import/tcg_edges_DHR.csv",
                "format": "CSV",
                "label": "CAUSAL_DHR",
                "header": 1,
                "SRC_ID": "Flow",
                "DST_ID": "Flow",
                "columns": ["SRC_ID", "DST_ID", "shared_ip", "src_port_f1", "src_port_f2", "delta_seconds"],
            },
            {
                "path": "/data/traffic_import/tcg_edges_SHR.csv",
                "format": "CSV",
                "label": "CAUSAL_SHR",
                "header": 1,
                "SRC_ID": "Flow",
                "DST_ID": "Flow",
                "columns": ["SRC_ID", "DST_ID", "shared_ip", "shared_port", "delta_seconds"],
            },
        ],
    }
    notes = {
        "description": "TuGraph import config for network traffic HCG and TCG graphs.",
        "import_command": "lgraph_import -c /data/traffic_import/import.json -d /tmp/traffic_lgraph --overwrite 1",
        "notes": [
            "HCG: Endpoint vertices are {IP, port}; COMMUNICATES edges are original traffic flows.",
            "TCG: Flow vertices are original flows. Four edge types per Liu Zhen paper:",
            "  CR (Communication Relationship): bidirectional counterpart flows",
            "  PR (Propagation Relationship): dstIp(f1) = srcIp(f2)",
            "  DHR (Dynamic-port Host Relationship): same IP, different ports",
            "  SHR (Static-port Host Relationship): same IP, same port",
            "Node2Vec embeddings: p=0.3, q=0.7, walk_length=7 per Su Yongcai paper.",
        ],
    }
    (IMPORT_DIR / "import.json").write_text(json.dumps(import_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (IMPORT_DIR / "tugraph_schema_reference.json").write_text(
        json.dumps({"reference": notes, **import_config}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_cypher() -> None:
    cypher = """// TuGraph Browser / lgraph_cli sanity checks
MATCH (n:Endpoint) RETURN count(n) AS endpoints;
MATCH ()-[e:COMMUNICATES]->() RETURN count(e) AS hcg_edges;
MATCH (f:Flow) RETURN count(f) AS flows;

// TCG edge counts by type
MATCH ()-[e:CAUSAL_CR]->() RETURN 'CR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_PR]->() RETURN 'PR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_DHR]->() RETURN 'DHR' AS type, count(e) AS count
UNION ALL
MATCH ()-[e:CAUSAL_SHR]->() RETURN 'SHR' AS type, count(e) AS count;

// Top application protocols on HCG edges
MATCH ()-[e:COMMUNICATES]->()
RETURN e.protocol_name AS protocol, count(e) AS flows
ORDER BY flows DESC
LIMIT 10;

// High-degree endpoints
MATCH (n:Endpoint)-[e:COMMUNICATES]-()
RETURN n.endpoint_id AS endpoint, count(e) AS degree
ORDER BY degree DESC
LIMIT 20;

// Example CR edges (bidirectional communication)
MATCH (a:Flow)-[e:CAUSAL_CR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.src_ip AS src_ip, e.dst_ip AS dst_ip, e.delta_seconds AS delta
LIMIT 10;

// Example PR edges (propagation chain)
MATCH (a:Flow)-[e:CAUSAL_PR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.shared_ip AS shared_ip, e.delta_seconds AS delta
LIMIT 10;

// SHR edges (same IP+port, potential port scanning)
MATCH (a:Flow)-[e:CAUSAL_SHR]->(b:Flow)
RETURN a.flow_id AS src, b.flow_id AS dst, e.shared_ip AS ip, e.shared_port AS port, e.delta_seconds AS delta
LIMIT 10;
"""
    (IMPORT_DIR / "sanity_checks.cypher").write_text(cypher, encoding="utf-8")


def write_report(cfg: ExperimentConfig, df: pd.DataFrame, results: dict[str, dict[str, float]]) -> None:
    rows = []
    for name, metrics in results.items():
        rows.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision_weighted']:.4f} | "
            f"{metrics['recall_weighted']:.4f} | {metrics['f1_weighted']:.4f} |"
        )
    class_dist = df["ProtocolName"].value_counts().to_frame("count").to_markdown()
    metrics_table = "\n".join(rows)
    report = f"""# 安全通论实验 3：网络流量分类

## 1. 数据集

- 数据集：IP Network Traffic Flows Labeled with 75 Apps / 87 attributes
- 本地文件：`Dataset-Unicauca-Version2-87Atts.csv/Dataset-Unicauca-Version2-87Atts.csv`
- 原始字段：87 列，包含五元组、流持续时间、包长统计、IAT、TCP flag、吞吐量、应用层协议等。
- 本次实验抽样：先扫描 {cfg.scan_rows:,} 行，选择出现次数最多的 {cfg.top_classes} 个应用类别，每类最多 {cfg.samples_per_class} 条。

类别分布：

{class_dist}

## 2. 论文依据与图建模

参考《互联网流量分类中流量特征研究_刘珍》对流量特征和连接图的讨论，本实验实现两种图建模：

- HCG：将 `{IP, port}` 二元组建模为 `Endpoint` 顶点；如果两个端点之间存在一条流，则建立 `COMMUNICATES` 边。
- TCG：将原始流建模为 `Flow` 顶点；按照刘珍论文定义四种流间因果关系边：
  - **CR**（Communication Relationship）：双向通信关系，`protocol(f1)=protocol(f2) ∧ src↔dst, dst↔src`
  - **PR**（Propagation Relationship）：传播关系，`dstIp(f1) = srcIp(f2)`，信息经中间主机转发
  - **DHR**（Dynamic-Port Host Relationship）：动态端口关系，`srcIp相同 ∧ srcPort不同`
  - **SHR**（Static-Port Host Relationship）：静态端口关系，`srcIp相同 ∧ srcPort相同`（如端口扫描）

参考《基于时空图神经网络的网络异常检测与流量分类_苏永才》第 4 章（Node2Vec，p=0.3, q=0.7, walk_length=7）进行图嵌入：

- HCG 节点嵌入：Node2Vec 对端点通信图做无偏随机游走嵌入（dim={cfg.embedding_dim}）
- HCG 边嵌入：拼接源/目的端点嵌入 + 绝对差 + 哈达玛积（Hadamard product）
- TCG 流嵌入：对 4 种边类型分别独立做 Node2Vec 有向图嵌入，然后拼接
- 结构特征：源/目的端点总度、出度、入度、加权字节度

TuGraph 导入文件已生成在 `tugraph_import/`：

- `hcg_vertices_endpoint.csv`
- `hcg_edges_communicates.csv`
- `tcg_vertices_flow.csv`
- `tcg_edges_causal.csv`
- `tugraph_schema_reference.json`
- `sanity_checks.cypher`

## 3. 分类器与评价指标

训练/测试集按 stratified split 划分，测试比例为 {cfg.test_size}。评价指标包括 Accuracy、weighted Precision、weighted Recall、weighted F1。

| 分类器 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{metrics_table}

## 4. TensorBoard 监控

PyTorch MLP 的训练过程已写入：

```powershell
tensorboard --logdir runs
```

或使用本机 Python：

```powershell
& D:\\Python313\\python.exe -m tensorboard.main --logdir runs
```

打开 TensorBoard 后可查看 `loss/train`、`metrics/test_accuracy`、`metrics/test_f1_weighted` 和 hparams。

## 5. 复现实验命令

```powershell
& D:\\Python313\\python.exe scripts\\run_experiment.py --scan-rows {cfg.scan_rows} --top-classes {cfg.top_classes} --samples-per-class {cfg.samples_per_class} --epochs {cfg.epochs}
```

## 6. 结果文件

- 分类报告：`outputs/*_classification_report.csv`
- 混淆矩阵：`outputs/*_confusion_matrix.png`
- 模型：`outputs/*.joblib`、`outputs/mlp_model.pt`
- 融合特征：`outputs/fused_features_sample.csv`
"""
    (REPORT_DIR / "实验3-网络流量分类报告.md").write_text(report, encoding="utf-8")


def plot_metric_summary(results: dict[str, dict[str, float]]) -> None:
    names = list(results)
    f1 = [results[n]["f1_weighted"] for n in names]
    acc = [results[n]["accuracy"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, acc, width=0.36, label="Accuracy")
    ax.bar(x + 0.18, f1, width=0.36, label="Weighted F1")
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "classifier_metric_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Network traffic classification with TuGraph-ready graph modeling.")
    parser.add_argument("--scan-rows", type=int, default=300_000)
    parser.add_argument("--top-classes", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=4_000)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--causal-window-seconds", type=int, default=60)
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))

    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    IMPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_balanced_sample(cfg)
    export_hcg(df)
    export_tcg(df, cfg.causal_window_seconds)
    write_tugraph_config()
    write_cypher()

    raw = safe_numeric_frame(df, target_col="ProtocolName")
    graph = build_graph_features(df, cfg)
    fused = pd.concat([raw.reset_index(drop=True), graph.reset_index(drop=True)], axis=1)
    fused = fused.replace([np.inf, -np.inf], np.nan).fillna(0)
    fused.to_csv(OUTPUT_DIR / "fused_features_sample.csv", index=False, encoding="utf-8")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ProtocolName"])
    class_names = label_encoder.classes_.tolist()
    scaler = StandardScaler()
    x = scaler.fit_transform(fused).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    joblib.dump({"scaler": scaler, "label_encoder": label_encoder, "columns": fused.columns.tolist()}, OUTPUT_DIR / "preprocess.joblib")

    results = train_sklearn_models(x_train, x_test, y_train, y_test, class_names)
    results["MLP_Torch"] = train_mlp(x_train, x_test, y_train, y_test, class_names, cfg)
    pd.DataFrame(results).T.to_csv(OUTPUT_DIR / "metrics_summary.csv", encoding="utf-8-sig")
    (OUTPUT_DIR / "metrics_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_metric_summary(results)
    write_report(cfg, df, results)

    print("\nDone.")
    print(f"TuGraph import files: {IMPORT_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"TensorBoard logs: {RUNS_DIR}")
    print(f"Report: {REPORT_DIR / '实验3-网络流量分类报告.md'}")


if __name__ == "__main__":
    main()
