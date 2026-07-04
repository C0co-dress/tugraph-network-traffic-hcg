from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"

joblib = None
np = None
pd = None
SGDClassifier = None
accuracy_score = None
f1_score = None
precision_recall_fscore_support = None
train_test_split = None
KNeighborsClassifier = None
LabelEncoder = None
StandardScaler = None
DecisionTreeClassifier = None


def load_ml_dependencies() -> None:
    global joblib, np, pd, SGDClassifier, accuracy_score, f1_score
    global precision_recall_fscore_support, train_test_split, KNeighborsClassifier
    global LabelEncoder, StandardScaler, DecisionTreeClassifier

    import joblib as _joblib
    import numpy as _np
    import pandas as _pd
    from sklearn.linear_model import SGDClassifier as _SGDClassifier
    from sklearn.metrics import (
        accuracy_score as _accuracy_score,
        f1_score as _f1_score,
        precision_recall_fscore_support as _precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split as _train_test_split
    from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder, StandardScaler as _StandardScaler
    from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier

    joblib = _joblib
    np = _np
    pd = _pd
    SGDClassifier = _SGDClassifier
    accuracy_score = _accuracy_score
    f1_score = _f1_score
    precision_recall_fscore_support = _precision_recall_fscore_support
    train_test_split = _train_test_split
    KNeighborsClassifier = _KNeighborsClassifier
    LabelEncoder = _LabelEncoder
    StandardScaler = _StandardScaler
    DecisionTreeClassifier = _DecisionTreeClassifier


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("run_experiment_pipeline", RUN_EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUN_EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local HCG-only reproduction entrypoint.")
    parser.add_argument("--scan-rows", type=int, default=600_000)
    parser.add_argument("--top-classes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=5_000)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "hcg_local")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "hcg_local")
    return parser.parse_args()


def build_hcg_features(df: pd.DataFrame, cfg, pipeline) -> pd.DataFrame:
    endpoint_emb, _ = pipeline.node2vec_embeddings(
        df["src_endpoint"],
        df["dst_endpoint"],
        dim=cfg.embedding_dim,
        walk_length=7,
        num_walks=10,
        p=0.3,
        q=0.7,
        directed=False,
    )
    src_hcg = np.vstack([endpoint_emb[e] for e in df["src_endpoint"]])
    dst_hcg = np.vstack([endpoint_emb[e] for e in df["dst_endpoint"]])
    hcg_edge_emb = np.hstack([src_hcg, dst_hcg, np.abs(src_hcg - dst_hcg), src_hcg * dst_hcg])

    degree_counts = pd.concat([df["src_endpoint"], df["dst_endpoint"]]).value_counts()
    out_counts = df["src_endpoint"].value_counts()
    in_counts = df["dst_endpoint"].value_counts()
    flow_bytes = (
        pd.to_numeric(df["Total.Length.of.Fwd.Packets"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Total.Length.of.Bwd.Packets"], errors="coerce").fillna(0)
    )
    src_byte_deg = flow_bytes.groupby(df["src_endpoint"]).sum()
    dst_byte_deg = flow_bytes.groupby(df["dst_endpoint"]).sum()

    structural = pd.DataFrame(
        {
            "src_degree": df["src_endpoint"].map(degree_counts).fillna(0).to_numpy(),
            "dst_degree": df["dst_endpoint"].map(degree_counts).fillna(0).to_numpy(),
            "src_out_degree": df["src_endpoint"].map(out_counts).fillna(0).to_numpy(),
            "dst_in_degree": df["dst_endpoint"].map(in_counts).fillna(0).to_numpy(),
            "src_byte_degree": df["src_endpoint"].map(src_byte_deg).fillna(0).to_numpy(),
            "dst_byte_degree": df["dst_endpoint"].map(dst_byte_deg).fillna(0).to_numpy(),
        }
    )

    hcg = pd.DataFrame(hcg_edge_emb, columns=[f"hcg_e{i}" for i in range(hcg_edge_emb.shape[1])])
    return pd.concat([hcg, structural.reset_index(drop=True)], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(weighted_f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_group(name: str, features: pd.DataFrame, y: np.ndarray, args: argparse.Namespace, writer) -> list[dict[str, object]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(dtype=np.float32),
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models = {
        "logistic_sgd": SGDClassifier(loss="log_loss", max_iter=1000, random_state=args.random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=3, random_state=args.random_state),
        "knn_sample": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    }

    rows: list[dict[str, object]] = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        row = {"feature_group": name, "model": model_name, **metrics(y_test, pred)}
        rows.append(row)
        writer.add_scalar(f"{name}/{model_name}/accuracy", row["accuracy"], 0)
        writer.add_scalar(f"{name}/{model_name}/weighted_f1", row["f1_weighted"], 0)
        writer.add_scalar(f"{name}/{model_name}/macro_f1", row["macro_f1"], 0)
    return rows


def main() -> None:
    args = parse_args()
    load_ml_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline_module()
    cfg = pipeline.ExperimentConfig(
        scan_rows=args.scan_rows,
        top_classes=args.top_classes,
        samples_per_class=args.samples_per_class,
        embedding_dim=args.embedding_dim,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        causal_window_seconds=60,
    )

    df = pipeline.load_balanced_sample(cfg)
    pipeline.IMPORT_DIR.mkdir(exist_ok=True)
    pipeline.export_hcg(df)
    pipeline.write_tugraph_config()
    pipeline.write_cypher()

    raw = pipeline.safe_numeric_frame(df, target_col="ProtocolName").reset_index(drop=True)
    hcg = build_hcg_features(df, cfg, pipeline).reset_index(drop=True)
    groups = {
        "A_raw": raw,
        "B_hcg": hcg,
        "C_raw_hcg": pd.concat([raw, hcg], axis=1),
    }
    for group_name, group_df in groups.items():
        group_df.to_parquet(args.output_dir / f"{group_name}.parquet", index=False)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["ProtocolName"])
    joblib.dump({"label_encoder": encoder, "groups": list(groups)}, args.output_dir / "preprocess_hcg_local.joblib")

    writer = pipeline.SimpleTensorBoardWriter(args.runs_dir)
    rows: list[dict[str, object]] = []
    try:
        writer.add_scalar("data/n_samples", len(df), 0)
        writer.add_scalar("data/n_classes", len(encoder.classes_), 0)
        for group_name, group_df in groups.items():
            writer.add_scalar(f"{group_name}/feature_count", group_df.shape[1], 0)
            rows.extend(train_group(group_name, group_df, y, args, writer))
    finally:
        writer.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "hcg_local_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "hcg_local_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"TuGraph HCG files: {pipeline.IMPORT_DIR}")
    print(f"Outputs: {args.output_dir}")
    print(f"TensorBoard logs: {args.runs_dir}")


if __name__ == "__main__":
    main()
