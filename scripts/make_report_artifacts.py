from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = ROOT / "tugraph_import"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"


def plot_dataset_distribution() -> None:
    flows = pd.read_csv(IMPORT_DIR / "tcg_vertices_flow.csv")
    counts = flows["protocol_name"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values, color="#3b82f6")
    ax.set_xlabel("flow samples")
    ax.set_title("Application protocol distribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dataset_protocol_distribution.png", dpi=180)
    plt.close(fig)


def plot_hcg_preview() -> None:
    edges = pd.read_csv(IMPORT_DIR / "hcg_edges_communicates.csv")
    top_nodes = (
        pd.concat([edges["src_endpoint"], edges["dst_endpoint"]])
        .value_counts()
        .head(24)
        .index
        .tolist()
    )
    sub = edges[edges["src_endpoint"].isin(top_nodes) & edges["dst_endpoint"].isin(top_nodes)].head(80)
    coords = {}
    for i, node in enumerate(top_nodes):
        theta = 2 * math.pi * i / max(1, len(top_nodes))
        coords[node] = (math.cos(theta), math.sin(theta))

    fig, ax = plt.subplots(figsize=(8, 8))
    for row in sub.itertuples(index=False):
        x1, y1 = coords[row.src_endpoint]
        x2, y2 = coords[row.dst_endpoint]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=0.8, alpha=0.55)
    for node, (x, y) in coords.items():
        ax.scatter([x], [y], s=90, color="#ef4444", zorder=3)
        ax.text(x * 1.08, y * 1.08, node, fontsize=7, ha="center", va="center")
    ax.set_title("HCG preview: Endpoint vertices and COMMUNICATES edges")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hcg_graph_preview.png", dpi=180)
    plt.close(fig)


def plot_tcg_preview() -> None:
    edge_files = ["tcg_edges_CR.csv", "tcg_edges_PR.csv", "tcg_edges_DHR.csv", "tcg_edges_SHR.csv"]
    all_edges = []
    for fname in edge_files:
        path = IMPORT_DIR / fname
        if path.exists():
            df_edge = pd.read_csv(path)
            if len(df_edge) > 0:
                all_edges.append(df_edge.head(30))
    if not all_edges:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.text(0.5, 0.5, "No TCG edges found", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(OUTPUT_DIR / "tcg_causal_preview.png", dpi=180)
        plt.close(fig)
        return

    edges = pd.concat(all_edges, ignore_index=True).head(120)
    flows = sorted(set(edges["src_flow"]) | set(edges["dst_flow"]))
    flows = flows[:50]
    idx = {flow: i for i, flow in enumerate(flows)}
    sub = edges[edges["src_flow"].isin(flows) & edges["dst_flow"].isin(flows)]

    edge_type_colors = {"CR": "#ef4444", "PR": "#3b82f6", "DHR": "#f59e0b", "SHR": "#10b981"}

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for flow, i in idx.items():
        ax.scatter([i], [0], s=35, color="#10b981")
    for row in sub.itertuples(index=False):
        x1, x2 = idx[row.src_flow], idx[row.dst_flow]
        height = 0.2 + min(1.8, abs(x2 - x1) * 0.06)
        ax.plot([x1, (x1 + x2) / 2, x2], [0, height, 0], color="#64748b", linewidth=0.8, alpha=0.7)
    ax.set_title("TCG preview: Flow vertices and causal edges (CR/PR/DHR/SHR)")
    ax.set_xlabel("sampled flow order")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tcg_causal_preview.png", dpi=180)
    plt.close(fig)


def write_checklist() -> None:
    metrics = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv", index_col=0)
    flows = pd.read_csv(IMPORT_DIR / "tcg_vertices_flow.csv")
    hcg_vertices = sum(1 for _ in open(IMPORT_DIR / "hcg_vertices_endpoint.csv", encoding="utf-8")) - 1
    hcg_edges = sum(1 for _ in open(IMPORT_DIR / "hcg_edges_communicates.csv", encoding="utf-8")) - 1
    tcg_vertices = sum(1 for _ in open(IMPORT_DIR / "tcg_vertices_flow.csv", encoding="utf-8")) - 1
    tcg_edges = 0
    for fname in ["tcg_edges_CR.csv", "tcg_edges_PR.csv", "tcg_edges_DHR.csv", "tcg_edges_SHR.csv"]:
        path = IMPORT_DIR / fname
        if path.exists():
            tcg_edges += sum(1 for _ in open(path, encoding="utf-8")) - 1
    best = metrics["f1_weighted"].idxmax()
    text = f"""# PPT 步骤完成核对

依据 `安全通论-实验3-网络流量分类.pptx` 第 3 页逐项核对。

| PPT 要求 | 完成情况 | 对应材料 |
|---|---|---|
| 1. 数据集的介绍、下载、展示 | 已完成。数据集已下载到本地，正式实验抽样 {len(flows)} 条流，覆盖 {flows['protocol_name'].nunique()} 个应用类别。 | `Dataset-Unicauca-Version2-87Atts.csv/`、`outputs/dataset_protocol_distribution.png`、`reports/实验3-网络流量分类报告.md` |
| 2. 两种流量图建模方法，提取字段，可视化或导入 TuGraph | 已完成。HCG 使用 `{{IP, port}}` 作为 Endpoint 顶点，TCG 使用 Flow 作为顶点并建立 CR/PR/DHR/SHR 四种因果边（刘珍论文）；正式 TuGraph 离线导入成功。 | `tugraph_import/import.json`、`tugraph_import/*.csv`、`outputs/hcg_graph_preview.png`、`outputs/tcg_causal_preview.png` |
| 3. 点嵌入/边嵌入、特征融合，文字/代码/截图描述 | 已完成。HCG 端点嵌入、HCG 边嵌入、TCG 流嵌入和结构度特征已与原始数值特征融合。 | `scripts/run_experiment.py`、`outputs/fused_features_sample.csv`、`reports/实验3-网络流量分类报告.md` |
| 4. 至少 3 种分类器，给出评价指标 | 已完成。已训练 Decision Tree、KNN、Random Forest、PyTorch MLP 四种模型。最佳 weighted F1 为 `{best}` 的 {metrics.loc[best, 'f1_weighted']:.4f}。 | `outputs/metrics_summary.csv`、`outputs/*classification_report.csv`、`outputs/*confusion_matrix.png` |
| 5. 通过 GitHub 提交报告，把链接放到学习通 | 提交材料已准备好，但当前目录不是 Git 仓库，尚未生成真实 GitHub 链接。 | `README.md`、`reports/实验3-网络流量分类报告.md`、全部代码和输出文件 |

## TuGraph 导入规模

- HCG Endpoint 顶点：{hcg_vertices}
- HCG COMMUNICATES 边：{hcg_edges}
- TCG Flow 顶点：{tcg_vertices}
- TCG 因果边 (CR+PR+DHR+SHR)：{tcg_edges}

## TensorBoard

训练进度已写入 `runs/network_traffic_mlp/events.out.tfevents.*`，记录了 `loss/train`、`metrics/test_accuracy` 和 `metrics/test_f1_weighted`。

## GitHub 提交建议

当前还缺一个真实仓库地址。创建仓库后执行：

```powershell
git init
git add README.md requirements.txt scripts configs reports tugraph_import outputs
git commit -m "Complete network traffic classification lab"
git branch -M main
git remote add origin <你的 GitHub 仓库 URL>
git push -u origin main
```
"""
    (REPORT_DIR / "PPT步骤完成核对.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    plot_dataset_distribution()
    plot_hcg_preview()
    plot_tcg_preview()
    write_checklist()
    print("Report artifacts generated.")


if __name__ == "__main__":
    main()
