# HCG 本地复现实验说明

本仓库已经把 HCG 图网络流量分类实验所需的关键文件本地化，便于推送到个人 Git 仓库后复现和检查。

## 本地代码入口

| 路径 | 说明 |
|---|---|
| `scripts/run_hcg_local.py` | HCG-only 本地复现入口，构建 A/B/C 特征、训练 3 个分类器、写 TensorBoard |
| `scripts/run_experiment.py` | 完整实验流水线，包含 HCG/TCG 构图、Node2Vec、分类器和 TensorBoard writer |
| `scripts/start_tensorboard.ps1` | Windows 下查看 `runs/` 的 TensorBoard 启动脚本 |
| `reports/安全通论实验3-HCG图网络流量分类实验报告.md` | HCG 实验报告 Markdown |
| `reports/林源卿2023312350安全通论实验报告3-HCG图网络流量分类.docx` | HCG 实验报告 Word 版 |

## 已保留的小型运行产物

`.gitignore` 已调整为允许提交 HCG 所需的小型证明文件：

| 路径 | 说明 |
|---|---|
| `tugraph_import/hcg_vertices_endpoint.csv` | HCG Endpoint 顶点表 |
| `tugraph_import/hcg_edges_communicates.csv` | HCG COMMUNICATES 边表 |
| `tugraph_import/import.json` | TuGraph 导入配置 |
| `tugraph_import/sanity_checks.cypher` | TuGraph 查询检查脚本 |
| `runs/network_traffic_mlp/` | TensorBoard 训练日志 |
| `outputs/*_confusion_matrix.png` | 分类器混淆矩阵图 |
| `outputs/classifier_metric_summary.png` | 分类器指标图 |

大文件仍然不建议提交，包括原始数据集、模型 `.joblib`、`outputs/fused_features_sample.csv` 等。

## 复现命令

先安装依赖：

```powershell
pip install -r requirements.txt
```

运行 HCG-only 本地实验：

```powershell
python scripts\run_hcg_local.py --scan-rows 600000 --top-classes 10 --samples-per-class 5000
```

查看 TensorBoard：

```powershell
.\scripts\start_tensorboard.ps1
```

如果中文用户名路径导致 TensorBoard 读取异常，可设置一个无中文工作目录：

```powershell
$env:TUGRAPH3_ROOT = "D:\tugraph3_work"
python scripts\run_hcg_local.py
```
