# PPT 步骤完成核对

依据 `安全通论-实验3-网络流量分类.pptx` 第 3 页逐项核对。

| PPT 要求 | 完成情况 | 对应材料 |
|---|---|---|
| 1. 数据集的介绍、下载、展示 | 已完成。数据集已下载到本地，正式实验抽样 50000 条流，覆盖 10 个应用类别。 | `Dataset-Unicauca-Version2-87Atts.csv/`、`outputs/dataset_protocol_distribution.png`、`reports/实验3-网络流量分类报告.md` |
| 2. 两种流量图建模方法，提取字段，可视化或导入 TuGraph | 已完成。HCG 使用 `{IP, port}` 作为 Endpoint 顶点，TCG 使用 Flow 作为顶点并建立 CR/PR/DHR/SHR 四种因果边（刘珍论文）；正式 TuGraph 离线导入成功。 | `tugraph_import/import.json`、`tugraph_import/*.csv`、`outputs/hcg_graph_preview.png`、`outputs/tcg_causal_preview.png` |
| 3. 点嵌入/边嵌入、特征融合，文字/代码/截图描述 | 已完成。HCG 端点嵌入、HCG 边嵌入、TCG 流嵌入和结构度特征已与原始数值特征融合。 | `scripts/run_experiment.py`、`outputs/fused_features_sample.csv`、`reports/实验3-网络流量分类报告.md` |
| 4. 至少 3 种分类器，给出评价指标 | 已完成。已训练 Decision Tree、KNN、Random Forest、PyTorch MLP 四种模型。最佳 weighted F1 为 `RandomForest` 的 0.7875。 | `outputs/metrics_summary.csv`、`outputs/*classification_report.csv`、`outputs/*confusion_matrix.png` |
| 5. 通过 GitHub 提交报告，把链接放到学习通 | 提交材料已准备好，但当前目录不是 Git 仓库，尚未生成真实 GitHub 链接。 | `README.md`、`reports/实验3-网络流量分类报告.md`、全部代码和输出文件 |

## TuGraph 导入规模

- HCG Endpoint 顶点：43783
- HCG COMMUNICATES 边：50000
- TCG Flow 顶点：50000
- TCG 因果边 (CR+PR+DHR+SHR)：65959

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
