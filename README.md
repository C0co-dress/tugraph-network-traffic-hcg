# 安全通论实验 3：网络流量分类

本目录包含基于 `IP Network Traffic Flows Labeled with 75 Apps` 数据集的网络流量分类实验。

核心流水线：

1. 从原始 CSV 抽样应用层协议类别。
2. 生成两类 TuGraph 图数据：HCG 与 TCG。
3. 基于图邻接矩阵生成节点/边嵌入，并与原始流统计特征融合。
4. 训练 Decision Tree、KNN、Random Forest 和 PyTorch MLP。
5. 使用 TensorBoard 监控 MLP 训练进度。

运行：

```powershell
.\scripts\run.ps1
```

或手动指定参数：

```powershell
.\scripts\run.ps1 --scan-rows 500000 --epochs 20
```

中文用户名路径说明：

如果路径含中文（如 `C:\Users\林椰椰\...`），可能导致 TensorBoard / Docker 异常。两种处理方式：

- **推荐**：设置 `TUGRAPH3_ROOT` 环境变量指向无中文的目录，所有输出写入该目录。
- **备用**：使用 `run.ps1` 启动（自动启用 Python UTF-8 模式）。

```powershell
$env:TUGRAPH3_ROOT = "D:\tugraph3_work"
.\scripts\run.ps1
```

查看 TensorBoard：

```powershell
.\scripts\start_tensorboard.ps1
```

TuGraph 文件在 `tugraph_import/`。如果 Docker 可用，可尝试：

```powershell
.\scripts\run_tugraph_docker.ps1
```
