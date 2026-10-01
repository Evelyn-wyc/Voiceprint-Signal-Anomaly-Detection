# 正则参数敏感性检验

对应 R2.5：量化既定正则系数变化时，分量恢复和检测结果怎样变化。入口为 [regularization.py](regularization.py)，配置为 [configs/regularization.json](configs/regularization.json)。

## 1. 实验内容

每次只调整一个参数组，另外两组保持原值：

| 参数组 | 作用 | 检验倍数 |
|---|---|---|
| λ1、λ2 | 异常稀疏性与总变差 | 0.5、1、2 |
| η1、η2、η3 | 幅值和时长的变化惩罚 | 0.5、1、2 |
| ψ | 相邻片段相似性 | 0.5、1、2 |

三个组共享同一份 1 倍结果，共 **7 种配置**。组内参数保持原比例。原值如下：

| 数据 | λ1、λ2 | η1、η2、η3 | ψ | K | 更新次数 |
|---|---|---|---|---|---|
| 仿真 | 1.5、1 | 0.15、0.15、0.05 | 2 | 10 | 2000 |
| 真实 | 1.5、1 | 0.3、0.3、0.1 | 10 | 12 | 1500 |

- **仿真：**常规正向异常，种子 0、1、2，共 3 条已保存输入。
- **真实：**每份源录音取编号最小的片段，共 8 条。选择只读取录音标识和片段编号。
- 共 **11 条输入 × 7 种配置 = 77 份结果**，复用 `necessary_v1` 中 11 份默认结果，**新增 66 次拟合**。
- 模型、归档默认初始化、Adam 0.01、目标乘 10、迭代预算和评价规则均保持固定。该实验与周期初始化对照分别评价两种选择的影响。

保存逐输入结果、P/A、初末参数、目标曲线、c=2/3/4 的指标及连续 AP/AUROC。主阈值 c=3。仿真报告 P/A RMSE，真实子集报告每份录音的配对检测变化；两类数据分别汇总。F1/AP/AUROC 汇总使用含正标签输入，误报率保留全部输入；未定义指标计数与失败项均保留。

所有配置都会报告。本协议保留原系数作为参照，用于衡量其附近的敏感性。结果支持已测倍数和数据条件下的判断；历史选参过程及采集、标注资料另按事实说明。加权目标随系数变化，最终目标值逐次记录，不用于挑选参数。输出之间的 P/A 差异用于衡量变化；真实数据的分量真值仍未知。

## 2. 同步与预检

在服务器已建立的 `qpad-rebuttal` tmux 会话内运行：

```bash
cd ~/QPAD_rebuttal
conda activate windturbine
git fetch origin
git switch review
git pull --ff-only origin review
python rebuttal/regularization.py check \
  --source rebuttal/outputs/necessary_v1 \
  --device cuda:0
```

预期显示：`simulation_inputs: 3`、`real_inputs: 8`、`total_results: 77`、`reused_results: 11`、`new_fits: 66`。

`--source` 指向已有 `necessary_v1/`，其中须包含 `manifest.json`、`inputs/` 和 `fits/`。程序从这里读取输入及标签，不需要重新读取原始音频或提供 `--data-root`。预检核对输入/拟合哈希、源模型与评分源码、默认参数、初值、迭代预算和重新计算的指标；复用还核对 NumPy/PyTorch、GPU 型号及 CUDA 构建版本。

若实际目录不同，修改 `--source`。若运行环境改变导致复用检查失败，可在预检及正式运行命令中都加 `--refit-defaults`，仍读取同一批保存输入并重算全部 77 次。恢复运行时保持这个选项一致。

## 3. 运行

```bash
python -u rebuttal/regularization.py run \
  --source rebuttal/outputs/necessary_v1 \
  --device cuda:0 \
  --output rebuttal/outputs/regularization_v1
```

按相同 11 条输入的已记录 A100 耗时估算，66 次拟合约 **28 分钟**；可预留 **30–40 分钟**。共享 GPU 负载与参数变化会影响实际时间。程序逐项打印进度，结束时自动核验并生成报告。

中断后执行同一命令并加 `--resume`：

```bash
python -u rebuttal/regularization.py run \
  --source rebuttal/outputs/necessary_v1 \
  --device cuda:0 \
  --output rebuttal/outputs/regularization_v1 \
  --resume
```

续跑核对配置、代码、输入、来源及环境。已完成的有效结果跳过，失败项重新尝试，损坏的完成结果会明确报错。修改代码或配置后应使用新的输出目录。

退出码：0 表示全部完成并通过核验，1 表示存在失败或缺失结果，2 表示配置、来源、环境或续跑检查错误。

## 4. 查看与回传

查看：

```bash
cat ~/QPAD_rebuttal/rebuttal/outputs/regularization_v1/analysis/report.md
```

回传 **整个 `regularization_v1/` 文件夹**：

```text
regularization_v1/
├── manifest.json             # 配置、源码哈希、输入和复用来源、每次实际系数
├── config.json
├── inputs/*.npz              # 11 份输入、标签；仿真包含分量真值
├── fits/*.npz                # 77 份 P/A、初末参数、目标曲线及连续分数
├── fits/*.json               # 状态、系数、指标、耗时和哈希
├── events.jsonl
├── progress.json
└── analysis/
    ├── report.md
    ├── summary.json
    ├── per_fit.csv
    ├── by_setting.csv
    ├── paired_per_input.csv
    ├── metrics.csv
    ├── failures.csv
    └── figures/              # 仿真、真实子集参数响应图；PNG 和 PDF
```

服务器打包：

```bash
cd ~/QPAD_rebuttal
tar -czf rebuttal/outputs/regularization_v1.tar.gz \
  -C rebuttal/outputs regularization_v1
```

下载并解压到 WSL：

```text
/home/wyc/code/thu/research/Voiceprint/results/rebuttal/server/regularization_v1/
```

回传后，可在本地重新验证全部保存结果并生成图表：

```bash
python rebuttal/regularization.py summarize \
  --output results/rebuttal/server/regularization_v1
```

该汇总命令读取本次结果即可，不依赖服务器原路径或 GPU。
