# QPAD 返修实验代码

`rebuttal/` 用于维护返修新增或修改的代码、配置和服务器运行入口。问题拆解与实验决策集中在 [notes](../TTQM/review/notes.md)，逐条回复集中在 [draft](../TTQM/review/Response_to_Reviewers_draft.md)。

## 实验设计与实现状态

| 工作 | 目的与设计 | 当前状态 |
|---|---|---|
| 服务器预检 | 检查 Python/PyTorch、指定设备、输入/真值及代码路径；读取输入形状和哈希 | 已实现：`run.py check`；不执行拟合 |
| 原模型相位诊断 | 简单 `h2_s2` 与复杂 `cpl_h2_s2`，各用起点偏移 0/50 点，共 4 次；每次 2000 步 | 已实现：`run.py diagnose` |
| C：初始化与求解方式 | 同一修正模型上比较均匀/自动初始化 × 联合更新/分块求解；2 条信号 × 4 配置 × 5 初值条件，共 40 次试运行 | 方案已写入 notes，修正模型和运行入口待实现 |
| A：公平比较与评价 | 修复 QPGP、核对周期单位，保留连续分数与有效零分样本；统一阈值及评价协议，补代表方法 | 待实现 |
| B：失配、噪声与异常形态 | 代表性非对称/双峰背景、噪声强度、负向/双向异常及无异常对照；独立生成并保存真值 | 待实现 |
| 回传结果汇总 | 逐次指标、相位变化引起的目标/异常误差变化，以及 P/A 分量差异；保留失败与缺失 | 已实现：`summarize.py` |

### 当前四次诊断具体做什么

| 信号 | 起点偏移 | 改变内容 |
|---|---:|---|
| `h2_s2` | 0 点 | 原默认初始化 |
| `h2_s2` | +50 点 | 所有事件起点整体后移，其他初值和参数保持一致 |
| `cpl_h2_s2` | 0 点 | 原默认初始化 |
| `cpl_h2_s2` | +50 点 | 所有事件起点整体后移，其他初值和参数保持一致 |

信号长度 N=2000、K=10；默认事件间隔约 200 点，因此 +50 点约为四分之一周期。Adam 学习率为 0.01，原目标乘以 10 后反向传播，正则参数由 `configs/original_phase_probe.json` 固定。它检查原模型对初始相位的敏感性，并核对服务器运行情况。只有两个相位条件，不能估计一般随机初值的方差。

原模型优先从 `scripts/sml2_decompose.py` 加载；旧服务器目录也支持根目录的 `sml2_decompose.py`。只读取模型定义，不执行原脚本批处理入口。设备由新入口明确指定。

**局部支持、连续相似性、自动初始化和凸异常子问题尚待实现。** 当前诊断没有应用这些修改，也没有计算新的 F1/AUROC/AP。正式 C 应在同一修正模型上比较四种配置，再决定扩大到 A/B 的规模。

```text
rebuttal/
├── run.py                          # 服务器预检与原模型相位诊断
├── summarize.py                    # 回传后的指标与分量差异汇总
├── configs/
│   ├── original_phase_probe.json   # 共享诊断配置
│   └── server.local.json           # 可选本机路径配置，Git 忽略
├── outputs/                        # 每次运行独立目录，Git 忽略
├── .gitignore
└── README.md
```

## 路径、环境和设备

只依赖 NumPy 和 PyTorch 执行当前诊断。优先使用服务器已有实验环境；原 `environment.yml` 的名称为 `windturbine`，其中包含 PyTorch 2.5.1/CUDA 12.4。实际版本和设备由 `check` 确认。

默认配置自动查找以下路径，先使用整理后的目录，再使用旧目录：

| 内容 | 整理后 | 旧目录 |
|---|---|---|
| 信号 | `data/simulation/synthetic_signal/` | `synthetic_signal/` |
| 异常真值 | `data/simulation/synthetic_gtanomaly/` | `synthetic_gtanomaly/` |
| 原模型 | `scripts/sml2_decompose.py` | `sml2_decompose.py` |

需要 `h2_s2.npy`、`cpl_h2_s2.npy`，以及真值目录下对应的 `*_A.npy`，均为长度 2000 的一维数组。数据通过服务器本地路径读取。

`--device auto` 有可用 CUDA 时选 `cuda:0`，否则用 CPU；可以明确指定 `--device cuda:1` 或 `--device cpu`。指定不存在的 GPU 时预检会失败，避免误以为正在 GPU 上运行。

配置中的相对路径和路径类命令行参数均以仓库根目录为基准。可选覆盖项：

```text
--config CONFIG.json
--signal-dir PATH
--anomaly-dir PATH
--model-source PATH
--output-root PATH
--device cpu|cuda:N|auto
--epochs N
```

服务器路径可写进 `rebuttal/configs/server.local.json`（复制共享配置后修改）。该文件被 Git 忽略；共享配置保留可复现的实验参数。

## 第一步：WSL 上传到 GitHub

目前工作区还有之前目录整理产生的变更。以下步骤只提交这次准备的服务器运行目录；服务器已有的原模型文件可以直接使用。

在 WSL 执行（当前分支为 `review`）：

```bash
cd /home/wyc/code/thu/research/Voiceprint
git branch --show-current
git add rebuttal/
git diff --cached --stat
git commit -m "Add QPAD revision diagnostics and result summaries"
git push origin review
```

确认当前分支为 `review`，暂存区内容符合本次上传范围。稿件及双语回复可以另行选定文件提交；批量数据和输出保留在各自机器上。

## 第二步：服务器同步与运行

建议用独立工作目录运行返修，便于复用旧数据并保留服务器已有代码。第一次执行：

```bash
cd ~/Voiceprint
git fetch origin
git worktree add --detach ../Voiceprint-rebuttal origin/review
cd ../Voiceprint-rebuttal
conda activate windturbine
```

若服务器环境名称不同，激活已安装 NumPy/PyTorch 的实际环境。下面按服务器旧数据目录给出示例；如果数据已经整理到 `data/simulation/`，将两个路径替换为对应位置。

先预检：

```bash
python rebuttal/run.py check \
  --signal-dir ../Voiceprint/synthetic_signal \
  --anomaly-dir ../Voiceprint/synthetic_gtanomaly \
  --device cuda:0
```

预检成功后启动原模型诊断：

```bash
python rebuttal/run.py diagnose \
  --signal-dir ../Voiceprint/synthetic_signal \
  --anomaly-dir ../Voiceprint/synthetic_gtanomaly \
  --device cuda:0
```

当前诊断顺序运行四次拟合，无多 GPU 并行假设。远程长任务可在已有的 `tmux` 会话或服务器作业系统中启动同一命令。

后续 WSL 更新并 push 后，在服务器返修工作目录执行：

```bash
cd ~/Voiceprint-rebuttal
git status --short
git fetch origin
git switch --detach origin/review
```

该工作目录用于拉取已提交的返修代码；本机配置放 `*.local.json`。如果直接在服务器修改共享代码，应先保存并同步那些修改，再更新工作目录。

## 第三步：找到服务器输出

使用上面的独立工作目录时，每次运行会打印输出位置，并建立：

```text
~/Voiceprint-rebuttal/rebuttal/outputs/phase_<UTC时间>/
```

目录包含：

- `manifest.json`：代码提交、相关工作区状态、模型/运行脚本/配置及输入哈希、Python 和依赖版本、设备与计时范围。
- `config.json`：解析后的实际参数、输入路径和设备。
- `summary.json`：每次运行的完成/失败状态、最终目标、异常 RMSE、起点/时长移动量及耗时。
- `*_trace.csv`：每 50 步及最终目标值，step=0 为首次更新前，step=N 为完成 N 次更新后。
- `*.npz`：P/A 分量及初始/最终时序参数。

达到 2000 步只标记为完成固定迭代预算，不标记为已收敛。单次失败保留错误记录，并使总命令返回非零退出码。需要比较 CPU/GPU 时应分别运行并保留版本信息，不能将浮点实现差异直接解释成初始化影响。

正式 C/A/B 的配置与结果将继续沿用独立输出、版本记录和保留失败样本的规则。输出目录按需传回 WSL，用于更新 notes、正文和回复中的结果；Git 用于同步代码与共享配置。

## 第四步：从服务器回传到 WSL

回传整个输出目录，包含分解数组、目标曲线、配置和运行记录。运行结束后，在 **WSL** 执行：

```bash
cd /home/wyc/code/thu/research/Voiceprint
mkdir -p results/rebuttal/server
QPAD_SERVER='wangyichun@服务器地址'
rsync -av --progress "${QPAD_SERVER}:Voiceprint-rebuttal/rebuttal/outputs/" results/rebuttal/server/
```

将 `QPAD_SERVER` 改成平时使用的 SSH 登录名或 SSH 别名。远端路径相对于该账号的 home；如果实际工作目录不同，相应调整。两个目录末尾的 `/` 表示把各个 `phase_*` 目录直接放入本地 `results/rebuttal/server/`。再次运行会增量同步；完成运行后再回传，便于核对齐备性。

如果任一端没有 rsync，可以使用：

```bash
scp -r "${QPAD_SERVER}:Voiceprint-rebuttal/rebuttal/outputs/." results/rebuttal/server/
```

输出不通过 Git 传输。代码与配置的版本信息已经随运行写入 `manifest.json`。

## 第五步：在 WSL 汇总分析

选择回传后的实际 `phase_*` 目录，运行：

```bash
python3 rebuttal/summarize.py results/rebuttal/server/phase_实际时间
```

将 `phase_实际时间` 替换为返回的目录名。该步骤仅需 NumPy，不需要 PyTorch 或 GPU；也可以先在服务器对同一个目录执行。输出保存在该目录的 `analysis/`：

| 文件 | 内容 |
|---|---|
| `runs.csv` | 每次拟合的目标、异常真值 RMSE、时序移动、耗时和状态 |
| `phase_comparison.csv` | 相对 0 偏移的 Δ目标、Δ异常真值 RMSE、P/A 分量间 RMSE |
| `report.md` | 可读的汇总表、失败/缺失情况和解释范围 |

汇总会核对配置与运行记录是否一致，并检查预期运行和分解/曲线文件是否齐备。失败、未执行或回传不完整的条目保留在表中，不参与分量差异计算。齐备时退出码为 0，有失败/缺失时为 1，输入无法解析时为 2。

拿到结果后，分析顺序为：核对代码/配置/输入版本 → 检查完成与失败状态 → 比较目标及异常误差 → 对比 P/A 分量与目标曲线 → 判断原问题是否在服务器复现，并据此确定修改模型后的 C 试运行。P 分量间差异反映初值敏感性，不能替代对真实背景的恢复误差。先将判断写入 notes，正式证据齐备后更新正文与逐条回复。

## 本地功能校验（2026-09-30）

已在 WSL 的 PyTorch 2.12.0+cpu 环境完成：真实输入的只读预检；临时合成数组的一次更新与原优化器逐值一致；输出及运行记录检查；旧目录识别；缺失输入、不可用 GPU 和无效迭代数的报错检查。正式拟合实验留在服务器执行，CUDA 路径仍需在服务器验证。

回传汇总使用临时构造的结果验证了分量差异与指标差值，并检查失败、缺失运行、回传材料不全及配置不一致的处理。汇总过程不执行模型拟合。
