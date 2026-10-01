# 周期先验与初始化对照

对应 R2.5：比较历史固定初始化与从当前输入估计周期后构造的初始化。入口为 `period_prior.py`，配置为 [configs/period_prior.json](configs/period_prior.json)。

## 1. 实验内容

- 常规正向异常仿真，长度 2000，周期 **160、200、240**。
- 每个周期使用种子 **0、1、2**，共 **9 条输入**。
- 每条输入比较 `fixed` 和 `input_estimated`，共 **18 份结果**。
- 复用 `necessary_v1` 中周期 200、种子 0–2 的三份默认 QPAD 结果后，**新增 15 次拟合**。

| 设置 | fixed | input_estimated |
|---|---|---|
| 周期信息 | 历史固定设置 | 从当前 X 的稳健自相关估计 p_hat |
| K | 10 | max(2, floor(2000/p_hat + 0.5)) |
| 初始时移 t_k | k·2000/10 | k·2000/K |
| 初始时间尺度 T_k | 100 | p_hat/2 |
| 幅值 M_k / 内部 A 参数 | 1 / 0 | 1 / 0 |
| 优化 | 归档 QPAD、Adam 0.01、2000 步 | 相同 |

两组都使用 λ=(1.5,1)、η=(0.15,0.15,0.05)、ψ=2，优化目标乘 10。模型和损失沿用 A/B/C 的归档定义。只增加可显式设置初值的接口，原默认更新过程保持一致。

周期估计复用 `baselines.estimate_period`：搜索范围为 [0.025N,0.25N]，固定截幅、平滑及峰值规则。生成周期与标签仅用于生成数据和评价，初始化函数只接收 X。零起始相位、半周期比例和周期搜索范围是公开的固定假设；该实验评价整套初始化规则随周期变化的表现。

主检测阈值 c=3，同时保留 c=2/4、连续 AP/AUROC、P/A RMSE、逐次耗时、初末参数和目标曲线。按同一输入配对，展示三个种子的全部结果及均值、样本标准差。K 随规则变化，最终目标值用于记录各次拟合，不用于挑选初始化方案。

## 2. 服务器同步与预检

在已有 `qpad-rebuttal` tmux 会话中执行：

```bash
cd ~/QPAD_rebuttal
git fetch origin
git switch review
git pull --ff-only origin review
conda activate windturbine
python rebuttal/period_prior.py check \
  --device cuda:0 \
  --reuse-from rebuttal/outputs/necessary_v1
```

预期显示 `inputs: 9`、`total_results: 18`、`reused_results: 3`、`new_fits: 15`。同时列出每条输入的 p_hat、K 和 T 初值。预检不拟合模型。

复用检查包括原模型/生成器/评分源码、参数、迭代预算、输入与拟合文件哈希、重新计算的指标，以及 NumPy/PyTorch、GPU 型号和 CUDA 构建版本。原输入在数值核对后直接复制，保证周期 200 两组使用完全相同的观测和真值。来源与耗时会写入本次输出。

- 若旧结果在其他目录，将 `--reuse-from` 改成包含 `manifest.json`、`inputs/`、`fits/` 的实际目录。
- 若旧结果不在服务器或运行环境不兼容，去掉 `--reuse-from`，程序将重新拟合全部 **18 次**。预检会明确显示次数。
- 如预检提示依赖缺失，可执行 `python -m pip install -r rebuttal/requirements.txt` 后重新预检。

本次输入由生成器创建，无需 `--data-root`。

## 3. 正式运行

```bash
python -u rebuttal/period_prior.py run \
  --device cuda:0 \
  --reuse-from rebuttal/outputs/necessary_v1 \
  --output rebuttal/outputs/period_prior_v1
```

程序逐项打印 `fit` 或 `reuse`，结束时自动核验并生成报告。返回码 0 表示 18 份结果完整；1 表示存在失败或缺失；2 表示预检、配置或环境错误。

中断后保留相同参数并加 `--resume`：

```bash
python -u rebuttal/period_prior.py run \
  --device cuda:0 \
  --reuse-from rebuttal/outputs/necessary_v1 \
  --output rebuttal/outputs/period_prior_v1 \
  --resume
```

续跑核对源码、配置、输入、复用来源和环境。已完成结果通过核验后跳过，失败项重新尝试。更改配置或代码后使用新的输出目录。

## 4. 结果与回传

先查看：

```bash
cat ~/QPAD_rebuttal/rebuttal/outputs/period_prior_v1/analysis/report.md
```

回传 **整个 `period_prior_v1/` 文件夹**：

```text
period_prior_v1/
├── config.json
├── manifest.json              # 源码/输入哈希、环境、任务及复用来源
├── inputs/*.npz               # X、P/A/noise 真值、标签
├── fits/*.npz                 # P/A、初末参数、目标曲线、连续分数
├── fits/*.json                # 每次状态、指标、耗时与复用信息
├── events.jsonl
├── progress.json
└── analysis/
    ├── report.md
    ├── summary.json
    ├── per_fit.csv            # 每个种子的实际结果与初始 p_hat/K/T
    ├── by_period.csv          # 分周期/规则的均值、标准差和范围
    ├── paired_per_seed.csv    # 输入估计初始化减去固定初始化
    ├── metrics.csv            # 全部阈值、单向及双向指标
    ├── failures.csv
    └── figures/               # 汇总、分解、目标曲线；PNG 和 PDF
```

服务器打包：

```bash
cd ~/QPAD_rebuttal
tar -czf rebuttal/outputs/period_prior_v1.tar.gz -C rebuttal/outputs period_prior_v1
```

下载后，在 WSL 将文件夹放到：

```text
/home/wyc/code/thu/research/Voiceprint/results/rebuttal/server/period_prior_v1/
```

回传后的 CPU 核验与重新汇总：

```bash
python rebuttal/period_prior.py summarize \
  --output results/rebuttal/server/period_prior_v1
```

本实验的范围是已列周期、种子和初始化规则。历史正则参数的选择依据仍需单独说明；原收敛定理与实际算法的对应由 R1.4 处理。
