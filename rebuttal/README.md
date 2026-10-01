# QPAD 返修实验

入口：`experiments.py`。固定方案与参数见 [PROTOCOL.md](PROTOCOL.md) 和 [configs/necessary.json](configs/necessary.json)。正式结果由服务器运行产生。

## 正则参数敏感性检验

入口 `regularization.py`：分别将 λ 组、η 组、ψ 调为原值的 0.5 倍与 2 倍，连同默认值共 7 种配置。使用三个仿真种子及八份录音各自的首个片段，77 份结果中复用 11 份，新增 66 次拟合。服务器命令、来源核验和回传步骤见 [REGULARIZATION.md](REGULARIZATION.md)。结果写入 `rebuttal/outputs/regularization_v1/`。

## 周期先验与初始化对照

入口 `period_prior.py`：周期 160/200/240，每个周期种子 0/1/2，比较历史固定初始化与当前输入估计初始化，共 18 份 QPAD 结果。复用三份已有默认结果后新增 15 次拟合。预检、服务器运行及回传步骤见 [PERIOD_PRIOR.md](PERIOD_PRIOR.md)，配置见 [configs/period_prior.json](configs/period_prior.json)。结果写入 `rebuttal/outputs/period_prior_v1/`，自动生成配对统计及图片。

## 模型与理论对应的求解实验

新增入口 `theory_experiments.py`：在相同局部模板、L1/TV 正则和参数约束下，比较投影 AMSGrad 与近端分块求解（PALM）。默认 24 次小规模拟合，检查最优性残差、逐块下降、初值敏感性和检测效果。服务器命令、模型定义和结果判读见 [THEORY.md](THEORY.md)，配置见 [configs/theory_probe.json](configs/theory_probe.json)。结果写入 `rebuttal/outputs/theory_probe_v1/`。

## 实验范围

| 实验 | 数据与运行 | 回应意见 |
|---|---|---|
| A：统一评价和基线比较 | 现有 88 个真实片段；仿真部分共用 B。比较 QPAD、STL、VMD、QPGP、周期矩阵 RPCA，保存连续分数；统一报告预定阈值和零阈值 | R1.3、R1.5；耗时支持 R2.5 |
| B：适用范围 | 7 个场景 × 5 个种子 = 35 条新仿真：常规、非对称、双峰、高噪声、负向、双向、无异常；五种方法使用同一输入和标签 | R1.1、R2.3、R2.4 |
| C：初值敏感性 | 共用 B 的两条固定输入及默认 QPAD 结果，各增加 5 个预定随机初值，保存全部结果并报告方差 | R1.2、R2.5 |

默认总计 **625 次方法拟合**：123 条输入 × 5 种方法，加 C 的 10 次 QPAD 拟合。QPAD 共 133 次，其余四种方法各 123 次。阈值 2/3/4 倍比较、单向/双向评分和统计汇总均复用分解结果。

QPAD 使用 `models/` 中的归档定义和原参数：仿真 2000 步、真实 1500 步。基线和评分的具体实现、模型限制与参数选择方式均写在 PROTOCOL.md。完成运行后才能判断哪些有效性结论得到支持。

## 1. 服务器从 GitHub 同步

已建立的目录为 `~/QPAD_rebuttal`，原数据位于相邻的 `~/Voiceprint`。

```bash
cd ~/QPAD_rebuttal
git fetch origin
git switch review
git pull --ff-only origin review
git log -1 --oneline
conda activate windturbine
python -m pip install -r rebuttal/requirements.txt
```

服务器的 GitHub SSH 连接可检查：

```bash
ssh -T git@github.com
```

GitHub 显示成功认证但不提供 shell 是正常反馈。若当前远端为 HTTPS、希望使用已经配置的 SSH 密钥，可设置：

```bash
git remote set-url origin git@github.com:Evelyn-wyc/Voiceprint-Signal-Anomaly-Detection.git
```

`git pull --ff-only` 若报告分叉或本地代码冲突，应保留报错和本地修改，再处理同步；输出目录与本机配置已由 `.gitignore` 排除。

## 2. 预检

```bash
cd ~/QPAD_rebuttal
python rebuttal/experiments.py check \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0
```

应显示 `status: ready`、88 个真实输入、35 个仿真输入、625 次拟合。预检读取标签、检查文件长度和设备、检查所有依赖，不执行拟合。

自动识别数据目录：

| 内容 | 整理后的路径（相对 data-root） | 旧路径 |
|---|---|---|
| 真实特征 | `data/processed/241230_vector_npy/` | `241230_vector_npy/` |
| 矩形标注 | `data/annotations/241230_gt/` | `241230_gt/` |
| 已转换标签（备用） | `data/annotations/241230_gt_xwidth/` | `241230_gt_xwidth/` |

可用 `--real-signal-dir /实际目录 --real-label-dir /实际目录` 显式指定。CSV 支持逗号/制表符分隔的 X/Width 或 start/end；也支持 `*_xwidth.npy` 二值标签。35 条仿真由本次固定生成器创建，X/P/A/noise/事件标签成套保存。

## 3. 运行 A/B/C

建议在 `tmux` 会话内执行，便于断开 SSH 后继续运行：

```bash
tmux new -s qpad-review
```

在会话内：

```bash
cd ~/QPAD_rebuttal
conda activate windturbine
python rebuttal/experiments.py run \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0 \
  --output rebuttal/outputs/necessary_v1
```

按 `Ctrl+B`，松开后按 `D` 可离开会话；返回用 `tmux attach -t qpad-review`。程序逐次打印进度，完成后自动生成 `analysis/report.md`。QPAD 使用指定 GPU，其他方法在 CPU 上运行，默认各设 1 个计算线程。

### 中断后继续

```bash
python rebuttal/experiments.py run \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0 \
  --output rebuttal/outputs/necessary_v1 \
  --resume
```

续跑会核对配置、代码哈希、输入和依赖版本，验证已有分解文件后跳过完成项；失败项重新尝试，历史失败保留在 `events.jsonl`。损坏的完成文件会报错。改变配置或代码后使用新的输出目录。

可用 `--parts A` 或 `--parts B C` 分开运行，**每个范围使用不同输出目录**；默认一次运行 A/B/C 最便于汇总。`--parts C` 会自动计算所需两条输入的默认 QPAD 结果，再运行随机初值。

## 4. 看结果、回传给 WSL

先粘贴这份报告的完整内容：

```bash
cat ~/QPAD_rebuttal/rebuttal/outputs/necessary_v1/analysis/report.md
```

如中途报错，粘贴终端报错；完整运行中的单次失败还会写入 `analysis/failures.csv`。返回码 0 表示运行和材料完整，1 表示存在失败/缺失，2 表示配置、环境或输入错误。

**进一步分析需要回传整个 `necessary_v1/` 文件夹**，保留如下内容：

```text
necessary_v1/
├── config.json                 # 实际配置
├── manifest.json               # Git、源码、输入哈希、环境、计划任务
├── inputs/*.npz                # X、标签；仿真另含 P/A/noise 真值
├── fits/*.npz                  # P/A、连续分数；QPAD 初末参数和目标曲线
├── fits/*.json                 # 每次拟合参数、状态、计时和全部指标
├── events.jsonl                # 每次尝试及失败记录
├── progress.json
└── analysis/
    ├── report.md
    ├── summary.json
    ├── fits.csv
    ├── metrics.csv
    ├── metrics_by_group.csv
    ├── paired_comparisons.csv
    ├── initialization.csv
    └── failures.csv
```

在 **WSL** 执行，替换实际服务器 SSH 地址或别名：

```bash
cd /home/wyc/code/thu/research/Voiceprint
mkdir -p results/rebuttal/server
QPAD_SERVER='wangyichun@服务器地址'
rsync -av --progress "${QPAD_SERVER}:QPAD_rebuttal/rebuttal/outputs/necessary_v1" results/rebuttal/server/
```

如果使用手动复制，可在服务器先打包：

```bash
cd ~/QPAD_rebuttal
tar -czf rebuttal/outputs/necessary_v1.tar.gz -C rebuttal/outputs necessary_v1
```

下载后在 WSL 解压到 `results/rebuttal/server/`。确认出现 `results/rebuttal/server/necessary_v1/manifest.json`。这些输出通过文件传输回传；Git 用于代码与配置。

回传后可重新核验汇总（需要 NumPy/SciPy/scikit-learn，无需 GPU）：

```bash
python rebuttal/experiments.py summarize \
  --output results/rebuttal/server/necessary_v1
```

## 代码与验证

- `datasets.py`：成套仿真真值和真实标注读取。
- `models/`、`qpad_fit.py`：归档模型、固定参数、初值扰动与曲线记录。
- `baselines.py`：按采样点估计周期，STL/VMD/QPGP/RPCA 连续输出。
- `evaluation.py`、`report.py`：阈值、有效零指标、按录音/种子分组的配对比较和材料核查。
- `run.py`、`summarize.py`：此前四次相位诊断的复现入口。

功能测试使用小型合成数组：

```bash
python -m unittest discover -s rebuttal/tests -v
```

QPGP 协方差求解、对数行列式和条件均值与直接稠密矩阵计算对照；同时验证原 QPAD 一步更新、RPCA、标签边界、评价、完整输出、续跑和损坏文件识别。正式 CUDA 拟合由服务器运行验证。
