# QPAD 模型与理论对应的求解实验

这轮实验检验三个问题：模型的局部事件定义是否能正常拟合；复合目标的最优性残差是否下降；不同初值下的分解与检测表现如何。入口是 `theory_experiments.py`，默认配置为 `configs/theory_probe.json`。

保留 QPAD 的背景加异常分解、平方重构误差及全部 L1/TV 正则。将局部包络、相邻片段对齐和参数可行域写成明确的函数，再在同一目标下比较两个求解器。服务器结果用于决定最终采用的求解方式和理论表述。

## 1. 本轮运行范围

| 输入与初值 | 设置数 | 两个求解器合计 |
|---|---:|---:|
| 常规、非对称、高噪声、无异常仿真，各 1 条，默认初值 | 4 | 8 |
| 常规仿真：局部扰动 3 次、可行域内随机初值 3 次 | 6 | 12 |
| 真实数据：首尾两组录音各取首个片段，默认初值 | 2 | 4 |
| 合计 | 12 | **24** |

真实录音与片段按文件名排序选取，选择过程不使用标签或检测表现。仿真沿用必要实验的数据生成器，种子为 0；初始化种子为 101、102、103。两个求解器共用相同输入、初值、目标、约束、迭代预算和评分方法。

- **投影 AMSGrad**：递减步长，显式投影到可行域，作为与稿件算法对应的数值比较。
- **PALM（近端交替线性化最小化）**：按 M、t、T、A 分块更新，近端步骤处理 L1/TV 和约束，回溯检查光滑项上界及充分下降。

默认仿真最多 1000 轮、真实最多 800 轮。PALM 一轮包含四个块更新，两个方法的每轮计算量不同，另存实际耗时。每 25 轮检查最优性残差。正式计算由服务器完成。

### 初值的具体含义

默认 M=1、A=0；仿真起点间隔 N/K、初始时长 100；真实起点从 250 开始均匀放置、初始时长 150。局部扰动给起点添加 ±0.1(N/K)，幅值与时长乘 0.9～1.1 的随机数。

`feasible_random` 在 M、t、T 的完整盒约束中均匀抽样，再将 t 排序，A 仍为零。它检查背景参数的较宽初值范围。K、可行域、正则权重和默认初始化的尺度知识在本轮保持固定；任意可行初值的普遍结论与未知周期/未知 K 的选参能力需要各自的证据。

## 2. 目标、变量与约束

设 N 为输入长度，K 为模板数，s=N/K，内部坐标为 τ=t/s、d=T/s。输出同时保存内部坐标和以采样点计的 t、T。

\[
P(x)=\sum_{k=1}^K M_k\phi_w\!\left(\frac{x/s-\tau_k}{d_k}\right),
\qquad X=P+A+\varepsilon.
\]

### 局部包络

取 w=0.05。`local_envelope` 在 [w,1−w] 上等于原半正弦 0.5 sin(πu)，在 [0,w] 和 [1−w,1] 上用三次多项式连接到零，区间外为零。令 a=(3v−wg)/w²、b=(wg−2v)/w³，其中 v=0.5 sin(πw)、g=0.5π cos(πw)：

\[
\phi_w(u)=
\begin{cases}
0,&u\notin[0,1],\\
au^2+bu^3,&0\le u<w,\\
\tfrac12\sin(\pi u),&w\le u\le1-w,\\
a(1-u)^2+b(1-u)^3,&1-w<u\le1.
\end{cases}
\]

函数及一阶导数在连接处连续，一阶导数 Lipschitz。配合严格正的时长下界，光滑重构项具有所需的局部梯度正则性。局部支持落实了稿件中“每个模板对应一次活动”的解释；边界连接需要在最终模型定义中写明。

### 连续相位对齐

取 J=64、u_j=(j+0.5)/J，定义总背景在第 k 个活动区间上的轮廓

\[
q_{kj}=P\bigl(s(\tau_k+d_ku_j)\bigr).
\]

相邻片段项为

\[
S=\psi\frac{s}{2}\sum_{k=1}^{K-1}\frac1J\sum_{j=0}^{J-1}(q_{kj}-q_{k+1,j})^2.
\]

这是固定采样网格上的连续对齐，梯度包括采样位置随 t、T 变化的链式求导。s/2 将均方差换算到默认初始时长对应的尺度：仿真 100 点、真实 150 点。该定义是对片段相似性项的明确化；最终稿应给出这一公式。

### 复合目标

\[
\begin{aligned}
f(M,\tau,d,A)&=\frac{\|X-P-A\|_2^2+S}{N},\\
h(M,\tau,d,A)&=\frac{\lambda_1\|A\|_1+\lambda_2\|DA\|_1
+\eta_1\|DM\|_1+\eta_2\|D^2M\|_1+\eta_3s\|Dd\|_1}{N}
+I_C(M,\tau,d,A).
\end{aligned}
\]

D 为一阶差分，D² 为二阶差分。所有 L1/TV 项按原形式求值。I_C 为闭凸可行集的指标函数。取 L=max(1,max|X|)，默认约束为：

- 0≤M_k≤4L；0≤A_i≤2L。
- −1.5≤τ₁≤…≤τ_K≤N/s。
- 0.15≤d_k≤1.5，即仿真时长 30～300 点、真实时长 45～450 点。

K 为仿真 10、真实 12。所有正则权重沿用必要实验中的 QPAD 参数，详见配置。边界由输入幅值与固定尺度定义，不使用异常标签。紧可行域显式保证迭代有界，d 的下界保证除法和梯度定义良好；h 保持凸且允许非光滑。

代码另支持配置 `anomaly_domain: signed`，对应 −2L≤A≤2L；默认运行采用稿件算法中的非负异常可行域。

## 3. 求解步骤与理论的对应关系

### PALM

在当前其余块固定时，对块 b 执行

\[
z_b=\operatorname{prox}_{\alpha_bh_b}
\left(x_b-\alpha_b\nabla_b f(x)\right).
\]

近端算子通过稀疏凸二次规划计算。以 δ=0.1 为默认值，回溯要求

\[
f(x_{-b},z_b)\le f(x)+\langle\nabla_bf(x),z_b-x_b\rangle
+\frac{1-\delta}{2\alpha_b}\|z_b-x_b\|^2,
\]

并检查

\[
F(x_{-b},z_b)\le F(x)-\frac{\delta}{2\alpha_b}\|z_b-x_b\|^2.
\]

程序保存每个已接受块的前后目标、步长、回溯次数、下降余量及近端残差。浮点下降容差为 10⁻¹⁰ max(1,|F|)。A 块使用不超过 0.45N 的步长；其平方损失梯度的 Lipschitz 常数为 2/N。其他块由回溯决定步长。

近端 QP 使用 OSQP 1.0.5，绝对/相对容差均为 10⁻⁸。每次求解必须返回 solved 状态，并通过另外计算的原始可行性和对偶残差检查，容许量为 10(10⁻⁸+10⁻⁸×数值尺度)。回溯或 QP 失败会记录为失败任务。

精确近端、精确下降条件下的理论推导可围绕：紧集上的光滑梯度、凸分块正则、充分下降、最优性残差和聚点驻性展开。整列收敛还需核对 KL 等条件。**有限精度程序记录数值容差下的证据；写渐近定理时仍需明确精确更新或满足相应条件的误差序列。** 参考 [PALM 原论文](https://bolte.perso.math.cnrs.fr/BST2013.pdf) 和 [OSQP 接口文档](https://osqp.org/docs/interfaces/python.html)。

### 投影 AMSGrad

使用 PyTorch Adam 的 `amsgrad=True`，步长

\[
\alpha_n=0.01\left(1+\frac{n-1}{100}\right)^{-0.6}.
\]

该标量序列满足不可求和、平方可求和条件。每步更新后投影到 C。起点顺序使用 AMSGrad 对角度量对应的加权保序投影；M、d、A 为盒裁剪。迭代在内部时间坐标中进行。

这个版本用于观察与稿件算法接近的更新在同一复合目标下的表现。**步长条件、约束和实测残差需要结合完整算法分析；原稿 AMSGrad 定理的推理仍需独立修订。** 如最终采用 PALM，正文算法与相应收敛命题需同步更新，模型的分解结构和正则项可以沿用。

### 最优性指标与停止原因

对同一当前点的每个块，以固定参考步长 ᾱ_b 计算

\[
G_b(x)=\frac{x_b-\operatorname{prox}_{\bar\alpha_bh_b}
(x_b-\bar\alpha_b\nabla_b f(x))}{\bar\alpha_b}.
\]

参考步长为 M:1、τ:0.05、d:0.05、A:0.45N。`mapping_inf`=max_b ||G_b||∞；精确近端时 G=0 等价于 0∈∇f(x)+∂h(x)。另外报告每个块近端位移相对 max(1,||x_b||) 的最大值，作为 `relative_fixed_point_residual`。

连续 3 次检查同时达到下列要求才标记 `stationarity_tolerance`：

- `mapping_inf`≤10⁻⁴。
- 相对固定点残差≤10⁻⁵。
- 约束违反量≤10⁻¹⁰。

预算用完标记 `iteration_budget`。`completed` 仅表示一次拟合及其材料已完整生成。最优性残差的数值依赖所声明的变量尺度和参考步长，跨方法采用相同设置。

## 4. 服务器同步与运行

在服务器执行：

```bash
cd ~/QPAD_rebuttal
git fetch origin
git switch review
git pull --ff-only origin review
git log -1 --oneline
conda activate windturbine
python -m pip install -r rebuttal/theory_requirements.txt
python rebuttal/theory_experiments.py check \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0
```

预检应输出 `status: ready`、`fits: 24`、两个求解器，以及 6 条输入。它读取完整真实数据目录后选择两条片段。真实文件支持旧目录和整理后的 `data/` 目录，也可通过 `--real-signal-dir`、`--real-label-dir` 指定路径。

程序自动寻找 `rebuttal/outputs/necessary_v1/` 或 `results/rebuttal/server/necessary_v1/` 中的原 QPAD 结果。输入和产物校验通过后，共匹配 6 份默认初值对照；已有结果放在其他目录时，可在 check 和 run 中添加同一个 `--reference-output /实际路径/necessary_v1`。没有归档结果也能运行，回传后进行对照即可。

进入已建立的 tmux：

```bash
tmux attach -t qpad-rebuttal
```

在会话内运行：

```bash
cd ~/QPAD_rebuttal
conda activate windturbine
python -u rebuttal/theory_experiments.py run \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0 \
  --output rebuttal/outputs/theory_probe_v1
```

按 Ctrl+B 再按 D 可离开会话。微分计算用指定 GPU，稀疏近端 QP 在 CPU 上执行；任务规模较小，A100 的利用率不能直接代表程序进度。程序逐项打印任务，拟合中每 100 轮打印目标与残差。

### 续跑

原命令末尾加 `--resume`：

```bash
python -u rebuttal/theory_experiments.py run \
  --data-root "$HOME/Voiceprint" \
  --device cuda:0 \
  --output rebuttal/outputs/theory_probe_v1 \
  --resume
```

已完成且校验通过的任务会复用，未完成/失败任务从该次拟合开头重跑。配置、代码、输入、依赖环境和归档对照必须一致；更改后使用新输出目录。

可选参数：`--sim-only` 为 20 次拟合；`--solver palm` 或 `--solver amsgrad` 为单求解器；`--max-iterations 2000` 同时改变仿真/真实预算。每种配置使用独立输出目录。首轮建议使用默认 24 次。

## 5. 输出、回传与判读

```text
theory_probe_v1/
├── config.json                 # 实际运行配置
├── manifest.json               # 代码/输入哈希、Git、环境、任务清单
├── inputs/*.npz                # 输入、标签；仿真另含背景/异常真值
├── fits/*.json                 # 指标、计时、停止原因、残差、参数范围
├── fits/*.npz                  # P/A、初末参数、完整目标曲线、块更新记录
├── live/*.json                 # 每次拟合最近一次诊断
├── reference/                  # 校验并复制的必要实验默认 QPAD 对照
├── progress.json
├── events.jsonl
└── analysis/
    ├── report.md               # 首先阅读这份报告
    ├── summary.json
    ├── fits.csv
    ├── reference.csv           # 有归档对照时生成
    ├── failures.csv
    └── figures/                # 目标/最优性曲线和背景/异常分解图
```

运行结束后自动汇总。退出码 0 表示材料完整，1 表示失败/缺失任务，2 表示配置/输入/环境错误。失败任务的 JSON 保存具体异常与 traceback。

先将报告内容粘贴回来：

```bash
cat ~/QPAD_rebuttal/rebuttal/outputs/theory_probe_v1/analysis/report.md
```

进一步分析需要整个输出目录。在 **WSL** 执行（替换实际 SSH 地址或别名）：

```bash
cd /home/wyc/code/thu/research/Voiceprint
mkdir -p results/rebuttal/server
QPAD_SERVER='wangyichun@服务器地址'
rsync -av --progress "${QPAD_SERVER}:QPAD_rebuttal/rebuttal/outputs/theory_probe_v1" results/rebuttal/server/
```

手动复制可在服务器打包：

```bash
cd ~/QPAD_rebuttal
tar -czf rebuttal/outputs/theory_probe_v1.tar.gz -C rebuttal/outputs theory_probe_v1
```

下载后解压到 WSL 的 `results/rebuttal/server/`，确认出现 `results/rebuttal/server/theory_probe_v1/manifest.json`。代码经 Git 传输，结果经文件复制回传。

回传后可在 CPU 上重新汇总：

```bash
python rebuttal/theory_experiments.py summarize \
  --output results/rebuttal/server/theory_probe_v1
```

### 结果支持哪些回复

1. **R1.4：理论条件与算法。**检查参数界、正时长、连续梯度、PALM 逐块下降、复合最优性残差，再按最终采用的更新步骤写证明。
2. **R1.2 / R2.5：初值与求解表现。**比较默认、局部扰动、可行域随机初值的残差、耗时、P/A 恢复和检测结果，保留全部运行。
3. **R1.1 / R2.4：模型形状与噪声。**查看非对称、高噪声和无异常输入的恢复、检测及误报表现。
4. **R2.2：稳定性含义。**分别报告驻性、固定背景下异常估计的性质和实测恢复结果；联合分解唯一性与全局恢复需要对应的数学条件。

主检测协议沿用必要实验：正向异常、稳健标准化、3 倍阈值，同时保存 2/4 倍阈值和双向指标。仿真额外报告 P/A RMSE；无异常场景重点看误报率。两求解器的目标值可直接比较；原归档实现和当前明确化的目标定义存在变化，对照使用检测与恢复指标。

首先看约束和下降检查，再看残差轨迹，最后看恢复和检测表现。有限次随机初值检验的是这些试验条件下的表现；本轮 K 与尺度约束固定。根据 24 次结果选择下一步，必要时只扩大选定求解器的验证范围。

## 6. 代码验证

```bash
python -m unittest discover -s rebuttal/tests -v
```

`test_theory.py` 使用小型数组检查包络连接处的函数/梯度连续性、近端闭式解与非扩张性、连续对齐梯度、下降条件、参数界、预算终止标签，以及完整运行、续跑和损坏产物识别。服务器正式拟合另行记录结果。
