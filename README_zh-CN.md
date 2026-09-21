# PEZ-CFC：基于码本候选过滤的自动硬提示词优化

[English README](README.md)

本项目是 [Hard Prompts Made Easy
(PEZ)](https://github.com/YuxinWenRick/hard-prompts-made-easy) 的小型研究扩展。
核心做法是在 PEZ 每一步最近邻投影前，根据 CLIP token embedding 各维度的标准差
筛选候选词。一次探索性实验包含 8 张图、3 个随机种子和 13 个阈值，共 312 次运行；
其中最佳阈值把平均最优 CLIP 余弦相似度从 **0.5095** 提升到 **0.5280**，
绝对提升 **0.0185**。

这里的筛选信号来自同一个 CLIP 模型的 token embedding，准确说是“模型内的码本
统计先验”。Token 分桶观察表明，embedding std 与 token 的语义可用性存在相关性：
低 std 区域集中了更多语义可用性较低的投影候选。因此，PEZ-CFC 利用这种相关性缩小
搜索空间，把有限的优化步骤集中到更有效的词表区域，同时不需要引入额外语言模型。

![阈值扫描结果](results/threshold_sweep.png)

## 研究问题与方法

PEZ 在连续 prompt embedding 上更新，然后投影到最近的离散 token。PEZ-CFC 将投影
候选限制为：

```text
candidate_ids = {i : std(W[i, :]) >= tau}
```

筛选后的局部索引会映射回原词表 ID，再进入原有 PEZ 前向和梯度更新。筛选结果会
缓存，不会每一步重复构建。

### Token 分桶观察

实验记录中有一次随机抽取 1,000 个码本 token、再按 embedding std 等量分成 10 组
的探测。两个端点组的代表样本展示了 std 与语义可用性的相关性：

| Std 分组 | 平均 std | 随机样本中的代表 token |
|---|---:|---|
| 最低组 `[0.0000, 0.0066]` | 0.0039 | `lmfa`、`galatasar`、`kaeper`、`kkkk`、`�▂�`、控制字符、重复 emoji、`growthhacking`、`sundaywith` |
| 最高组 `[0.0113, 0.0121]` | 0.0116 | `calendar`、`cleaning`、`dessert`、`turkey`、`school`、`chairs`、`water`、`horse`、`doctor`、`walking` |

低 std 样本中更集中地出现词片段、损坏文本和平台标签式拼接；高 std 样本中则有更多
可以直接使用的完整词。同一次探测中，两个端点组的平均总梯度分别为 `0.2203` 和
`0.0281`。这项分桶观察提供了 std 与语义可用性相关的定性证据；后续的阈值扫描则
直接度量 std 作为候选筛选信号的实用价值。

## 实验设置与核心结果

- 模型：OpenCLIP `ViT-H-14`，预训练标签 `laion2b_s32b_b79k`
- prompt 长度：16
- 优化步数：3,000
- 随机种子：0、1、2
- 图像：8 张作者收集的便利样本
- 阈值：0.000 到 0.012，间隔 0.001
- 总运行数：8 × 3 × 13 = 312

每个阈值下，先分别对每个 seed 的 8 张图求平均，再计算 3 个 seed 均值的平均值和
样本标准差。`results/threshold_sweep.csv` 和图中的误差条采用的都是这个定义，并非
24 个单次结果的标准差。

最佳测试阈值 `0.005` 的结果为 `0.527953 ± 0.001239`；未过滤基线为
`0.509497 ± 0.005223`，差值为 `+0.018456`。该阈值保留 46,164 / 49,408
个 token（93.4%）。当阈值升至 `0.012` 时只剩 191 个 token，结果下降到
`0.325137`，说明这里存在筛选质量与搜索空间覆盖范围之间的权衡。

## 安装与运行

原实验环境为 Python 3.8.10、PyTorch 1.13.0（CUDA 11.7 runtime）和 RTX 2080 Ti。
建议使用 Python 3.8–3.10：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行未过滤基线：

```bash
python run.py path/to/image.png --std-threshold 0 --seed 0
```

运行实验中最佳的筛选阈值：

```bash
python run.py path/to/image.png --std-threshold 0.005 --seed 0
```

安装后的快速冒烟测试可以追加 `--iterations 2`；它只验证执行路径，输出的 prompt
没有实验意义。

更换模型后不能直接假设 `0.005` 仍然最优，因为不同 embedding 码本的数值尺度可能
不同。

批量扫描与结果分析：

```bash
MAX_JOBS=1 bash scripts/run_threshold_sweep.sh image_a.png image_b.png
python scripts/extract_scores.py logs_sweep_YYYYMMDD_HHMMSS results/per_run_scores.csv
python -m pip install -r requirements-analysis.txt
python scripts/analyze_results.py
python scripts/plot_codebook_stats.py
python -m pytest -q
```

## 已验证内容

- 远端 312 个日志均能提取最终分数，并已生成匿名逐次结果。
- 分析脚本能从逐次结果重新得到基线、最佳阈值和 `+0.018456` 差值。
- 单元测试验证阈值为 0 时与完整码本投影一致。
- 单元测试验证筛选后的局部索引能正确映射回原词表 ID。
- 在原 RTX 2080 Ti 环境完成了 2 步 GPU 冒烟测试：模型加载、筛选投影、前向、反向和
  参数更新均正常；阈值 `0.005` 保留 46,164 个候选。

## 局限与尚未验证的推测

- 最佳阈值和汇报结果来自同一批图，没有独立验证集，可能高估收益。
- 8 张非标准数据集图片太少，不能支持普遍性结论；因隐私与再分发权不明确，原图和
  原 prompt 日志不公开，所以公开仓库只能精确复现结果分析，不能独立重跑原 312 次
  GPU 实验。
- 优化目标和评估指标都来自 CLIP。CLIP 分数提升不等于 prompt 更自然，也不等于
  文生图质量一定提升。
- Token 分桶已经观察到 std 与语义可用性的相关性；后续可以进一步量化这种相关性与
  token 频率、自然度及训练覆盖程度之间的关系。
- 尚未验证阈值跨 CLIP 模型、tokenizer、数据集或硬件的迁移性。
- 本实验没有证明优于使用 LLM 先验或其他可解释性约束的方法；它们引入了额外信号，
  并不属于完全相同的实验设置。

详细英文表格、仓库结构、归属说明与上游引用见 [README.md](README.md)、
[results/README.md](results/README.md) 和 [NOTICE.md](NOTICE.md)。
