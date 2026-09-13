# FSEML on CORe50：现实连续视觉流实验设计

## 1. 实验目的

本实验用于回应“现有实验集中于受控图像分类基准，方法对更真实持续学习场景的泛化能力仍不清楚”的审稿意见。它不只是增加一个图像数据集，而是检验 FSEML 在以下条件下是否仍然有效：

- 样本来自连续拍摄的视频序列，具有明显的时间相关性；
- 同一类别会以新的物体实例、背景、光照、姿态和尺度反复出现；
- 数据流由许多小型、非 IID 增量批次组成；
- 新类别学习和旧类别再次出现同时发生；
- 测试时不给出 task ID，模型必须在所有已见类别中预测。

主结论应限定为：CORe50 为 FSEML 在现实连续物体识别流中的适用性提供证据，而不是证明其适用于所有真实世界或所有模态。

## 2. 与早期图像实验实现的关系

本设计复用现有 CIFAR100 实验的以下原则：

- 使用不可变的 JSON protocol manifest 固定数据流、类别映射、随机种子和划分；
- 采用单一全局分类头；
- 保存完整准确率矩阵 `R`，由同一函数计算 ACC、FM 和 LA；
- FSEML 使用 SRN/FSSAE + CPN，内循环只更新 CPN fast weights，外循环联合更新整个网络；
- FSEML-ER 继续采用容量受限的稀疏 latent buffer 和 decoder replay；
- 所有运行保存 checkpoint、完整参数、protocol 路径、逐批次指标和资源消耗。

但不能照搬当前 CIFAR100 主评估中的两个设置：

1. CORe50 主结果必须使用真正的 sequential 模式，不能使用每个阶段都从同一 checkpoint 重新开始的 `legacy-cumulative` 模式。
2. CORe50 主结果必须在所有已见类别中预测，即 `prediction_class_scope=seen`，不能使用 task-aware 的 `prediction_class_scope=task`。

task-aware 结果可以作为与旧实验对齐的诊断结果放在附录，但不能作为 CORe50 的主表结果。

## 3. 数据集与官方协议

### 3.1 数据来源

使用 CORe50 官方数据和官方 batch configuration，不自行随机重建 NIC 数据流。原始数据、协议文件及说明来自：

- CORe50 官方代码库：https://github.com/vlomonaco/core50
- CORe50 数据集论文：https://arxiv.org/abs/1705.03550
- fine-grained continual learning/NICv2 论文：https://arxiv.org/abs/1907.03799

在 protocol manifest 中记录数据版本、下载来源、文件校验值和官方 scenario/run 编号。

### 3.2 主协议：NICv2-79

论文 Table III 主结果采用 **NICv2-79** 流，并保留官方批次结构和时间顺序。50 个对象划分为互不重叠的 30 个 meta-training 类和 20 个 meta-test 类；20 个未见测试类组成四个连续的 5-way tasks。

必须遵循以下规则：

- 保留官方批次顺序和每个批次内部的原始顺序；
- 不跨批次 shuffle；
- 不将流重新整理成均衡的 class-incremental tasks；
- 一个流样本只用于一次在线更新，除非它由显式 replay 机制再次采样；
- 使用官方独立测试集，在规定 evaluation points 上测试所有已经出现的类别；
- 不向模型提供 batch/task identity；
- 全局类别标签固定为 0–49，不随批次重映射。

### 3.3 扩展协议：NICv2-391

NICv2-391 可用于更长时间尺度的扩展实验，但不属于 Table III 核心复现范围，也不替代论文采用的 NICv2-79 主协议。

### 3.4 运行与随机性

- 使用官方提供的多个 run/order 时，优先将官方 run 作为序列重复；
- 若官方配置只固定一个目标顺序，则使用 `seed = 9, 19, 29` 控制初始化、buffer admission、batch 内采样及优化器随机性；
- 不允许每种方法选择不同的顺序或数据增强随机数；
- protocol manifest 必须记录每个增量批次的样本索引、类别集合、首次出现类别和再次出现类别。

## 4. 训练与评估协议

### 4.1 两阶段结构

CORe50 实验采用“初始适配 + 在线持续流”结构，而不是把官方测试流随机拆成 60% meta-train 和 40% meta-test。

#### 阶段 A：初始批次训练

- 使用官方 batch 0 作为初始训练经验；
- 在 batch 0 内构造 support/query：support 为当前小批次样本，query 为同一初始经验中不重叠的样本；
- 只允许通过 validation split 选择超参数，不能查看官方 test 标签；
- 所有方法使用相同的 batch 0 样本、更新次数和增强方式。

#### 阶段 B：NIC 在线流

对官方后续增量批次依次处理：

1. 模型接收当前批次，不接收 task ID；
2. 在当前批次内部按时间顺序或固定的小窗口形成 support；
3. query 由当前批次中与 support 不重叠的样本构成；
4. FSEML-ER 可按预先固定的 replay 规则将历史 pseudo-samples 加入 query；
5. 每个当前样本只进行一个数据遍历；
6. 完成该批次更新后，在官方测试集的所有已见类别上评估；
7. 记录该 evaluation point 的逐类准确率、整体准确率和资源开销。

这里的“单遍”是现实性主协议。允许另外报告多 epoch 结果作为优化上界，但必须与单遍结果分开。

### 4.2 Support/query 形成

`BaseContinualMetaLearner.sample_training_data()` 当前假设 support 由 `update_step` 个单样本组成，query 是 loader 的第一个 batch。CORe50 适配时建议保留这一接口：

- `update_step = 10` 作为从 CIFAR100 继承的默认值；
- 如果增量批次不足以提供 10 个独立 support 样本，则使用 `min(10, batch_size - query_size)`，禁止复制当前样本来凑数；
- query batch size 默认 64；若当前增量批次更小，则使用剩余的全部样本；
- support 与 real query 索引严格不重叠；
- support 不做类别均衡重采样，以保留真实非 IID 性；
- FSSAE 的 FDA 项若当前 query 只有一个类别，允许退化为零 between-class 权重或普通重构项，不能从未来批次引入其他类别来人为满足 FDA。

最后一项需要单元测试，因为当前 `fda_feature_weights()` 在单类别 query 上可能得到全零 Fisher score，并自动回退到均匀权重；实验日志应记录这种回退发生的次数。

### 4.3 图像预处理

为复用现有 FSSAE，主方案使用统一的 RGB 输入：

- resize 到 `128 × 128`；
- train：随机 resized crop、轻量 horizontal flip；
- test：确定性 resize/center crop；
- 归一化采用 ImageNet mean/std，或 CORe50 train split 统计量；只能二选一并写入 manifest；
- decoder 输出范围与归一化方式必须匹配，不能继续依赖当前“所有非 Omniglot 数据均为 [-1, 1]”的隐含假设。

不建议将 CORe50 压缩到 28 × 28 作为主结果，因为这会丢失真实场景中的细粒度实例和背景变化。28 × 28 只可用于 smoke test。

## 5. 模型设计

### 5.1 CORe50 FSSAE

当前 `FSSAEConfig` 默认 28 × 28，且 encoder 最终固定池化到 `4 × 4`。CORe50 建议配置：

| 参数 | 主设置 |
|---|---:|
| input channels | 3 |
| image size | 128 |
| encoder channels | 32 |
| latent dim | 128 |
| adapter dim | 256 |
| output classes | 50 |
| sparsity target | 0.05 |
| sparsity weight | 5e-3 |
| reconstruction weight | 0.1 |
| Fisher auxiliary weight | 0.1 |
| FSSAE total weight (`fda_weight`) | 0.1 |

第一轮保持 latent/adapter 和损失权重与 CIFAR100 一致，只允许通过 NICv2-79 validation 在一个有限网格内调节学习率和 reconstruction weight。若 128 × 128 的 decoder 显存过高，先降低 `encoder_channels`，不要优先降低输入分辨率。

### 5.2 CPN 与预测范围

- 使用固定 50-way global head；
- 训练 loss 只在当前已见类别 logits 上计算；
- 测试从所有已见类别 logits 中选最大值；
- 未出现类别不能参加预测；
- 不重置已经出现类别的 classifier row；
- 当某个类别第一次出现时，只初始化该类别 row 一次。

这与现有 `reset_classifer()` 在每次采到任务时重置分类器的行为不同。CORe50 中类别会再次出现，重复重置会破坏持续学习，必须增加 `seen_class_ids` 状态并测试。

### 5.3 Replay

主表包含三个 FSEML 变体：

- `FSEML`：无 replay；
- `FSEML-ER-decoder`：存储 128-D latent，回放 decoder reconstruction；
- `FSEML-ER-raw`：相同 admission 和 sampling，但存储原图，作为 replay 上界。

主预算使用**等字节**比较。另提供“buffer 1000 items”结果与论文现有设置对齐。

等字节计算必须包含：

- latent/图像数值本身；
- 标签；
- Top-P/partition 元数据；
- dtype；
- 不计模型参数，因为所有 FSEML replay 变体共享模型。

默认 `replay_gap=960` 来自旧数据集，不能直接假定适合 CORe50。将其改为相对于已处理流样本数的比例：每处理 `G` 个真实样本触发一次，回放 `floor(rG)` 个样本。开发网格：

- `r ∈ {0.05, 0.10}`；
- `G ∈ {256, 512, 960}`；
- buffer capacity 以等字节主预算决定。

只在训练阶段完成配置选择，并在 NICv2-79 的最终四任务 meta-test 流到达前冻结所有超参数。

## 6. 对比方法与公平性

### 6.1 最低可接受主表

- Naive/Sequential fine-tuning；
- EWC；
- ER；
- DER++；
- OML；
- FSEML；
- FSEML-ER-decoder；
- FSEML-ER-raw。

如果实现成本受限，最低限度保留 Naive、ER、OML、FSEML 和 FSEML-ER-decoder。

### 6.2 公平约束

- 所有方法使用同一输入分辨率和相同 backbone 容量级别；
- 所有方法单遍处理相同的数据流；
- replay 方法报告等字节主结果；
- validation 搜索次数一致；
- 不允许为 FSEML 使用 test stream 选择超参数；
- 报告可训练参数量、峰值显存、总训练时间和每千样本更新时间；
- 如果某基线来自 Avalanche，必须核对其 task-free/online 设置，不能直接引用不同协议下的论文数字。

## 7. 指标

### 7.1 主指标

设 `R[i, j]` 为学习第 `i` 个增量批次后，对评价单元 `j` 的准确率。由于 NICv2 中类别会重复，除任务级 `R` 外还应保存逐类结果。

- **Stream Accuracy / Online Accuracy**：每个增量批次更新前或更新后的在线预测准确率，二者必须明确区分；主报告推荐 prequential（先测后学）。
- **Final ACC**：整个流结束后，在所有 50 类官方测试集上的准确率。
- **Average Incremental Accuracy (AIA)**：所有 evaluation points 的测试准确率均值。
- **Forgetting**：对每个类别使用历史最佳准确率减去最终准确率，再宏平均。
- **Learning Accuracy (LA)**：类别首次学习后的准确率宏平均。
- **Backward Transfer (BWT)**：最终性能相对类别首次学习后性能的变化。

### 7.2 现实性与资源指标

- 新类别与复现类别分开的准确率；
- 宏平均 per-class accuracy，避免频繁类别主导结果；
- buffer 的真实字节数；
- 参数量、FLOPs、峰值显存；
- 总更新时间和每批次平均/95th percentile 延迟；
- replay 次数和 replay/real sample 比例。

### 7.3 统计报告

- 至少 3 个 seeds/runs；
- 报告 mean ± std；
- 主方法与 OML、ER/DER++ 进行配对比较，因为它们共享同一数据流顺序；
- 如只有 3 次重复，不强调渐近显著性检验，报告逐 run 数值和配对差值更透明。

## 8. 消融与压力测试

主论文至少加入下列两个消融：

1. `FSEML w/o FSSAE`：验证现实流上的收益不是仅来自 MAML；
2. `FSEML-ER decoder vs raw`：验证压缩 replay 的收益和上限。

附录建议增加：

- 清晰 batch 边界 vs 固定窗口且不提供真实边界；
- 原始时间顺序 vs batch 内 shuffle；
- 128-D vs 64-D/256-D latent；
- 单类别 query 上 Fisher 回退次数与性能关系；
- task-aware 结果，仅用于和旧 CIFAR100 评价口径对照。

## 9. 实现映射

建议新增以下文件，而不是在 CIFAR100 文件中堆叠条件分支：

```text
repository/
├── datasets/
│   └── core50_continual.py
├── core50_protocols/
│   ├── core50_nicv2_79_<run>.json
│   └── core50_nicv2_391_<run>.json
├── prepare_core50_protocol.py
├── train_core50_fseml.py
├── evaluate_core50_continual.py
├── summarize_core50_results.py
└── core50_results/
```

需要修改：

- `model/fseml_model.py`：支持 128 × 128 和显式 normalization adapter；
- `model/meta_learner_fseml.py`：加入“类别只在首次出现时初始化”的逻辑；
- `model/replay_Fse.py`：记录严格字节预算，并支持在线 meta-test replay；
- `datasets/datasetfactory.py`：注册 `core50`；
- 训练入口：不能继续把 replay 限制为 meta-training-only。

注意：当前类注释明确写着 replay 只发生在 meta-training，evaluation checkpoint 不携带 replay state。CORe50 主实验是在线流，必须保存/恢复 replay state；否则测到的只是“预训练表示 + CPN 顺序微调”，不是现实流中的 FSEML-ER。

## 10. 运行矩阵与执行顺序

### 10.1 Phase 0：smoke test

- NICv2-79 前 3 个 batches；
- seed 9；
- Naive、FSEML、FSEML-ER；
- 检查无未来类别泄漏、类别不重复重置、support/query 不重叠、buffer 字节统计和结果文件。

### 10.2 Phase 1：协议验证

- 完整 NICv2-79 论文主实验；
- seed 9；
- 全部候选方法；
- 只在 validation 上选择学习率、`r/G` 和 reconstruction weight。

### 10.3 Phase 2：论文主结果

- NICv2-391 扩展实验；
- seeds/runs 9、19、29；
- 固定所有超参数；
- 生成主表、AIA 曲线、逐类 forgetting 图、内存–准确率图。

### 10.4 Phase 3：必要消融

- seed 9、19、29；
- FSEML、w/o FSSAE、decoder replay、raw replay；
- 其余设置与主结果完全相同。

## 11. 必须保存的产物

每个 run 至少输出：

```text
<run_name>.train_config.json
<run_name>.protocol.json
<run_name>.checkpoint.pt
<run_name>.replay_state.pt
<run_name>.events.jsonl
<run_name>.class_accuracy_matrix.csv
<run_name>.metrics.json
<run_name>.resource_metrics.json
```

`metrics.json` 必须包括数据版本、scenario/run、seed、预测范围、是否提供 task ID、单遍/多遍、buffer items、buffer bytes、最终 ACC、AIA、FM、LA、BWT 和在线准确率。

## 12. 成功判据与结果解释

建议预先规定以下判据，避免结果导向选择：

- FSEML 在无 replay 条件下，相对 OML 在 AIA 或 Final ACC 上有稳定正增益；
- FSEML-ER-decoder 在等字节预算下优于普通 ER，或达到 raw replay 大部分增益；
- 三个 runs 中至少两个方向一致，并报告全部 runs；
- 如果原始时间流下优势缩小，应如实解释为任务非平稳性和单类别小批次削弱 Fisher/meta-objective，而不是删除该结果。

无论最终是否达到最高准确率，这个实验都能回答方法在现实时间流中何时有效、何时退化。论文不应把“CORe50 上有效”扩大为“适用于所有真实世界序列”。
