# FSEML on CLINC150：跨模态表示层持续学习实验设计

## 1. 实验目的与结论边界

本实验不是端到端语言模型持续学习，而是检验 FSEML 的核心机制能否从图像迁移到冻结的文本语义表示：

```text
utterance → frozen sentence encoder → fixed embedding → vector FSSAE → CPN
```

它主要回答：

- FSEML 的 meta-optimized sparse representation 是否只对图像像素有效；
- Fisher 引导、稀疏约束和重构 replay 能否用于一般向量表示；
- 在未见文本领域和未见意图组成的连续流中，FSEML 是否优于顺序微调、OML 和普通 replay。

允许的论文结论是“FSEML 在冻结文本表示上的收益表明其表示层机制不固有依赖图像输入”。不能据此声称已经证明端到端文本、视觉语言或任意模态持续学习。

## 2. 重要实现判断

`model/fseml_model.py` 中现有 FSSAE 是图像专用实现：

- encoder 使用 `Conv2d`；
- decoder 使用 `ConvTranspose2d`；
- reconstruction loss 沿 `(channel, height, width)` 求均值；
- normalization 逻辑区分 Omniglot 和其他图像数据。

因此，CLINC150 **不能保持当前 encoder/decoder 结构原样不变**。应保持算法目标不变，但新增向量版本：

- Conv encoder → MLP encoder；
- ConvTranspose decoder → MLP decoder；
- 图像重构 → embedding 重构；
- Fisher feature weighting、KL sparsity、latent consistency 和 CPN 接口保持一致。

这应表述为同一 FSSAE 原理在 vector input 上的实例化，而不是“未经修改直接用于文本”。

## 3. 数据集

### 3.1 数据来源与统计

使用 CLINC150 官方 full split：

- 150 个 in-scope intents；
- 10 个领域，每个领域 15 个 intents；
- 每个 intent 有 100 train、20 validation、30 test utterances；
- 另有 out-of-scope（OOS）样本。

数据与论文来源：

- 官方数据仓库：https://github.com/clinc/oos-eval
- 原始论文：https://aclanthology.org/D19-1131/
- UCI 数据说明：https://archive.ics.uci.edu/dataset/570/clinc150

主实验先排除 OOS，避免将开放集检测和持续类别学习混为一谈。OOS 作为可选扩展单独报告。

### 3.2 官方 split 的用途

- `train`：meta-training 或 meta-test 流中的在线更新样本；
- `validation`：仅用于选择超参数、early stopping 和阈值；
- `test`：只用于最终及逐任务评价；
- 不把同一 utterance 或其 embedding 放入多个 split；
- 所有方法共享同一预计算 embedding 文件。

## 4. 任务协议

### 4.1 主协议：跨领域 meta-train/meta-test

论文协议将 10 个领域划分为互不重叠的 meta-training 和 meta-test 两部分：6 个完整领域用于 meta-training，4 个完整领域用于 meta-test。

- meta-train：6 domains，共 90 intents；
- meta-test：4 个互不重叠的 unseen domains，共60 intents；
- 每个 meta-test domain 形成三个连续的 5-way 测试任务，四个测试域共 12 个任务；
- 每个 phase 中 domain 的先后顺序和 domain 内 intent 顺序由 protocol manifest 固定；
- 主协议采用与本文其他 held-out adaptation 实验一致的 task-aware 5-way prediction；
- 额外保留所有已见意图候选的严格 seen-class 设置作为诊断，不用于主表结果。

训练域与测试域严格互斥；论文结果在 12 个连续 5-way 任务上报告，使遗忘能够随任务序列累积。

### 4.2 领域划分

主划分必须在看结果前固定：6 个领域属于 meta-training，另外 4
个领域属于 meta-test，二者完全隔离。领域名单、领域顺序和意图顺序
均写入 protocol manifest；不得根据测试结果更换划分。

如后续扩展稳健性实验，应预先注册额外的6/4 domain splits，而不能通过 test 结果挑选。

### 4.3 辅助协议：随机 intent split

为与现有 CIFAR100 的 60%/40% 类别划分对齐，可增加：

- 随机 90 intents meta-train；
- 随机 60 intents meta-test；
- meta-test 为 12 × 5-way class-incremental tasks；
- seeds 9、19、29 固定 class order。

该协议用于说明结果不是由某个领域 holdout 特例造成，但现实性和跨域说服力低于主协议，放在附录。

## 5. 文本表示

### 5.1 冻结 sentence encoder

所有 utterances 在实验前一次性编码，并保存为不可变特征：

- 推荐 encoder：`sentence-transformers/all-MiniLM-L6-v2`；
- embedding dimension：384；
- encoder 全程冻结；
- 采用模型推荐的 pooling，并对最终向量做 L2 normalization；
- tokenizer、模型 revision、库版本和权重哈希写入 embedding manifest；
- 所有方法读取同一 `.pt`/`.npz` 特征，禁止方法内部重新编码文本。

如果项目环境或离线部署无法获得该模型，可改用已缓存的 BERT-base pooled embedding，但必须对所有方法保持一致，并在论文中写明实际模型和 pooling。

冻结 encoder 的优点是把研究问题限定在 FSEML 的表示层持续学习，避免大语言模型更新成本主导实验；限制是无法检验语言 backbone 的 representation drift。

### 5.2 防止预处理泄漏

- 不使用全数据 PCA；
- 如需标准化，只用 meta-train train split 计算 mean/std；
- validation/test 不参与归一化统计；
- label names、domain names 和测试 utterances 不进入 encoder prompt；
- embedding 文件保存 source split 和 sample ID，加载时进行不重叠断言。

## 6. Vector FSSAE 设计

### 6.1 结构

新增 `VectorFSSAE`，建议初始结构：

```text
384-D input
  → Linear(384, 256) + LayerNorm + ReLU
  → Linear(256, 128)                         # sparse latent
  → Linear(128, 256) + LayerNorm + ReLU
  → Linear(256, 384)                         # reconstructed embedding
```

CPN 保持与图像版相同的职责，但缩小隐藏层以避免参数量远高于输入：

```text
128-D latent → Linear(128, 256) → ReLU
             → Linear(256, 512) → ReLU
             → Linear(512, 150)
```

不建议照搬当前 CPN 的 2304-D hidden，因为在 384-D 句向量上会产生不必要的参数开销，并削弱与轻量文本基线的公平性。

### 6.2 损失

保留现有目标形式：

```text
L = L_cls + λ_fssae (
      λ_rec L_embedding_rec
    + λ_sparse L_KL
    + λ_fisher L_fisher_aux
    + λ_wd L_weight_decay
)
```

向量重构建议同时考虑：

- 主项：embedding MSE；
- 诊断指标：real/reconstructed embedding cosine similarity；
- Fisher-weighted latent consistency 与当前实现保持同一数学定义。

首轮参数从 CIFAR100 继承：

| 参数 | 初始值 |
|---|---:|
| latent dim | 128 |
| adapter dim | 256 |
| sparsity target | 0.05 |
| sparsity weight | 5e-3 |
| reconstruction weight | 0.1 |
| Fisher auxiliary weight | 0.1 |
| FSSAE total weight | 0.1 |
| inner update steps | 10 |
| inner learning rate | 0.05 |

只允许使用 meta-training 六个领域的官方 validation split 进行有限网格选择：

- meta LR：`{1e-4, 5e-4, 1e-3}`；
- inner LR：`{0.01, 0.05, 0.1}`；
- reconstruction weight：`{0.01, 0.1, 1.0}`；
- latent dim：`{64, 128, 256}`，只在主 seed 上选择一次。

所有 FSEML 变体共享选定超参数；所有基线获得规模相当的验证预算。

### 6.3 单类别 support 与多类别 query

与现有实现对齐：

- inner support 使用当前任务样本，`update_step=10`；
- outer query 使用当前 meta-training task 的不重叠样本，batch size 64；
- 主设置采用 task-scoped query，防止 global query 让每次更新都看到所有 meta-train intents；
- 额外报告 global query 作为和 CIFAR100 现有默认配置对齐的诊断。

每个 5-way task 必须保证 query 中每个类别至少有 2 个样本，使 Fisher between/within-class 估计有意义。使用 balanced query sampler 是允许的，因为 CLINC150 主数据本身每类样本数相同；meta-test 在线流则保留 protocol 中固定的样本顺序。

## 7. Meta-training 与持续测试

### 7.1 Meta-training

- 仅使用 6 个 meta-train domains 的 train split；
- 每个 meta-training episode 为 5-way task；
- support 和 query utterances 不重叠；
- meta-training domains 的官方 validation 样本只用于选择 checkpoint 和 online LR，不引入独立的验证领域；
- 输出头为 150-way，但 loss 只在当前允许的 meta-train intent 集合上计算；
- 保存最佳 validation checkpoint 和最终 checkpoint，主结果使用预注册规则选定其中一个。

训练步数不直接假定为 CIFAR100 的 70,000。先用学习曲线确定预算，然后为所有 FSEML 消融固定同一 meta-update 数。建议开发起点为 20,000 updates，必要时扩展到 70,000。

### 7.2 Continual meta-test

从训练侧 validation 选定的固定 meta-trained checkpoint 开始，只运行一次完整12-task测试流：

1. 每个新任务到达时重置其5个新 intent 对应的 classifier rows；
2. 到达任务 `t` 后，仅使用该任务 train utterances 顺序更新；
3. 每个样本单遍使用；
4. 不重新加载 checkpoint；
5. 每学完一个任务，在所有已学习任务的官方 test split 上评价；
6. 主协议评价每个任务时只开放该任务的5个intent；seen-class prediction作为附加诊断；
7. 保存6 × 6 accuracy matrix、Final ACC、Learning ACC和Forgetting。

必须预先固定 test prediction scope，不允许查看 test 后切换协议。

### 7.3 测试阶段 replay 的两种口径

主实验采用与 Omniglot、CIFAR100 和 CORe50 一致的 **Transfer-only FSEML-ER**：decoder replay 只存在于 meta-training，meta-test 适应不使用 replay。若未来增加 Online FSEML-ER，必须单独命名，不能与主实验混报。

## 8. Replay 设计

文本场景中 decoder 重构 384-D embedding，而不是生成自然语言句子。论文中应称为 `pseudo-embedding replay`。

主比较：

- no replay；
- raw embedding replay：存储 384-D float embedding；
- FSEML decoder replay：存储 128-D latent，解码为 384-D pseudo-embedding；
- 可选 int8/raw-compressed embedding replay，作为严格存储基线。

主表采用等字节预算，附表采用 buffer 1000 items 与原论文一致。字节统计包括 vector、label 和 replay metadata。

初始 replay 设置：

- replay rate `r = 0.05`；
- replay gap `G = 256` 个真实 utterances；
- buffer capacity = 1000 items（兼容表）或等字节容量（主表）；
- partition 数量 10；
- Top-P 32。

`G=960` 对 CLINC150 太稀疏：每个 5-way task 只有 500 个 train utterances，因此它会使部分任务几乎没有 replay。应只把 960 放入敏感性分析，不作为默认值。

## 9. 对比方法和消融

### 9.1 最低可接受方法

- Frozen encoder + sequential MLP；
- ER with raw embeddings；
- OML；
- FSEML；
- FSEML-ER with pseudo-embeddings。

若资源允许，增加 EWC 和 DER++。所有方法只能访问相同的冻结 embeddings。

### 9.2 必要消融

- FSEML w/o FSSAE：MLP projection 替代 VectorFSSAE；
- FSEML w/o reconstruction；
- FSEML w/o KL sparsity；
- FSEML uniform reconstruction：移除 Fisher weighting；
- pseudo-embedding replay vs raw embedding replay。

主论文至少放 `w/o FSSAE` 和 replay 对照，其余放附录。

### 9.3 公平性

- 统一 sentence encoder 和 embedding 文件；
- 统一 150-way output head；
- 主表统一task-aware 5-way prediction；seen-class结果若报告则单独标注为严格诊断；
- 统一单遍 meta-test 流；
- replay 使用等字节主预算；
- 不允许某种方法访问 domain label；
- 报告 FSEML 新增 MLP encoder/decoder 的参数和 FLOPs；
- 不与端到端微调 BERT 的文献数字直接比较。

## 10. 指标

### 10.1 持续学习指标

- ACC：最终 12 个任务平均准确率；
- FM：历史最佳到最终的平均遗忘；
- LA：每个任务刚学完后的平均准确率；
- BWT；
- Average Incremental Accuracy；
- macro intent accuracy；
- prequential/online accuracy。

### 10.2 跨域指标

- 每个 held-out domain 的最终准确率；
- domain macro average；
- 相似领域（banking/credit cards）与差异领域的分组结果；
- 每个 domain 首次到达后的 adaptation curve。

### 10.3 Replay 质量与资源

- embedding MSE；
- real/pseudo embedding cosine similarity；
- 使用独立冻结 linear probe 对 real/pseudo embeddings 的 intent accuracy；
- replay effectiveness：相对 raw embedding replay 恢复的增益比例；
- buffer bytes、训练时间、峰值显存、每任务更新时间。

不能把高 cosine similarity 直接解释成语义或 intent 完全保留；必须同时报告 independent probe accuracy。

## 11. 可选 OOS 扩展

OOS 不进入主实验。若主实验完成且资源允许，可增加开放集诊断：

- OOS 样本只在 test stream 中出现；
- 使用 validation OOS 选择统一置信度阈值；
- 报告 OOS recall、in-scope accuracy、AUROC/FPR95；
- 所有方法使用同一阈值选择协议。

该实验可进一步增强现实性，但它检验的是 open-set recognition，不应与 catastrophic forgetting 主指标混合成一个分数。

## 12. 实现映射

建议新增：

```text
repository/
├── datasets/
│   └── clinc150_continual.py
├── clinc150_protocols/
│   ├── clinc150_domain_holdout_split0_seed9.json
│   ├── clinc150_domain_holdout_split1_seed9.json
│   └── clinc150_domain_holdout_split2_seed9.json
├── prepare_clinc150_embeddings.py
├── prepare_clinc150_protocol.py
├── train_clinc150_fseml.py
├── evaluate_clinc150_continual.py
├── summarize_clinc150_results.py
└── clinc150_results/
```

模型层建议：

- 在 `FSSAEConfig` 新增 `input_kind` 和 `input_dim`；
- 新增 `VectorFSSAE`，不要在现有 Conv FSSAE 内堆积大量 shape 条件；
- `SRN` 根据 `input_kind=image/vector` 构造对应实现；
- CPN hidden dim 配置化，去除硬编码 2304；
- replay fidelity 逻辑新增 vector metrics，禁止调用 SSIM 或图像归一化；
- `BaseContinualMetaLearner` 的 patch augmentation 对 vector 输入必须永久禁用。

## 13. 运行矩阵与执行顺序

### 13.1 Phase 0：数据与模型 smoke test

- 2 个 meta-train intents + 2 个 meta-test intents；
- 每类 5 个 utterances；
- seed 9；
- 检查 embedding 确定性、split 无泄漏、VectorFSSAE 前向/反向、decoder replay shape 和 classifier masking。

smoke test 只验证代码，不进入论文。

### 13.2 Phase 1：开发

- 固定主 domain split；
- seed 9；
- 完整 90-intent meta-training 和论文规定的 12 个连续 5-way meta-test tasks；
- 仅使用训练侧信息确定 meta LR、inner LR、latent dim 和 reconstruction weight；
- 检查是否出现 sentence encoder 导致的准确率饱和。

若所有方法接近满分，优先使用 `data_small.json` 或限制每个新 intent 的在线训练样本，而不是缩短到 3–5 个任务。

### 13.3 Phase 2：主结果

- 固定主 domain split；
- 主跨模态 sanity check 固定 seed 9；
- 至少报告 FSEML 和 FSEML-ER-decoder；
- 最终 test 只在 validation 完成模型选择后运行一次。

### 13.4 Phase 3：稳健性和消融

- 两个额外 domain splits，至少 seed 9；
- 主 split 上运行 w/o FSSAE、w/o reconstruction、raw embedding replay；
- 可选随机 intent split 和 OOS 诊断。

## 14. 必须保存的产物

```text
clinc150_embeddings_<encoder_revision>.pt
clinc150_embeddings_<encoder_revision>.manifest.json
<run_name>.protocol.json
<run_name>.train_config.json
<run_name>.checkpoint.pt
<run_name>.replay_state.pt
<run_name>.R_matrix.csv
<run_name>.intent_accuracy.csv
<run_name>.metrics.json
<run_name>.resource_metrics.json
```

manifest 必须包含原始 sample ID、split、intent、domain、embedding encoder/revision、pooling、normalization 和特征哈希。

## 15. 成功判据与论文表述

预先规定：

- FSEML 相对 OML 在 12-task Final ACC、AIA 或 FM 上取得稳定改善；
- w/o FSSAE 后性能或抗遗忘能力下降；
- FSEML-ER 在等字节预算下优于 raw embedding ER，或恢复其大部分增益；
- 至少报告所有 seeds 和 domain splits，不因某个 split 结果较弱而删除；
- 如果只在相似领域有效，应把它解释为任务共享结构是方法的适用条件。

论文建议表述：

> We evaluate FSEML on a continual intent-classification stream built from CLINC150. All methods operate on identical frozen sentence embeddings; therefore, this experiment isolates the portability of the proposed meta-optimized sparse representation mechanism at the feature level. It does not constitute end-to-end continual language learning.

该措辞既能打破“图像专属”的印象，也不会超过实验实际支持的结论范围。
