# 模型实验流程：定稿记录

**写于 2026 年 9 月 19 日。**

这份文档记录的是论文 Model 部分经过几轮讨论后**确定下来**的实验方案。它不是论文草稿，也不是代码说明，而是一份给未来的自己看的工作记录：几周之后回来看，你会记得做了什么决定，但很可能忘记**为什么**那么决定。所以下面每个决定后面都跟着一行理由。

读这份文档不需要事先了解 self-repair、SBN 或计算语义学，第一部分会把必要的背景补上。

---

## 目录

1. [背景：这个项目在做什么](#一背景这个项目在做什么)
2. [这个实验要回答的四个问题](#二这个实验要回答的四个问题)
3. [任务长什么样](#三任务长什么样)
4. [模型和实验条件](#四模型和实验条件)
5. [训练数据怎么配](#五训练数据怎么配)
6. [训练怎么设置](#六训练怎么设置)
7. [测试数据怎么组织](#七测试数据怎么组织)
8. [怎么评分](#八怎么评分)
9. [统计怎么做](#九统计怎么做)
10. [必须在论文里写明的限制](#十必须在论文里写明的限制)
11. [接下来要做的事](#十一接下来要做的事)
12. [关键数字速查](#十二关键数字速查)
13. [RP 修订期间的设计决定（2026-09-20）](#十三rp-修订期间的设计决定2026-09-20)

---

## 一、背景：这个项目在做什么

### 1.1 什么是 replacement self-repair

人说话的时候经常说到一半改口。比如：

> "How is the weather in Berlin, I mean, in Dublin?"

说话人先说了 Berlin，然后改口说 Dublin。语言学上把这三部分分别叫做：

- **reparandum**（被放弃的部分）：`in Berlin`
- **interregnum / editing term**（中间的插入语）：`I mean`
- **repair**（最终采纳的部分）：`in Dublin`

这种「用后面的说法替换前面的说法」的改口，叫 **replacement self-repair**，本文档里简称 **RSR**。

### 1.2 现有做法的问题

目前处理这类句子的标准做法叫 **disfluency detection**（不流利检测）：找出 reparandum 和 editing term，**把它们删掉**，然后把剩下的流利句子交给下游系统处理。

这个做法的问题是：被删掉的东西有时候后面还要用。比如接着上面那句问：

> "Show me the airports of these two cities."

"these two cities" 指的是 **Berlin 和 Dublin 两个**。如果 Berlin 在前一步被删掉了，这个指代就无法解析。**被放弃的材料虽然不被说话人所主张，但在语篇中仍然是可及的。**

### 1.3 这个项目的提案

本项目提出的方案是：不删除 reparandum，而是把它**保留在语义表示里，并明确标记为"已撤回"**。

用的语义表示格式叫 **SBN**（Simplified Box Notation，简化盒式记法）。它是 Parallel Meaning Bank（PMB）这个语料库使用的一种线性化的语义图表示。举例：

```
句子：  The movie starts at ten o'clock.
SBN：   movie.n.01 start.v.01 Theme -1 Time +1 Time +2 time.n.08 TSU now time.n.08 ClockTime 22:00
```

可以把 SBN 理解成「把一张语义图压扁成一行文字」。`movie.n.01` 这样的是概念（来自 WordNet 的词义编号），`Theme`、`Time` 这样的是语义角色，`-1`、`+2` 这样的数字是相对指针，表示"往前数第 1 个概念"、"往后数第 2 个概念"。

本项目对 SBN 的扩展是：用 `CORRECTION` 和 `CONJUNCTION` 这**一对**分隔符来表示 RSR。`CORRECTION` 开启一个装被撤回材料的语境，`CONJUNCTION` 开启一个装被采纳材料的语境，两者是兄弟关系。Berlin–Dublin 的例子写出来是：

```
entity.n.01 EQU ?                              % what
be.v.01 Theme -1 Time +1 Co-Theme +2           % 's ... like
time.n.08 EQU now
weather.n.01                                   % the weather
    CORRECTION <1
city.n.01 Name "Berlin" LocationOf -1          % [被撤回] in Berlin
    CONJUNCTION <2                             % I mean,
city.n.01 Name "Dublin" LocationOf -2          % [被采纳] in Dublin
```

### 1.4 关于 CORRECTION 的归属（重要，写论文时要用）

这件事查证过三次，结论是一致的：

| 查的地方 | CORRECTION | CONJUNCTION |
|---|---|---|
| `clf_signature.yaml`（DRS 标准签名文件，列出 20 个话语关系） | 没有 | 没有 |
| PMB 5.1.0 随附的 SBN 工具的分隔符清单 | 没有（2026 年 5 月由本项目加入） | 有 |
| PMB 5.1.0 的全部 300,133 条数据 | 出现 0 次 | 出现 84 次 |

**所以 RP 里现在那句 "without introducing a new separator"（没有引入新分隔符）是不成立的，必须删掉。**

但有一个更好的说法，而且是真的：**SBN 的分隔符清单本来就已经超出了 DRS 的标准签名**——`CONJUNCTION` 就在 SBN 工具里而不在 `clf_signature.yaml` 里。所以加入 `CORRECTION` 是在**沿用已有的先例**，不是另起炉灶。

建议的英文写法：

> Drawing on the SDRT Correction relation, the extension adds `CORRECTION` to the separator inventory supported by the PMB 5.1.0 SBN tooling and defines its paired use with `CONJUNCTION` for within-clause replacement self-repair. This follows existing practice: the SBN inventory already extends beyond the DRS clause signature, which lists neither `CORRECTION` nor `CONJUNCTION`. The extension otherwise retains the existing machinery of contexts, connectors, relative indices and semantic roles.

*（残留的小疑问：我查的是标准 DRS 评测仓库那份签名文件，理论上 PMB 内部可能另有变体。配合另外两条证据，这个可能性小到不必阻塞写作，但如果有机会问 Johan 一句可以彻底坐实。）*

### 1.5 pilot 已经发现了什么

在主实验之前做过一个小规模的先导实验（pilot）：拿 Gemma-2 9B，只用**流利的** PMB 数据做 LoRA 微调，然后丢给它 RSR 句子看会发生什么。结果：

| 输入类型 | 能解析成合法图的比例 | F1（失败记 0 分） |
|---|---:|---:|
| 流利句 | 97.5% | 0.950 |
| RSR 句 | 94.6% | 0.786 |
| RSR 句 + "I mean" | 76.5% | 0.609 |

**关键发现不是"分数掉了"，而是掉的方式。** 模型面对没见过的改口，**不会**丢掉 reparandum，而是把 reparandum 和 repair **粘成一个更大的图**。加上 "I mean" 之后更糟——它把 "I mean" 当成一个真实的言语行为，包住两边，图越撑越大，836 条输出里有 205–230 条出现了环，而SBN的本质是有向无环图（DAG），出现了环就意味着图已经不是良构的了。。

**一句话总结这个先验：模型"拿不准就多吐点结构"。** 后面很多设计决定都是在应对这一条。

另外还做了一个识别探针：让**没有经过任何微调的** Gemma-2 9B 用纯文本标出哪里是 reparandum、哪里是 editing term。有 "I mean" 时识别率 100%，没有时只有 44.8%（按位置细分是 30.7%–64.2%）。

**注意引用这个数字时的限定**：这是**未经训练的基座模型**做的**纯文本识别**，而且只用了 "I mean" 一个标记词。不能推广成"模型对所有 editing term 都这样"，也不能推广成"解析器会这样"。

---

## 二、这个实验要回答的四个问题

整个实验围绕四个问题组织。这个结构是讨论的最终产物，比"加了数据有没有用"那种问法清楚得多。

### 问题一：能学会吗（Learnability）

**模型能把这套 repair 表示学到什么程度？错在哪里？**

这不是一个明摆着的问题。`CORRECTION` 在 PMB 的全部 30 万条数据里出现 **0 次**，`CONJUNCTION` 只在 9,552 条 gold 训练数据里出现 **73 条**（0.76%）。更重要的是，这两者的**配对用法、兄弟约束、以及"句内替换改口"这个解读**，在训练前的任务监督数据里**完全不存在**。

所以问题是：一个通过 LoRA 微调、只看约几千条样本的模型，能不能学会一个这么罕见的构式，并且正确处理它的兄弟关系、索引约束和作用域范围？

**答案往哪个方向都有价值。** 学得好，是一个实打实的结果；学不好，说明这套表示法的学习成本高于预期，这对后续工作同样是有用的信息。

*（措辞提醒：只能说"在 PMB 的任务监督数据里罕见"，**不能**说"在预训练语料里罕见"——我们数的是 PMB，不是 Qwen 的预训练数据，无从推断后者。）*

### 问题二：保留 reparandum 换来了什么（Representation **benefit**）

**保留被撤回的材料，让模型能做到哪些"删掉就做不到"的事？**

这一问的证据是 **C2 与 C3 在 anaphora 挑战集上、指向被撤回先行词的命中率**。

**为什么是这个指标而不是别的**：C2 被训练成"看到 RSR 输入，输出不含 Berlin 的图"。**它从未被监督去保留被撤回的指称，也从未被监督去赋予它一个语篇可及的地位。** C3 则被监督去做这两件事。

英文写法：

> C2 receives no target supervision to preserve the withdrawn referent or to assign it a discourse-accessible status.

> ⚠️ **不要写成"C2 在结构上不可能产出指向 Berlin 的边"。** 这过强了——生成模型可以从**输入**里复制 Berlin。它没有被监督这么做，但这不是逻辑上的不可能。
> 而且反过来看：**如果 C2 偶尔真的生成了指回 Berlin 的指代边，那是一个值得单独分析的现象**，不该被"按定义不可能"的写法挡住。不要把预期结果写成定义保证。

> ⚠️ **这一问曾经填错过指标。** 早先版本把 Metric B 列为问题二的主要证据，但 Metric B 会**先把 C3 的改口结构剥掉**再和流利 gold 比，它测的是**代价**不是**收益**。用一个测代价的指标去支撑一个关于收益的主张，审阅人一定会问。已改正：收益归问题二，代价归问题三。

### 问题三：保留它花了什么代价（Representation **cost**）

这一问有两个方面，要分开报：

**（a）保留改口结构，有没有损害最终语义的恢复？**
证据：**C2 vs C3 的 Metric B**（把改口脚手架剥掉之后，和流利 gold 比）。

**（b）学了改口表示之后，普通的流利句解析会不会变差？会不会在没有改口的句子上乱吐改口结构？**
证据：流利 PMB 测试集上的**整图 F1** 和**误报率（false-alarm rate）**。

**(b) 是唯一一个新模型可能输的比较**，因此也是赢了才算数的比较。

### 问题四：泛化到哪里为止（Generalisation）

**模型学到的是真正的结构关系，还是某种表面模板？**

需要排除的模板有两层，现在两层都测：

| 要排除的捷径 | 用什么测 | 成本 |
|---|---|---|
| **标记词触发器**："看到 I mean 就输出 CORRECTION" | 有/无标记词的配对；一个训练时没见过的标记词（`sorry`） | 已在设计中 |
| **词对记忆**：记住了具体的 WordNet 词对 | **词对留出**（同一词对不跨划分） | 近乎免费 |
| **并置模板**："相邻两个竞争词 → 第一个 CORRECTION，第二个 CONJUNCTION" | **硬负例**：同位语、列举、无连词并列——表面像改口但不是 | 从 PMB 里挑，见 7.1 |
| 跨句与后续指代 | 挑战集 + 控制样本 | 部分手写 |

> **结论能主张到哪一步，取决于做了几层。**
> 只做第一层，只能写 *generalisation beyond seen editing markers and generated surface variants*。
> 加上词对留出和硬负例，才能写 *generalisation beyond seen editing markers, seen lexical pairs, and surface configurations that resemble repair without being repair*。
> **任何一版都不能笼统写成"证明了模型学的是结构而非模板"。**

---

## 三、任务长什么样

### 3.1 输入输出

这是一个标准的文本生成任务（causal language model 的监督微调）。**输入是纯文本，输出是一整行 SBN 字符串。**

训练数据的一条记录长这样：

```
user:      Please parse the following sentence to discourse representation structure:
           The movie starts at ten o'clock.
assistant: movie.n.01 start.v.01 Theme -1 Time +1 Time +2 time.n.08 TSU now time.n.08 ClockTime 22:00
```

**损失只算 assistant 那一段**，不算提示词部分。这是指令格式监督微调里最关键的一个正确性开关。

推理时用**贪心解码**（不做采样），因为输出是结构化的，采样只会给结构化输出增加随机性，没有好处。

### 3.2 提示词

提示词沿用 pilot 的那一句，**保持不变**：

```
Please parse the following sentence to discourse representation structure:
```

**为什么保持中性、不提改口**：

- 不能在提示词里说"请找出自我修正"，否则模型是在照着提示做，而不是从输入本身判断有没有改口。
- 流利句和 RSR 句必须用**同一个**提示词。如果给 RSR 句一个特殊提示，模型就知道"这句里一定有改口"，而我们恰恰想测它能不能自己发现。
- 所有尺寸、所有训练条件用同一个提示词。

**注意**：换到 Qwen3.5 之后，聊天模板（chat template）变了，"提示词与训练完全一致"这个约束要在**新的分词器上重新建立**，不能照搬 Gemma-2 那套手写模板。

### 3.3 一个完整的 RSR 例子

输入：

```
She prefers coffee, I mean, tea.
```

期望输出（简化示意）：

```
person.n.01 EQU speaker prefer.v.01 Agent -1 Theme +1
    CORRECTION <1
coffee.n.01
    CONJUNCTION <2
tea.n.01
```

模型要同时做三件事：识别出这里有改口、把 coffee 放进"已撤回"的语境、把 tea 放进"已采纳"的语境并接回句子的其余部分。

---

## 四、模型和实验条件

### 4.1 用哪个模型

**Qwen3.5 系列，取 2B / 9B / 27B 三个尺寸。**

**为什么是它**：同一个模型家族里有多个开放权重的稠密（dense）检查点，覆盖多个参数规模。这样比较尺寸时，架构和训练家族大体保持不变，是一个受控的比较。

Qwen3.5 的稠密尺寸实际有 0.8B / 2B / 4B / 9B / 27B（另有 35B-A3B 是稀疏专家模型，不用）。取 2B / 9B / 27B 作为小/中/大。

**关于"为什么不用更新的 Qwen3.8"**（2026 年 8 月发布）：

查证结果是——**Qwen3.8 只有 27B 稠密版**，另有一个稀疏专家的 Flash-Next。`Qwen3.8-2B`、`Qwen3.8-4B`、`Qwen3.8-9B` **都不存在**。

所以选 Qwen3.5 的理由不是"我们在某个日期冻结了选型"，而是**正面的**：它是最近一代里**唯一提供完整稠密尺寸梯子**的 Qwen 家族。论文里这么写：

> Qwen3.5 is selected because it is the most recent Qwen release that provides open-weight **dense** checkpoints across the full range of parameter scales required for a size comparison. The subsequent Qwen3.8 release provides a 27B dense model and a sparse-expert variant, but not the smaller dense sizes, and therefore does not support the same controlled comparison.

**不要**把 Qwen3.5 称作"最新"或"当前最强"——那是可以被证伪的，而且这个论证根本不需要它。

**论文里也不要逐一点名否定其他模型家族**（GLM、Kimi、DeepSeek、Gemma、Mistral 等）。理由：会把论证带进不必要的跑分争论；"明显更差"需要同一基准上的证据；模型排名很快过时。正面论证（同一家族多尺寸）已经足够。

**也不要说"三个尺寸在统计上有重要意义"。** 三个点能看出规模趋势，但统计可靠性来自别处（多次运行、足够的测试样本、配对比较）。

**两个 Qwen3.5 特有的坑**：

1. Qwen3.5 全系列是**多模态**架构。LoRA 要挂载的模块必须**显式指定为语言塔的投影层**，不能靠默认推断，否则三个尺寸挂载的模块集合可能不一致，那会直接污染尺寸比较。
2. pilot 出过一个已经确诊的 bug：Gemma-2 的聊天模板把句末终止符从损失里屏蔽掉了，导致模型**学不会停止生成**，adapter 作废重训。换家族意味着这个检查必须在新分词器上**重新做一遍**，不能假设它跟着代码一起修好了。

### 4.2 三个训练条件

这是整个设计的核心，要理解清楚三者的差别：

| 代号 | 训练输入 | 训练目标 | 说明 |
|---|---|---|---|
| **C1** | 流利句 | 流利 gold SBN | 只用原始 PMB 训练。参照条件。 |
| **C2** | 流利句 + **RSR 句** | 流利 gold SBN + **流利 gold SBN** | RSR 句配的是**不含 reparandum** 的原始 SBN。 |
| **C3** | 流利句 + RSR 句 | 流利 gold SBN + **repair-aware SBN** | RSR 句配的是**含 CORRECTION 结构**的目标。 |

**C2 是这轮讨论最重要的新增条件，名字叫 normalisation-target control（正常化目标对照）。**

**C2 为什么有用**：C2 和 C3 的**输入完全相同、样本数完全相同、优化步数完全相同**，唯一的差别是**目标的表示法**。所以 C2–C3 的对比隔离出来的就是"表示法本身"的作用，叫 **target-representation effect**。

**这个对照数据是现成的**：数据生成器在每一行里同时存了 `nl_clean`（原始流利句）、`sbn_clean`（原始流利 gold）、`nl_repair`（RSR 句）、`sbn_repair`（repair-aware gold）。C2 只是把同一批输入配上另一列目标，不需要重新生成任何数据。

**C2 还顺带把 pilot 的设计升级了**：pilot 做的正是"改句子、保留流利目标"，但它是零样本的。C2 把这个条件变成**受过训练的**条件——于是"没有 repair-aware 目标会怎样"这个问题，是在一个**专门训练过 RSR 输入**的模型上回答的，而不是在一个从没见过改口的模型上。

**措辞注意**：写 *C2 receives no target supervision to preserve or structurally distinguish the withdrawn material*，**不要**写"即使训练过也做不到"。后者把预期结果写成了定义保证。

### 4.3 一个必须理解的陷阱

**C1 和 C2 在 Metric A（改口结构分）上预期得分很低，而这是设计的结果，不是发现。**

- C1 从没见过 `CORRECTION` 这个记号（PMB 里 0 次）。
- C2 的训练目标里没有改口结构，它**从未被监督去保留或在结构上区分被撤回的材料**。

所以"C3 的 Metric A 比 C1/C2 高"**不能作为核心证据**——那等于说"训练过 X 的模型比没训练过 X 的更会做 X"。

英文写法：

> C1 and C2 are expected to obtain little repair-structure credit. Their scores function as controls, not as evidence for learnability.

**注意措辞的分寸**：写"预期得分很低"和"作为对照条件"，**不要**写"必然接近 0"或"按定义为零"。这两个条件的实际得分是要**测出来报告**的；若它们意外地拿到了分（例如从输入里复制出被撤回的词），那本身是值得分析的现象。

**真正承重的数字是**：

1. C3 的 **Metric A 绝对值**有多高、错在哪里（对应问题一）
2. C2 vs C3 在 anaphora 挑战集上**指向被撤回先行词的命中率**（对应问题二，收益）
3. C2 vs C3 的 **Metric B** + 流利 PMB 上的**误报率和整图 F1**（对应问题三，代价）
4. 标记词配对、held-out 标记词、**词对留出**、**硬负例**、挑战集（对应问题四）

**注意第 2 条和第 3 条不能互换。** 指代边命中率测的是"保留它换来了什么"，Metric B 测的是"保留它花了什么"。用后者去论证前者是早先版本的错误。

### 4.4 实际要跑的运行清单

**13 次微调**（这是在算力确认之前的计划方案；smoke run 之后可能上调到 17 次）：

| 尺寸 | C1 | C2 | C3 | 每条件 seed 数 | 小计 |
|---|:---:|:---:|:---:|:---:|---:|
| 2B | ✓ | — | ✓ | 1 | 2 |
| 9B | ✓ | ✓ | ✓ | 3 | 9 |
| 27B | ✓ | — | ✓ | 1 | 2 |
| | | | | **合计** | **13** |

**为什么这么分**：

- **六个主条件（三尺寸 × C1/C3）必须齐全**。Michael 明确要求跑满组合，而且如果某个尺寸只跑一个条件，那个尺寸上就无法把"尺寸效应"和"训练条件效应"分开。
- **C2 只在 9B 上跑**。它是核心对比所在，而且三个尺寸都跑 C2 成本翻倍、收益有限。
- **9B 跑 3 个 seed**，因为核心结论在这里，必须有训练随机性的估计。
- **2B 和 27B 单 seed**，明确标注为**描述性比较**，不做运行层面的统计推断。

**如果算力允许**，优先把 2B 加到 3 个 seed（+4 次，共 17 次）。**不要**先加 27B 的 seed——27B 单项成本最高而尺寸效应是四个问题里最弱的一个。

---

## 五、训练数据怎么配

### 5.1 基础语料

PMB 5.1.0 的 gold 部分：

| 划分 | 文档数 |
|---|---:|
| train_gold | 9,552 |
| dev_standard | 1,194 |
| test_standard | 1,194 |

（另有 silver 14.7 万条、bronze 14.1 万条，**不使用**——质量不够，而且里面 `CONJUNCTION` 出现 0 次。）

### 5.2 RSR 数据是怎么造出来的

从一个流利的 PMB 句子出发，配合它的 gold SBN 和词—概念对齐信息：

1. 在一个已经对齐的概念位置上，从 WordNet 取一个**竞争词**；
2. 把这个词**插在原词前面**，作为 reparandum；
3. 同时对 gold SBN 做规则化转换，加上 `CORRECTION` / `CONJUNCTION` 配对结构。

例子：`She prefers tea.` → `She prefers coffee, tea.`

**为什么用"生成"而不是"从真实语料标注"**：真实的口语改口语料里，没有配套的 gold 语义图。从流利的 PMB 出发，我们同时掌握原句、原图和改口位置，**每一步转换都可以对着原图检查**。

**注意这不是一个可以反过来用的解析器**：语料构造时"改口发生在哪里"是已知的；推理时它是未知的。这个区别在论文里要讲清楚。

### 5.3 语料规模（重新划定后）

原计划的配额网格里，有一半的单元依赖**尚未实现**的转换算子。经过重新评估，改成三层：

**第一层：核心（算子已实现，可以立刻生成）**

| 单元 | train | dev | test |
|---|---:|---:|---:|
| 概念替换 × 丢弃角色 | 500 | 60 | 100 |
| 概念替换 × 反转角色 | 500 | 60 | 100 |
| 概念替换 × 锚点占位 | 500 | 60 | 100 |
| 概念替换 × 移位 | 300 | 35 | 45 |
| 概念替换 × 无需装置 | 400 | 50 | 80 |
| **小计** | **2,200** | **265** | **425** |

**第二层：优先扩展（时间允许就做）**

| 单元 | train | dev | test |
|---|---:|---:|---:|
| 常量 × 人名 | 500 | 60 | 100 |
| 常量 × 数字 | 400 | 50 | 80 |
| 常量 × 时间 | 220 | 30 | 50 |
| 边标签（装置 ③） | 500 | 60 | 100 |
| 转发（机器草稿 + 人工复核） | 200 | 30 | 60 |
| **小计** | **1,820** | **230** | **390** |

**第三层：移出主实验**

| 单元 | 原配额 | 处置 |
|---|---:|---|
| 时体（tense/aspect） | 500 / 60 / 100 | **移出**，列为后续工作 |

**为什么移出时体**：设计文档自己的勘误表把它定性为"计划中最大的一块新机器"——它需要**多概念 span 支持**（记法文档的标注在每个盒子里放两个概念），最初被误判为廉价扩展。现在是 9 月 19 日，它在关键路径上，而砍掉它不影响任何一个核心比较。

论文里的写法：

> The main experiment focuses on replacement types supported by the completed and validated operators; tense and aspect replacements require multi-concept transformations and are left for subsequent extension.

**训练集规模区间**：核心 2,200 行（对 **9,447** 条流利文档约 1:4.3）；全做完 4,020 行（约 1:2.4）。

*（流利侧是 9,447 而不是 9,552，因为硬负例在训练前按文档留出了 105 篇，见 7.1。留出只占 1.1%，两个比例四舍五入后不变。）*

**重要原则：配额是上限，不是必须凑满的硬指标。**

如果某一格的合格候选不够，就让它少，**不要**为了凑数放松质量过滤或塞进近重复样本。重新分配时按其余单元的原配额比例走，并继续遵守"每个 PMB 源文档最多取 2 条训练样本"的限制（防近重复）。

### 5.4 改口数据和流利数据的比例

**目标比例约为 改口 : 流利 = 1 : 2 到 1 : 4。**

**为什么改口数据不能占多数**——这是整个数据设计里最重要的一条理由：

pilot 已经证明模型**已经在过度生成结构**。要扭转的先验是"拿不准就多吐点结构"。如果训练集以改口数据为主，模型不会纠正这个先验，而是会**把它反转**——学到 `CORRECTION` 是常态，然后在流利句上也开始吐 `CORRECTION`。

**必须保留流利数据作为负例。** 也就是目标里**完全不含** `CORRECTION` 的样本。没有它们，就没有"什么时候**不该**产出这个构式"的信号。

**关于两组训练数据量不等的问题**（C1 是 9,447 篇，C3 是 9,447 + N 条）：

这个不等是**故意保留**的。强行配平只有两种做法，都更糟：砍掉等量的 PMB（破坏上面的负例论证），或者复制 PMB 凑数（引入重复曝光这个新混淆）。

**混淆靠 C2 消除，不靠配平。** C2 与 C3 样本数相同、输入相同、步数相同，只有目标不同——这才是干净的隔离。C1 与 C3 的对比因此只能称作 "the effect of adding repair-aware training data"，**不能**称作被严格隔离的 supervision effect。

### 5.5 Editing term（改口标记词）怎么配

**候选集合是五个标记词**，刻意选小、刻意避开有歧义的：

```
I mean · no · no wait · sorry · actually
```

**被排除的**（记下理由，论文要用）：

| 排除的词 | 理由 |
|---|---|
| `that is` / `or rather` / `in fact` | 这些是**重新表述**标记，不是**改正**标记。它们的意思是"换个说法讲"，两种说法都成立、后者是对前者的注解——和 `CORRECTION` 主张的正好相反。 |
| `well` | 主要是犹豫或缓冲。`X, well, Y` 可能是改正，也可能只是停顿。歧义太大。 |
| `I mean to say` | 不歧义，只是不自然——口语里没人这么说。 |

`actually` 是保留下来的里面最靠近边界的（它在别处可以是普通强调词），但在 `X, actually, Y` 这个位置上读作改正是可靠的。**明确决定保留。**

**训练用四个，留一个做 held-out**：

- 训练集用：`I mean`、`no`、`no wait`、`actually`
- **`sorry` 完全不进训练集**，只出现在测试集

**为什么留一个**：pilot 已经测出有标记时识别率 100%、无标记时 44.8%。这个 55 个百分点的落差**已经证实**了词汇捷径的风险，不是假想。留一个没见过的标记词，才能区分"学会了结构"和"记住了词表"。

**训练集不做配对，只做少数层**：

- 训练集里约 **30%** 的改口样本带标记词，且这些样本**来自不同的站点**（不是同一个句子的有标记/无标记两版）。
- **为什么不在训练集里配对**：(1) 设计文档明确论证"有标记的是**容易**的那一类"，不应占多数，全配对会让它占到 50%；(2) 配对产生的两行除了一个插入词之外完全相同，这正是"每源文档最多 2 条"规则要防的近重复。

**配对放在测试集**（详见第七节）。配对是**评测**的需求，不是学习的需求。

### 5.6 NLI 过滤：降级为启发式

原计划用双向自然语言推理（NLI）给替换词打分，过滤掉两类坏样本：意思太接近的（`couch` / `sofa`，语义对比不足）和位置对但语境不合理的（`cake` / `carburettor`）。

**现状**：150 条人工标注给出 **149 否 / 1 是**。这不是"阈值需要调"，而是**标注样本里根本没有决策边界**。

**决定：取消阈值校准，NLI 降级为保守的启发式过滤。** 重新构造一个围绕边界抽样的标注集需要时间，而这个组件目前几乎不筛除任何东西，却卡在关键路径上。

论文里的写法（注意不能写"calibrated"，那和结论自相矛盾）：

> A 150-item manual assessment did not provide enough positive cases to calibrate a decision threshold. NLI is therefore retained only as a conservative heuristic, while final corpus quality is estimated on a separate manually assessed sample.

---

## 六、训练怎么设置

### 6.1 统一适配协议

所有尺寸用**同一套适配协议**（standardised adaptation protocol）：

| 参数 | 值 | 是否固定 |
|---|---|---|
| LoRA rank `r` | 32 | **固定** |
| `alpha` | 64（= 2r 惯例） | **固定** |
| dropout | 0.05 | **固定** |
| 学习率 | 2e-4 | **固定** |
| epochs | 3.0 | **固定** |
| 最大序列长度 | 1024 | **固定** |
| scheduler / warmup | cosine / 0.03 | **固定** |
| weight decay | 0.0 | **固定** |
| **有效批大小** | **16** | **固定** |
| per-device 批大小 × 梯度累积 | 4 × 4（可变） | **允许按显存调整，乘积必须是 16** |
| 被适配的模块范围 | 语言塔投影层 | **固定**（三个尺寸必须一致） |

**为什么固定的是"有效批大小"而不是"单卡批大小"**：训练数据量相同 + 有效批大小相同 ⟹ optimizer 步数自动相同。27B 很可能要退到 `1 × 16`，2B 可以 `8 × 2`，但两者走过的优化步数完全一致。

**措辞注意**：不能说这样就得到了"干净的尺寸效应"。即使固定 `r = 32`，三个模型的层数、隐藏维度、被适配模块数量、LoRA 可训练参数总量都不同，同一个学习率对不同尺寸的适合程度也不同。

正确的说法是：

> The comparison is between model sizes **under one pre-specified adaptation protocol**, not between their best attainable performance, and not with adaptation capacity held constant — trainable parameter count necessarily varies with depth and hidden size.

**建议在附录报告每个尺寸的 LoRA 可训练参数量和被适配模块列表**，让读者自己判断适配容量差了多少。这个数在 smoke run 里顺手就能拿到。

### 6.2 正式训练之前必须做的 smoke run

**这是目前唯一的瓶颈。在拿到这些数之前，"13 次运行可行"只是一个信念，不是判断。**

三个 smoke run：

- **2B**：固定数据跑约 100 步
- **9B**：固定数据跑约 100 步
- **27B**：至少完成模型加载、LoRA 挂载、跑 20–50 步

每个都要记录：

| 记录项 | 用途 |
|---|---|
| GPU 型号与数量 | 可复现性 |
| 显存峰值 | 决定 per-device 批大小怎么拆 |
| 每步秒数 | 推算完整训练时长 |
| 每 epoch 的 optimizer 步数 | 核对三个尺寸是否一致 |
| checkpoint 大小 | 存储规划 |
| 是否 OOM | 显然 |
| **LoRA 可训练参数量** | 附录要报 |
| **终止符行为** | 见下 |
| 截断情况 | 确认 1024 够用 |

**终止符检查要单独说**：生成 20 条输出，看模型会不会**自己停下来**。这是 pilot 那个已确诊 bug 的直接回归测试——当时聊天模板把句末终止符从损失里屏蔽掉，模型学不会停止生成，adapter 作废重训。放在 smoke run 里成本为零，放到正式训练之后才发现则是一周。

### 6.3 检查点选择与生成设置

**这些必须在看到测试结果之前就定死**，否则等于看完结果再选规则。

| 项 | 决定 | 理由 |
|---|---|---|
| Base 还是 instruct | **instruct**（`Qwen/Qwen3.5-*`，**不是** `-Base`） | pilot 用的是 `google/gemma-2-9b-it`，保持一致；而且提示词本身是一条指令 |
| 精确版本 | **在附录钉死 revision / commit hash** | 模型仓库会更新，不钉死则不可复现 |
| 用最后一轮还是最佳检查点 | **最佳检查点** | 3 个 epoch 有过拟合空间 |
| 选择标准 | **失败惩罚后的整图 dev F1**（失败记 0） | 见下面的重要说明 |
| Metric A / Metric B 的角色 | **只用于诊断，不参与检查点选择** | 防止选择标准偏向某个条件 |
| 最大生成长度 | **512 new tokens** | gold SBN 最长约 66 个概念，512 有充足余量，同时防无限生成 |
| 解码方式 | **贪心**（见 3.1） | 结构化输出不需要采样 |
| 非法输出的后处理 | **不做任何修复** | L1 层测的就是合法率，修了就测不到了 |

**关于选择标准的一个关键细节，必须在论文里明说**：

C2 和 C3 的 **dev 目标分布不同**——C2 的 RSR 条目配的是流利目标，C3 配的是 repair-aware 目标。所以"整图 dev F1"必须理解成**每个条件对着自己的目标分布计算**，也就是每个模型在**它自己的训练目标**上选最好的检查点。

这是正确的做法（否则就是拿 C3 的标准去卡 C2，不公平），但**必须写出来**，否则读者会以为三个条件用的是同一把尺子。

> Checkpoints are selected on failure-penalised full-graph dev F1, computed against each condition's own target distribution. Metric A and Metric B are used for diagnosis only and play no part in checkpoint selection. The criterion was fixed before any test-set result was inspected.

---

## 七、测试数据怎么组织

### 7.1 三块测试数据

**（1）主测试分区** —— 自动生成的 RSR 样本，来自 PMB 的 test 划分。

**完全配对**：每一个自动生成的改口站点，都产出**两个表面形式**——一个不带 editing term，一个带——**共享同一个 gold 目标**。

**为什么在测试集配对**：配对让"有无标记"的对比可以用同一个源句作为配对单位做统计检验，这是 pilot 已经在用的分析方法。配对只在**推理**时增加成本，不增加任何训练成本。

规模：核心层 425 个站点 → 配对后约 850 行；全做完 815 个站点 → 配对后约 1,630 行。与 pilot 的 836 句同一量级，pilot 的分析代码和统计设计可以直接迁移。

**（2）挑战集** —— 手写或半自动，只进测试集。

**原设计只有两类，现在补上控制样本**，因为原来那一条 Berlin–Dublin 同时测了**多句解析、`CONTINUATION`、后续指代、改口**四样东西，失败了不知道该怪哪一样。三级设计：

| 类型 | 数量 | 来源 | 作用 |
|---|---:|---|---|
| 多句 + 指代，**无改口** | 15–20 | 多数可从 PMB 那 337 条含 `CONTINUATION` 的多句文档里挑 | 基线：跨句解析本身会不会 |
| 多句 + 改口，指代**只指向被采纳材料** | 15–20 | 手写，但结构简单 | 加了改口之后跨句还会不会 |
| 多句 + 改口，指代**同时指向采纳与撤回** | **40 →（建议扩到 80–100）** | 手写 | 核心主张的直接测试 |
| 跨话轮（cross-turn） | 40 | 手写 | 区分句内改口与语篇否认 |

> 🔴 **如果手写预算只够做一件事，做扩充第三类。**
> 论文最核心的主张（被撤回材料仍须在语篇中可及）目前**只有 40 条证据**——比尺寸效应薄，比标记词泛化薄。这是整个设计里证据最薄的地方，而它恰恰是主线。

**挑战集不做机械配对**。它们的表面形式是语篇设计的一部分（Berlin–Dublin 那条里的 `I mean` 和 `CONTINUATION` 结构是一起构思的），硬生成一个"无标记版"没有意义。

论文里的写法：

> All automatically generated repair sites in the main test partition receive paired marked and unmarked realisations; manually constructed challenge cases retain the surface form required by their discourse design.

**（3）硬负例集** —— 新增。数据本身现成，但**要付一个划分上的代价**（见本节末的 ⚠️）。

**要测什么**：模型有没有学到一个更基础的表面模板——

> 相邻两个竞争词 → 第一个进 `CORRECTION`，第二个进 `CONJUNCTION`

现有设计**完全没有排除**这个。而且自动生成的 RSR 测试集**全是正例**，所以 A1 的精确率本身几乎没有信息量。加入表面相似但不是改口的负例之后，"会不会误报"才第一次成为可测的量。

> 🔴 **这里有过一个泄漏 bug，修法和第一反应不一样。**
> 早先版本写的是"从 PMB gold 全部 11,940 条筛选，直接进测试集"。**这是数据泄漏**：train_gold 里的句子模型训练时见过（配着无 `CORRECTION` 的目标），在上面测误报是**测记忆，不是测泛化**；dev 里的还参与了检查点选择。
> 但"只从 test_standard 选"也行不通——实测数量不够。见下。

**候选的分布（实测）**：

| 划分 | 文档数 | 不含数字的逗号并置句 | 占比 |
|---|---:|---:|---:|
| test_standard | 1,194 | **30** | 2.5% |
| dev_standard | 1,194 | 24 | 2.0% |
| train_gold | 9,552 | **105** | 1.1% |
| test_long | 40 | 12 | 30.0% |

**只用 test_standard 那 30 条是不够的。** 里面有八条反义疑问句（`Tom is angry, isn't he?`）、若干状语前置（`In Colombia, Catholic customs prevail.`）和条件从句（`If you hurry, you will overtake him.`）——这些**根本不构成混淆**。真正同形的只剩 **8–10 条**，做不了任何检验。

**决定：在训练之前，把 train_gold 里那 105 条候选整体留出。**

**留出的单位是 PMB 文档，不是句子。** 这一点在当前数据格式下是自动满足的——我核对过：train_gold 的 9,552 条记录对应 9,552 个**唯一**文档 id，无重复，多句文档整篇装在一条记录的 `sentence` 字段里（88 条记录含一句以上）。所以**留出一条记录就是留出整个文档**，不存在"同文档的其他句子仍进训练"的漏洞。

这也和设计文档原有的原则一致：**"Split by source document, never by row"**。

**完整的留出清单（四条都要做）**：

1. 留出**包含候选句的整个 PMB 文档**；
2. 从 **C1、C2、C3 的全部训练数据中删除**——三个条件必须建立在**完全相同**的缩减训练集上，否则又引入一个新的条件间差异；
3. 从 **RSR 生成的源句池中删除**，否则它们的改口变体会从另一条路径漏回训练集；
4. 在**任何 smoke run 或检查点选择之前**把名单**冻结**并存档。

**结果**：留出的 105 篇 + test_standard 的 30 篇 = **135 篇候选**，筛选后估计 **40–50 篇**真正可用。训练集从 9,552 降到 **9,447**（−1.1%）。

> ⏰ **这条有时效。** 它改变的是数据划分，**必须在冻结语料之前定**，之后改不了。

> ⚠️ **这不是"免费"的，论文里要说清楚。** 它做了几件有代价的事：
> - **改动了 PMB 的标准训练划分**，所以本研究的数字与其他 PMB 解析器的公开数字**不再完全可直接比较**；
> - pilot 的流利基线（F1 0.950 / 合法率 97.5%）是在**完整**划分上测的，绝对值与新的 C1 不严格可比——**内部**比较（C1/C2/C3 之间）不受影响，因为三者用同一个缩减集；
> - 所有条件都必须在同一个缩减集上重建，这是一条必须执行到底的一致性约束。
>
> 论文里应当称它为 **custom held-out diagnostic set**，不要暗示训练用的还是原封不动的 PMB train split：
>
> > Surface-similar non-repair controls are held out at the document level before fine-tuning, yielding a custom training partition of 9,447 documents. All training conditions use this identical reduced partition; absolute figures are therefore not directly comparable with parsers trained on the standard PMB split.

**这个取舍值得做吗**——我的判断是值得：用 1.1% 的训练数据换一个**全新的评测维度**（模型会不会在像改口但不是改口的句子上误报），而这个维度目前完全没有被测到。但如果你更看重与既有 PMB 数字的可比性，可以只用 test_standard 那 8–10 篇做**定性探针**，不改划分——那样就拿不到任何统计量。**二选一，不能两全。**

**筛选标准**：保留同位语、无连词并列、两项以上的列举；剔除反义疑问句、状语前置、条件从句、日期。test_standard 里符合的长这样：

| 句子 | 为什么难 |
|---|---|
| `I am from Tokyo, Japan.` | 地名同位语 |
| `Goode was born in Exeter, Devon.` | 地名同位语 |
| `What altitude is Orange City, Florida?` | 地名同位语 |
| `I love thick, creamy mushroom soup.` | 无连词并列形容词——和**未加标记的形容词改口**同形 |
| `Lucy, an aristocratic, becomes a victim of Dracula.` | 同位语 |
| `I speak Japanese, English, and French.` | 列举 |

train_gold 留出的那 105 条里还有这些（同样自带 gold SBN）：

| 句子 | 为什么难 |
|---|---|
| `You have such beautiful, hazel eyes.` | 无连词并列形容词——和**未加标记的形容词改口**表面形状完全相同 |
| `This is a cargo ship, not a passenger ship.` | 两个竞争 NP 用逗号隔开，是对比不是改口 |
| `Machine guns were mounted on window-sills, rifles stacked between the mattresses.` | 无连词并列小句 |
| `I'm from Wellington, the capital of New Zealand.` | 同位语 |
| `Italy has two mountain ranges, the Alps and the Apennines.` | 同位语 + 列举 |
| `Tom, Mary and John were playing tag on the playground.` | 列举 |

**这批数据不需要生成、不需要标注**，带着原有的 gold SBN 直接用。

**期望结果**：模型在这些句子上应当输出**不含任何 `CORRECTION`** 的正常图。输出了改口结构，就是误报。

**怎么报分（注意不要重蹈覆辙）**：

硬负例集里 gold 的改口数**恒为 0**，所以在它上面单独算精确率／召回率／F1 **会全部退化**——这和流利 PMB 测试集的情况完全一样（见 8.5 的 A1 部分）。

| 想要的量 | 在哪个集合上算 |
|---|---|
| **误报率（false-alarm rate）** | **硬负例集单独算**——这是它的主要产出 |
| **A1 精确率** | 必须把 **RSR 正例与硬负例合并**成混合测试集再算 |

**绝不要**写"硬负例上的 A1 精确率"，那是一个退化的数。

**只报误报率，不要同时报 specificity。** 两者互为补数（specificity = 1 − 误报率），并列写等于把同一个数说两遍。选误报率，因为流利 PMB 测试集上报的也是它，两处口径一致。

**坏消息，要记下来**：还有一类硬负例——**标记词的非改口用法**（普通的 `actually`、普通的 `I mean`）——**没法从 PMB 取**。我数过：

| 标记词 | 在 PMB 全部 11,940 条 gold 里出现次数 |
|---|---:|
| `I mean` | **0** |
| `no wait` | **0** |
| `actually` | **1** |
| `sorry` | 10 |
| `well` | 26 |

PMB 是书面语料，口语标记词几乎不出现。这一类只能**手写，而且要自己标 SBN**。成本不低，**优先级排在扩充 anaphora 挑战集之后**。

**（4）流利 PMB 测试集** —— 原封不动的 PMB test_standard，1,194 条。

用来测"学了改口之后，普通解析有没有变差"。

### 7.2 挑战集测的是什么（要讲清楚，不然读者会以为任务变了）

先看一个完整的手写目标：

```
% 第一句：What's the weather like in Berlin, I mean, in Dublin?
entity.n.01 EQU ?
be.v.01 Theme -1 Time +1 Co-Theme +2
time.n.08 EQU now
weather.n.01
    CORRECTION <1
city.n.01 Name "Berlin" LocationOf -1        % [被撤回]
    CONJUNCTION <2                            % I mean,
city.n.01 Name "Dublin" LocationOf -2        % [被采纳]
% 第二句：Show me the airports of these two cities.
    CONTINUATION <1
person.n.01 EQU hearer
show.v.04 Agent -1 Recipient +1 Theme +2
person.n.01 EQU speaker
airport.n.01 Location +2
quantity.n.01 EQU 2
city.n.01 ANA -7 ANA -6 Quantity -1          % these two cities
```

**输入是两个话轮，作为一个字符串。输出是一条 SBN 序列覆盖两个话轮**，用 `CONTINUATION <1` 把第二句接到**被采纳的**那个语境上。

**任务形式没有变。** PMB 的一个"文档"本来就可以含多个句子——gold 训练集里有 **337 / 9,552（3.53%）**的文档含 `CONTINUATION`。所以模型在流利训练阶段**已经见过多句输入**，只是罕见。任务定义自始至终是"文档 → SBN 序列"，不是"句子 → SBN 序列"。挑战集没有改变任务，只是采样到了训练分布的稀疏尾部。

**它测的是两件事**：

1. **跨句解析泛化**：能否在只占训练数据 3.5% 的构型上工作。
2. **表示法对后续指代的支持**——这才是核心。注意 `city.n.01 ANA -7 ANA -6`：**两条**指代边，一条指向 **Dublin（被采纳）**，一条指向 **Berlin（被撤回）**。"these two cities" 的复数先行词**横跨了撤回／采纳的界线**。

第 2 点是整套形式化方案的核心主张——**被撤回的材料虽然不被主张，仍然必须在语篇中可及**——的唯一直接测试点。

---

## 八、怎么评分

### 8.1 先讲清楚为什么不能只报一个分数

这是整节的论证支点，论文里必须放在最前面，否则后面所有的复杂性看起来都像过度设计。

设想一个解析器，输入 `A goat, I mean, a cow gave birth to a calf`，它输出了一个**完全正确的流利句 SBN**——只有 cow，没有 goat，没有 `CORRECTION`。

- 它把改口**无声地正常化掉了**。
- 从"说话人最终承诺的语义"看：**完全正确**。
- 从"有没有识别出改口"看：**完全失败**。

单个分数会给它一个中不溜的数（实测夹具是 **0.67**），而这个 0.67 同时也可能是"发现了改口但内容写错"的分数。**两种性质相反的失败被压成了同一个数字。**

拆开之后，这个解析器是 **Metric A = 0.00 / Metric B = 1.00**，另一种失败则正好相反。

**措辞纪律：不要用 "orthogonal"（正交）这个词。** 两个指标会**互相影响**（见 8.6），统一用：

> complementary but interacting evaluation dimensions

### 8.2 三个层次

| 层 | 问的问题 | 数字 |
|---|---|---|
| **L1 结构合法性** | 输出还是不是一个合法的 SBN 图？ | 解析状态的分布 |
| **L2 整图准确度** | 完整的 repair-aware 图对不对？ | 整图 SMATCH++ F1 |
| **L3 拆分** | 错在改口结构上，还是错在最终语义上？ | Metric A / Metric B |

L2 是"显然会被问到所以报出来"的基线数字。**主要结论在 L3。**

### 8.3 L1：结构合法性

报"有多少比例的输出能被读成良定义的 SBN 图"。

**关键约定：生成失败不许从分母里消失。** 不能解析的输出在 Metric A 里当作"什么都没检测到"，**损失的是召回率**。在 L2 里记为失败。

**为什么这条重要**：pilot 显示模型的典型失败是**过度生成结构**（836 条里 205–230 条含环）。如果把崩溃的输出排除在外，一个"越垃圾越爱崩溃"的模型反而分数更好看。

### 8.4 L2：整图 SMATCH++

拿预测图和完整的 repair-aware gold 图比，算精确率／召回率／F1。

**一个必须写进论文的技术细节**：评分器必须配 `GenericStandardizer`。不配的话，评分工具**不会**把反向角色（`:X-of`）标准化，于是每一个反向角色都被当成完全不同的边。实测：两个只在 `Theme +1` vs `ThemeOf -1` 上不同、其余完全相同的图，默认打 **93.33**，加了标准化器打 **100.0**。PMB gold 里约有 2,600 条反向边。

**这不是偏好问题，是正确性问题。** pilot 的分数已经用修复后的配置全量重跑过了（修复前 clean 条件 F1 是 0.5519，修复后是 0.9746），RP 里引用的是修复后的数。

### 8.5 Metric A：改口结构恢复得怎么样

**它问的是先于语义的问题**：解析器注意到改口了吗？隔离对了吗？合并到对的语篇位置了吗？

**Metric A 对所有条目都跑**，包括挑战集。这正是拆成两个指标的目的。

#### A1. 检测（item 级二值）

预测里到底有没有句内改口？按条目聚合成精确率／召回率／F1。

判定是**纯结构性**的，不看词：一个句内改口必须有一个**兄弟 CONJUNCTION**——即从**同一个源盒子**出发的 `CONJUNCTION` 连接。跨话轮的 `CORRECTION`（真正的语篇否认）没有这个兄弟，因此不会被误判成改口。

**两个报告上的注意事项**：

- **在全是 RSR 的测试集上**，gold 的改口数恒为正，**精确率信息量很低**，标注为 not informative 并说明原因，主报召回率。
- **在流利 PMB 测试集上**，gold 改口数恒为 0，精确率／召回率／F1 会**全部退化成 0**，毫无信息量。那里真正要报的是 **false-alarm rate（误报率）**＝有多少比例的流利输入被错误地赋予了改口结构。
  **这个数目前的聚合函数还不输出，需要加。** 原材料都在（每条的预测改口数都存了），大约三行代码。
  这个数恰好是设计文档那条验收测试的量化形式，也恰好是问题三的核心证据。pilot 已经量好了对照基线（流利条件 F1 0.950 / 合法率 97.5%）。

**一个要写进论文的非对称性**：`CONJUNCTION` 索引偏移一位（本该 `<2` 写成 `<3`）**不会**表现为"合并位置错"。它会让 `CONJUNCTION` 不再是 `CORRECTION` 的兄弟，于是**根本检测不到改口**，错误落在 **A1 召回率**上。这是正确行为，但意味着**索引写错是在检测层而不是合并层现形的**。

#### A2. 被撤回材料的范围（故意报两遍）

哪些概念被隔离进了 `CORRECTION`。**同一件事用两种方式量，两个都报。**

| 指标名 | 是什么 | 性质 |
|---|---|---|
| **reparandum-content F1** | 被隔离元素的**多重集**，不看位置 | 宽松 |
| **positioned reparandum-content F1** | `{(线性概念索引, 元素)}` 的集合 | 严格 |

（**注意用 "content" 而不是 "token" 或 "span"**。原来的 token/span 命名会被读成自然语言的词和字符区间，那是不相干的东西。"content element" 定义为：被 `CORRECTION` 隔离的概念，**加上**它引入的常量。）

**为什么两个都要**：

- **只看集合抓不到"写反了"**。在时体类改口里，被撤回的和被采纳的**含有完全相同的概念**，把两者对调，集合分数照样是 1.0。
- **只看带位置的版本又太脆**。句子前面掉一个概念，后面**每一个索引都平移**，一个语义完全正确的改口会被打成 0。
- **配对读**：集合分高 + 带位置分低 = "材料抓对了，位置放错了"。这个落差本身就是诊断信号。

**为什么常量要算进去**：不算的话"时间状语"这一类直接不可评分——被撤回的和被采纳的都是 `time.n.08`，唯一区分 `monday` 和 `tuesday` 的就是那个常量。

#### A3. 合并语篇位置

从根盒子走到"改口并入的那个盒子"的**分隔符路径**（根盒子是 `()`，否定域下是 `("NEGATION",)`），按元组精确匹配。

**为什么需要它**：这是记法明确标记为**静默失败**的那一类错误——把改口放进主盒子而不是放在否定域下，**图能干净解析，意思完全不同**。没有这个子指标，这类错误只会表现为少了几条边，看不出性质。

**为什么用路径而不是盒子编号**：预测图和 gold 图的盒子数量可能不同，编号不可比，路径可比。

**一个已量化的局限**：路径不是唯一的——如果同一个父盒子下挂着两个带相同分隔符的兄弟盒子，两者路径相同。我在 PMB gold 全部 9,550 条可解析图上测过：**23 条（0.24%）**存在这种配置，而且只有当改口恰好并入这两个之一时才真正成为问题。**建议在附录声明这个边界，不改指标。**

#### A3 之外：Metric A 不评什么

**Metric A 明确不评价改口内部的语义角色是否正确。** 那由 L2 整图分数和 Metric B 承担。

这是刻意的分工：如果 Metric A 也去评角色，它就会和整图分数高度相关，拆分就失效了。论文里要写一句明说。

### 8.6 Metric B：说话人最终承诺的语义对不对

**做法**：把预测图里的改口脚手架**在图的层面**拆掉，得到一个"干净图"，再和**自然流利句的 SBN** 比。

**注意这不是"按 triple 归属划分"**。不是去判断哪些 triple 属于原图、哪些属于改口，而是**做一次结构手术，然后和一个独立存在的参照物比对**。

**为什么必须在图层面做**：SBN 解析时已经把每个 `+n/-n` 解析成了具体的节点引用，所以**删节点不需要重编号**。反过来说，**绝不能转回 SBN 字符串**——那才需要重算每一个索引，明确排除在范围之外。

**拆掉什么**：

1. `CORRECTION` 子树（被撤回材料），连同嵌套的盒子一起。
2. `CONJUNCTION` 盒子，压平并入它的合并目标。
3. 两种镜像的脚手架占位节点（一种在改口盒子内部、是等价边的源；一种在父盒子里、是候选项等价边的目标）。

**这两种占位节点靠结构特征识别，绝不靠词。** 因为 `entity.n.01` 是这套记法的通用占位符（gold 里出现 1,315 次），绝大多数时候表示的是**真实内容**（代词、回指）。按词认会把真实内容删掉。

**跨话轮的 `CORRECTION` 不拆**，因为它是真实的语篇内容，靠"有没有兄弟 CONJUNCTION"区分。

**定义域边界：两道闸门**

Metric B 是一个**偏函数**——它有明确的、可机器检查的适用范围，范围外的条目返回"不适用"。这是特性不是缺陷：主张不是"所有改口都能清洗"，而是"这是可清洗的那一部分，这是它的边界"。

| 闸门 | 拒绝什么 | 为什么 |
|---|---|---|
| **闸门一** | `CONJUNCTION` 盒子上挂着"合并之外"的东西 | 压平会溶解这个盒子，所有碰到它的边都得改指到合并目标上，而这种改指正是记法认为会**改变意义**的操作 |
| **闸门二** | 有存活节点指进被撤回材料 | 典型情形是一个回指同时够到被撤回和被采纳的材料。这类图**没有无争议的干净读法** |

**闸门一划的界线，恰好是"改口结束于本句"和"改口之上还建了语篇"的分界。** 这也正是挑战集的定义。

**注意 Berlin–Dublin 那个例子两道闸门同时触发**：`ANA -7` 指进被撤回材料（闸门二），`CONTINUATION <1` 让 `CONJUNCTION` 盒子成为分隔符的目标（闸门一）。这不是巧合，是设计预见到的。

**分区在建表时固定，不在评分时算**：`core`（两个指标都跑）／`challenge`（Metric B 不跑）。这样如果闸门以后改了，变化会表现为 gold 表里的一处 diff——这是想要的——而不是条目在指标之间**静默漂移**。

**失败策略**：不能解析的、或结构落在定义域外的预测，得到 `NaN` 加一个说明**是哪一种**的状态。汇总时**两个均值都打印**，但**引用被惩罚的那个**（失败记 0）作为主要数字。理由一句话：只算成功样本的均值**在奖励一个大声失败的解析器**。

### 8.7 挑战集上跑哪些指标（这里之前记错过，以代码为准）

| 指标 | 挑战集上是否计算 |
|---|---|
| L1 结构合法性 | **跑** |
| L2 整图 SMATCH++ | **跑**（挑战集有完整的 repair-aware gold 图，这个指标有定义） |
| Metric A | **跑** |
| Metric B | **不跑**（唯一被分区门控的指标） |
| 指代边量化（见下） | **跑**（仅 anaphora 那 40 条） |

论文里的写法：

> Challenge items receive structural-validity, full-graph SMATCH++ and Metric A evaluation. Only Metric B is undefined for them: stripping is refused by an explicit gate either because an anaphor reaches into the reparandum, so that removal would leave a dangling dependency with no uncontroversial clean reading, or because discourse structure is built on top of the repair. Anaphora is additionally examined qualitatively.

### 8.8 指代边的量化评分（新增，需要写代码）

**这是问题二（表示法的收益）的主要证据，不是补充观察。** 因为它是"被撤回材料仍须在语篇中可及"这个核心主张的**唯一直接测试**。

对 anaphora 挑战条目（现有 40 条，**建议扩到 80–100**），检查预测图里从回指概念出发的 `ANA` 边，判三个二值量：

1. **指向被采纳材料是否命中**
2. **指向被撤回材料是否命中** ← **这一条是关键**
3. **两条都命中且没有多余的指代边**

**匹配方式**：按 `概念 + 识别性属性`（比如 `city.n.01` + `Name "Berlin"`），**不是**裸 `city.n.01`，也**不是**按线性索引（索引会随概念数变动而平移）。如果同一个图里出现重复签名，标记为 ambiguous 单独报，不要硬匹配。

**统计方法**：

- 比较两个模型：**exact McNemar 检验**（同一批 40 条上的配对观测）
- 报告单个模型：**exact binomial 置信区间**
- **不要用 Fisher 精确检验**——那适用于独立组，不适用于同条目配对比较

**为什么第 2 条是关键**：一个把被撤回材料当成"已删除"处理的解析器，在第 1 条上可能命中，在第 2 条上**必然失败**。

**这一条就是 C2 与 C3 的分水岭**：C2 的训练目标里没有被撤回的材料，**它从未被监督去保留那个指称、也从未被监督去赋予它语篇可及的地位**，所以预期它在第 2 条上的命中率很低。C3 被监督去做这两件事。这个差距就是"保留 reparandum 换来了什么"的答案。

**但要按实测报告，不要按定义断言**（见第二节的 ⚠️）：C2 仍可能从输入里复制出被撤回的词。若真的出现，单独分析，不要当成噪声。

### 8.9 oracle-cleaned 参照条件

**做法**：用语料构造时记录的信息，把被撤回材料和标记词删掉，得到原始流利句，交给**只在流利数据上训练的**解析器（C1）解析。

**这几乎是免费的**：数据生成器把原始流利句 `nl_clean` 存在每一行里，删除的结果就是它本身。**不需要新模型，不需要 span 标注，只需要跑一次推理。**

**它的参照物和 Metric B 完全相同**（都是流利句的 gold SBN），所以两者直接可比。

**措辞纪律：叫 reference condition，不要叫 upper bound（上界）。**

**为什么不是上界**：C3 训练时见过更多样本，它在 Metric B 上**完全可能高过**这个参照。而且两者的差距**不能**严格分解成"检测误差"和"表示误差"。

论文里的写法：

> An **oracle-cleaned reference condition** is reported alongside Metric B. It indicates what accepted-meaning performance looks like when repair identification is free, and it uses information unavailable at inference time. It is not a bound: a repair-aware parser trained on more data may exceed it, and the gap does not decompose additively into detection and representation error.

> ⚠️ **oracle-cleaned 不能兼任 pipeline 对比。** 它用了推理时拿不到的 gold 改口信息，所以回答不了"现实系统里 pipeline 和端到端谁强"。那需要下面 8.10 的**预测式** pipeline。文档早先版本有让 oracle-cleaned 含糊兼任这个角色的嫌疑，已改正。

### 8.10 预测式 pipeline baseline（建议做，比看上去便宜）

**要回答的问题**：传统的"先检测、再删除、再解析"流水线，到底有多强？C3 的优势是不是只来自**避开了检测器的错误**？

**流程**：

```
RSR 句 → 零样本改口检测器 → 删掉预测出的 span → C1 解析 → Metric B
```

**不需要实现 reinsertion**。只比最终语义（Metric B），就足以回答上面那个问题。

**为什么便宜：检测器已经写好了。** `colloquium_prep/ablation_repair_sensitivity/` 里是一整套现成的东西：

| 文件 | 作用 |
|---|---|
| `run_inference_ablation.py` | 用一个"知识注入"提示词做**零样本改口检测与定位**，输出模型标出的 reparandum / interregnum 位置 |
| `ground_truth.py` | 通过对比原句和注入后的句子（diff）自动推出 gold span，**不需要新标注** |
| `evaluate_ablation.py` | 带 McNemar / Friedman / Holm 的评分 |

所以成本是：**一次检测推理 + 一次解析推理**。不需要新模型、不需要新标注、不需要训练。

**一个必须处理的混淆：检测器的家族。** pilot 那个探针跑在 **Gemma-2** 上，主实验用 **Qwen3.5**。直接比，会把检测器家族、提示词和模型能力的差异全混进来。

**修法比"放软措辞"好**：那个探针是**零样本**的，换个基座就行。**把同一个检测提示词跑在 Qwen3.5 上**，家族混淆直接消失，成本还是一次推理。

**即便如此，结论的范围仍然有限**：

> This is a comparison with **one instantiated zero-shot detect-then-parse pipeline**, not with detect-then-parse approaches in general. A stronger detector, or one trained for the task, could change the comparison.

**措辞纪律**：不要预先写"它应当在无标记样本上崩掉"。pilot 的确测出这个检测器在无标记条件下识别率只有 44.8%，所以**按标记有无分层报告**是合理的设计；但把预期写进计划里读起来是结果导向的。按分层**报**，不要按分层**预言**。

**如果最后来不及做**：必须在限制里**明确写出来**（见 10.9），不能让 oracle-cleaned 含糊地顶替。

### 8.11 哪个实验条件看哪些数

| 要回答的问题 | 主要看 | 辅助看 |
|---|---|---|
| **一、能学会吗** | C3 的 Metric A **绝对值** + 整图 F1 + 错误类型分布 | 按转换装置分层看 |
| **二、收益：保留换来了什么** | C2 vs C3 在 anaphora 挑战集上**指向被撤回先行词的命中率** | 三类挑战控制样本的对比 |
| **三、代价：保留花了什么** | (a) C2 vs C3 的 **Metric B**；(b) 流利 PMB 上的**误报率** + 整图 F1，**三路分解**（见下） | 和 pilot 基线 0.950 / 97.5% 比 |
| **四、泛化到哪为止** | 标记词配对差值、`sorry` 上的表现、**硬负例上的误报率**、混合集上的 A1 精确率、**词对留出** | 挑战集、指代边命中率 |
| **（对照）pipeline 有多强** | 预测式 pipeline 的 Metric B，按有无标记词分层 | oracle-cleaned 参照条件 |
| **（尺寸）** | 三层**分别**报：合法性 / Metric B / Metric A | 单 seed，**仅描述，不做检验** |

**为什么尺寸要分三层报**：否则整图 F1 涨了，说不清是"更会生成合法 SBN"、"更懂改口"还是"普通解析本来就更好"。

**流利输入上的代价要三路分解**，三个对比各答一件事：

| 对比 | 回答的问题 |
|---|---|
| C1 vs C2 | 只加 RSR 输入 + 正常化训练，影响是什么 |
| **C2 vs C3** | **目标表示法**对流利解析和误报的影响（配平的那一组） |
| C1 vs C3 | 整个 repair-aware 介入的**总代价** |

**注意一个约束**：C2 只在 9B 上跑，所以**这个三路分解只在 9B 上可得**。2B 和 27B 只有 C1 vs C3，也就是只有"总代价"这一路。

**关于精确率算在哪里**：在全是正例的 RSR 测试集上，或在全是负例的硬负例集／流利集上，精确率都会退化。**误报率**在纯负例集上算，**精确率**必须在正负混合的集合上算。（同样的退化问题在 8.5 的 A1 部分已经讲过一次。）

---

## 九、统计怎么做

### 9.1 已经在做的（pilot 建立的惯例，继续沿用）

- 配对单位是**源句**，不是每一行。同一个源句产生的多个变体**绝不**当成独立观测——先在源句内平均，再做检验。
- 连续量（F1）用 **Wilcoxon 符号秩检验**；二值量（是否合法、是否检测到）用 **McNemar 检验**。
- 多重两两比较用 **Holm 校正**。

pilot 就是这么做的：2,499 条输入先折成 833 个配对观测再检验。

**一处要改的**：pilot 报的效应量是 **Cohen's $d_z$**，但它和 Wilcoxon 配不上——Wilcoxon 是**秩**检验，$d_z$ 是基于**均值差**的参数型效应量。

**主实验改用 rank-biserial correlation**（秩二列相关），它是 Wilcoxon 的配套效应量。

**连带影响**：pilot 的效应量要一并重算，否则文中两处用的是不同的效应量。pilot 的分数表已经用修复后的评分器重跑过一次了，再算一个效应量是几行代码，比解释为什么不一致省心。

### 9.2 新增的：训练随机性

之前完全没有估计——pilot 只跑了一个种子。

**方案**：

| 尺寸 | seed 数 |
|---|---:|
| 2B | 1 |
| 9B | **3** |
| 27B | 1 |

**怎么用这三个 seed**（这一点之前搞错过，以下是修正后的做法）：

1. **每个条目的分数先跨三个 seed 求平均**；
2. 在**源句层面**对两个条件做**一次**配对检验；
3. 同时报告三个 seed 各自的条件均值，以及 mean ± SD；
4. **逐 seed 的检验降级为稳健性检查**，放附录。

**为什么不能"对每个 seed pair 分别检验、看三次是否一致"**：三次检验高度相关，一致性不构成独立证据，实际上是在做一个没有校正的多重检验。

**但要诚实说明这个方案能做到什么、不能做到什么。**

先跨 seed 平均、再对条目做配对检验，**主要估计的仍然是测试样本的变异**。它**没有**把三次训练当成真正的独立复制，也**没有**完整传播训练随机性。

所以结论只能写成：

> item-level paired inference averaged over three trained models

**不能**写成"已经完整估计了训练随机性"。三个 seed 的 mean ± SD 是对训练变异的一个**描述**，不是一个被传播进显著性检验的**不确定性来源**。

**27B 和 2B 单 seed 的处理**：任何涉及它们的差异**只能描述，不能做运行层面的推断**。论文里写明：

> Three seeds are trained at 9B, where the central comparison is located. Single runs are reported at 2B and 27B; training variance is not estimated there, and differences involving those sizes are described rather than tested.

### 9.3 校正的范围

按**研究问题分族校正**，而不是把所有比较塞进一个族：

| 族 | 包含的比较 |
|---|---|
| 收益族 | C2 vs C3 的指代边命中率；三类挑战控制样本之间 |
| 代价族 | C2 vs C3 的 Metric B；流利 PMB 上的 C1/C2/C3 三路对比 |
| 泛化族 | 有无标记词；四个训练标记词之间；`sorry` vs 已见标记词；硬负例上的误报 |

否则统计功效被无谓地吃掉。

> ⚠️ **尺寸不构成一个校正族。** 早先版本把"尺寸族"列了进去，这和"2B/27B 单 seed、只作描述性比较"自相矛盾——**不做推断检验就不需要多重校正**。尺寸只报三层指标的描述性趋势，不报 p 值，因此不进 Holm。

**指代边那一族用的检验不同**（见 8.8）：exact McNemar 比较两个模型，exact binomial 置信区间报告单个模型，**不用** Fisher 精确检验——后者适用于独立组，不适用于同条目配对比较。

---

## 十、必须在论文里写明的限制

这一节是给写论文时直接抄的。**每一条都是主动声明，不是等人来问。**

### 10.1 两个训练条件的数据量不等

C1 是 9,447 篇，C3 更多。这个混淆**不可能**靠配平消除（见 5.4），只能靠 C2 隔离。

> The two supervision conditions necessarily differ in training volume, since removing the fluent negatives from the mixture would remove the very signal the design depends on. C1 vs C3 therefore measures *the effect of adding repair-aware training data*. The term *supervision effect* is reserved for C2 vs C3, which holds inputs, example count and optimizer steps fixed and varies only the target representation.

**附带一条要报的数**：C3 的目标串更长，所以 C2 和 C3 虽然优化步数相同，**携带损失的目标 token 总数并不完全相同**。这个差异**无法消除，因为它就是处理本身的组成部分**。报告双方的目标 token 总量即可。

### 10.2 尺寸比较不是纯粹的尺寸效应

见 6.1。固定超参数固定的是协议，不是适配容量。

### 10.3 held-out 标记词的混淆

`sorry` 同时是**词汇上新**和**语用上不同**（它是道歉标记，训练用的四个不是）。五个标记词的集合**不足以**把这两个因素分开。

**能做的是把失败模式分开看**：

- 失败集中在**检测召回率**上 → 指向**词汇触发器**，模型确实在靠词表；
- 检测正常但**被撤回材料范围**变差 → 指向**语用差异**，模型认出了是个线索但对它的功能判断不同。

> A fifth marker, *sorry*, is held out of training. The set of five is not large enough to separate lexical novelty from pragmatic difference, and this confound is stated rather than controlled. The failure mode is diagnostic even so.

### 10.4 挑战集的规模

现有 40 条 anaphora + 40 条 cross-turn，另加两类各 15–20 条的控制样本。这个规模只能报比例和精确检验，**做不了精细比较**。

**而且要承认一件更要紧的事**：论文的核心主张（被撤回材料仍须在语篇中可及）**只压在 anaphora 这一类上**。若最终仍是 40 条，必须在限制里明写证据基础的规模，不要让读者以为这个断言有大样本支撑。

### 10.5 语料本身的边界

- 只覆盖**紧邻的替换型**改口，不覆盖延迟改口、非对齐概念。
- 词汇和分布来自**书面的** PMB 句子，与真实口语不同。
- 时体类改口**不在**主实验范围内（见 5.3）。
- 改口位置（句首／句中／句末）**刻意不作为采样轴**——pilot 发现的位置效应是注入过程的产物（句首插入读起来像呼语，句中像无连词并列，句末像同位语），按它分层会（a）在一个假象上设配额，（b）等于给模型一个表面线索。

> The experiments test whether parsers learn the proposed representation within these boundaries; they do not establish coverage of every naturally occurring RSR.

### 10.6 Metric A 和 Metric B 不独立

把跨话轮的否认误判成句内改口，会被**罚两次**：一次在 A（误识别），一次在 B（因为拆解于是删掉了说话人确实承诺了的内容）。

这不是 bug，是任务的真实结构。**但报告时必须说**，否则读者会把两者当成独立证据来源，从而高估证据强度。

### 10.7 评分模块副本的漂移风险

pilot 用的是 SBN 模块的**冻结快照**（为了让已发表的数字保持可复现），主实验用的是权威副本。如果权威副本的角色定义有更新，两份副本会**静默产生评分差异**——症状不是报错，是数值不同。

**处理办法**：主实验的全部数字（包括重训后的 C1）一律用权威副本重新评分；pilot 的数字以"pilot，冻结快照评分"的身份保留，**不与主实验的数字放在同一张表里直接相减**。

### 10.8 泛化的主张只能到哪一步

即使做齐了标记词留出、词对留出和硬负例，**也不等于证明了模型学的是"结构"而不是"模板"**——只是排除了三种已知的、具体的捷径。

而且有一类硬负例**做不到**：标记词的非改口用法（普通的 `actually`、普通的 `I mean`）。PMB 里这些词分别只出现 1 次和 0 次，无法从语料中取，只能手写并自行标注 SBN，优先级低于扩充 anaphora 挑战集。

**允许写的**：

> generalisation beyond seen editing markers, seen lexical pairs, and surface configurations that resemble repair without being repair

**不允许写的**：任何形式的 "the model learns the structure rather than a template"。

另外，**非相邻改口**（reparandum 和 repair 中间隔着已承诺的材料）明确不在覆盖范围内，因此关于"结构而非邻接"的主张也无法做出。

### 10.9 预测式 pipeline baseline（若未实施）

oracle-cleaned 参照条件使用了推理时**不可获得**的 gold 改口信息。它回答的是"若清洗免费会怎样"，**不能**回答"现实系统里 pipeline 与端到端谁更强"。

若 8.10 的预测式 pipeline 最终没做，必须写明：

> No pipeline baseline with predicted repair spans was run. The oracle-cleaned condition bounds neither direction of that comparison, since it removes detection error by construction. Whether a detect-then-parse pipeline would match the end-to-end parser on accepted meaning therefore remains open.

**不要**用 oracle-cleaned 的数字去暗示 pipeline 的表现。

### 10.10 训练用的不是原封不动的 PMB 划分

为了让硬负例不泄漏，105 篇表面相似的非改口文档在训练前被**按文档留出**，训练划分因此是 **9,447** 篇而非标准的 9,552 篇。

**三个后果，都要写明**：

1. 本研究的**绝对数字**与在标准 PMB 划分上训练的解析器**不再完全可直接比较**；
2. pilot 的流利基线（F1 0.950 / 合法率 97.5%）是在完整划分上测的，与新的 C1 不严格可比——**内部**比较（C1/C2/C3 之间）不受影响，因为三者共用同一个缩减集；
3. 这个缩减集对**所有条件**都必须完全一致，否则会引入一个新的条件间差异。

> Surface-similar non-repair controls are held out at the document level before fine-tuning, yielding a custom training partition of 9,447 documents. All training conditions use this identical reduced partition. Absolute figures are therefore not directly comparable with parsers trained on the standard PMB split; comparisons within this study are unaffected.

**替代方案**（若更看重与既有数字的可比性）：不改划分，只用 test_standard 里那 8–10 篇做**定性探针**。代价是拿不到任何关于误报的统计量。二选一。

---

## 十一、接下来要做的事

按顺序，**上一项没完成不要开始下一项**：

### 第 1 步：smoke run（唯一的瓶颈）

跑 2B / 9B / 27B 三个，记录 6.2 那张表里的全部项目，**特别是终止符行为**。

**在拿到每步秒数和显存峰值之前，"13 次运行可行"不是一个可回答的问题。**

### 第 2 步：据实测速度确定运行数

13 次还是 17 次。这个数决定 Model 节里条件表怎么写，所以必须在写正文之前确定，否则要返工。

### 第 3 步：冻结语料范围（不需要新信息，今天就能做）

- 时体移出主实验
- NLI 降级为保守启发式
- 确定第二层扩展做到哪一格

### 第 4 步：补评分代码

- 聚合函数加 **false-alarm rate**（约三行）
- **指代边量化评分**（约二十行，复用现有图机制）
- 效应量从 $d_z$ 改为 **rank-biserial correlation**，并重算 pilot 的效应量

### 第 5 步：改数据生成器

- 测试集**完全配对**（同一站点两个表面形式）
- 训练集标记词**均衡采样**（四个词之间），`sorry` 完全排除
- 训练集保持约 30% 带标记，来自不同站点
- **加词对留出约束**：同一 WordNet 词对不跨划分（采样时一个过滤条件）

### 第 5b 步：硬负例（**有时效，必须在冻结数据前做**）

1. 在 train_gold 里找出那 105 篇"不含数字的逗号并置句"文档，**训练前按文档整体留出**；
2. 从 **C1 / C2 / C3 的全部训练数据**中删除——三个条件必须用**完全相同**的缩减训练集；
3. 同时把它们**从 RSR 生成的源句池里排除**（否则改口变体会从另一条路漏回训练集）；
4. **冻结名单并存档**，且必须在任何 smoke run 或检查点选择**之前**完成；
5. 和 test_standard 的 30 篇合并，筛掉反义疑问句／状语前置／条件从句／日期，估计剩 **40–50 篇**；
6. 带着原有 gold SBN 直接进测试集。

训练集从 9,552 降到 **9,447**（−1.1%）。**这不是零成本**：它改动了 PMB 的标准划分，所有条件必须在同一个缩减集上重建，且与其他 PMB 解析器的公开数字不再完全可比（见 7.1 的 ⚠️ 和 10.10）。

### 第 5c 步：补挑战集控制样本

- 多句 + 指代、**无改口**：15–20 条（多数可从 PMB 那 337 条多句文档里挑）
- 多句 + 改口、指代**只指向被采纳材料**：15–20 条（手写，结构简单）
- **若有余力，把 anaphora 主类从 40 扩到 80–100**——这是整个设计里证据最薄、却最承重的一块

### 第 5d 步：接上预测式 pipeline baseline（可选但建议）

复用 `ablation_repair_sensitivity/` 里已有的零样本检测器，跑一遍新测试集 → 删 span → C1 解析 → Metric B。两次推理，无训练。若不做，按 10.9 写明限制。

### 第 6 步：重写论文的 Model / Data / Timeline 三节

**正文只放**：研究问题、实验条件、指标是什么、统计方案。

**附录放**：闸门的细节、剥离器的机制、LoRA 模块列表、终止符检查、评分实现、可训练参数量。

### 第 7 步：更新时间表

原时间表从 8 月 24 日起算，前两个区间已经过去，而且它们的交付物有一部分还没完成。修订后的大致安排：

| 期间 | 工作 |
|---|---|
| 9/19–10/4 | smoke run；实现第二层算子（常量、边标签）；冻结语料范围 |
| 10/5–10/18 | 生成全语料；测试集配对；**冻结数据** |
| 10/19–11/15 | 13（或 17）次微调；边完成边评测 |
| 11/16–11/29 | 完成评测、指代边量化、按装置的错误分析 |
| 11/30–12/13 | 论文定稿与校对 |
| 12/14 | 提交 |

---

## 十二、关键数字速查

**语料**

| 项目 | 数字 |
|---|---:|
| PMB 5.1.0 gold train / dev / test | 9,552 / 1,194 / 1,194 |
| PMB 全部数据里 `CORRECTION` 出现次数 | **0**（共 300,133 条） |
| PMB gold train 里 `CONJUNCTION` 出现次数 | 73（0.76%） |
| PMB gold train 里含 `CONTINUATION` 的多句文档 | 337（3.53%） |
| 训练划分里合法的改口站点 | 16,425 |
| 只能用移位装置的站点 | 525（481 形容词、44 动词、0 名词） |
| RSR 训练集（核心层 / 全做完） | 2,200 / 4,020 行 |
| 改口 : 流利 比例（核心层 / 全做完） | 1:4.3 / 1:2.4 |
| 挑战集（主类） | 40 anaphora + 40 cross-turn |
| 挑战集控制样本（新增） | 15–20 + 15–20 |
| 逗号并置句（去数字后）train / dev / test | 105 / 24 / 30 |
| 硬负例：train 留出 105 + test 30，筛后估 | **40–50 篇** |
| **留出后的训练集（三个条件共用）** | **9,447 篇**（−1.1%） |
| train_gold 记录数 = 唯一文档 id 数 | 9,552 = 9,552（留出单位＝文档） |
| PMB gold 里 `I mean` / `no wait` / `actually` 出现次数 | **0 / 0 / 1**（共 11,940 条） |

**模型**

| 项目 | 值 |
|---|---|
| Qwen3.5 稠密尺寸 | 0.8B / 2B / 4B / 9B / 27B |
| Qwen3.8 稠密尺寸（2026-08） | **仅 27B**（故不适用于尺寸比较） |
| 用 instruct 还是 base | **instruct** |

**pilot 基线**

| 条件 | 合法率 | F1（失败记 0） |
|---|---:|---:|
| 流利句 | 97.5% | 0.950 |
| RSR | 94.6% | 0.786 |
| RSR + "I mean" | 76.5% | 0.609 |

*识别探针（未微调的基座模型，只用 "I mean"）：有标记 100%，无标记 44.8%（按位置 30.7–64.2%）*

**训练**

| 项目 | 值 |
|---|---|
| LoRA r / alpha / dropout | 32 / 64 / 0.05 |
| 学习率 / epochs / 最大长度 | 2e-4 / 3.0 / 1024 |
| 有效批大小 | 16（固定） |
| 运行总数 | 13（算力允许则 17） |

**评分**

| 项目 | 值 |
|---|---|
| 评分器必须配 | `GenericStandardizer` + ILP 求解器 |
| 不配的代价 | 反向角色图 93.33 而非 100.0 |
| 合并路径歧义率 | 23 / 9,550（0.24%） |
| 剥离器的两道闸门 | CONJUNCTION 边界；悬空指向被撤回材料 |

---

## 十三、RP 修订期间的设计决定（2026-09-20）

这一节记录 9 月 19 日定稿**之后**、在改写 RP 的 Model / Data / Timeline 三节过程中新增或收紧的决定。

**前面各节没有错。** 出问题的是 RP 正文没跟上前面各节——Data 节写在"时体移出""NLI 校准取消""位置不作采样轴"这三个决定之前，四轮之后才被对照出来。这一节的存在就是为了不让同一件事再发生一次：**这几条决定此前只存在于讨论里，没有落盘。**

### 13.0 一条贯穿性的前提：RP 是瓶颈

截至今天，RP 尚未通过。**smoke run、硬负例冻结、语料冻结、正式训练，全部排在它之后。** RP 晚通过一天，下游全部顺延一天。

两个直接后果：

1. **RP 里任何一句话都不能依赖一个只有 RP 通过之后才能跑出的结果。** 凡是未决的，写成**预注册的决策规则**，不写成数字或状态。
2. 修改清单按"**会不会触发返修**"排序，不按"是否正确"排序。只有审稿人读一遍就会发现、且发现了就要改一轮的，才进 RP。

### 13.1 新增：算力／工期不足时的降级路径

**此前完全不存在。** 第四节只有"13 次，算力允许则 17 次"，没有任何**向下**的路径。

| Tier | 2B | 9B | 27B | 合计 |
|---|---:|---:|---:|---:|
| **首选（preferred）** | 2 条件 × 1 | 3 条件 × 3 | 2 条件 × 1 | **13** |
| **降级（reduced replication）** | 2 条件 × 1 | 3 条件 × 2 | 2 条件 × 1 | **10** |
| **最低矩阵（last resort）** | 2 条件 × 1 | 3 条件 × 1 | 2 条件 × 1 | **7** |

**降级的是 9B 的 seed 数，不是别的。** 注意 4.4 那句"优先把 2B 加到 3 个 seed"是**扩张**优先级（13→17），不能反过来当缩减路径用——13-run 方案里 2B 本来就只有 1 个 seed，没有东西可砍。

**为什么降 seed 而不是砍 27B**：砍 27B 能释放最多算力，但会破坏 Michael 明确要求的"三尺寸 × 两个主条件"矩阵。降 seed 保留全部条件及其解释逻辑，只削弱对训练随机性的描述。**这是一个可辩护但有代价的取舍，不能包装成无损缩减。**

> ⚠️ **三档不是平滑等价的。** 13→10 只是描述变弱；**10→7 是训练复制归零**，等于把 pilot 的单 seed 缺陷原样复原——而那正是主实验相对 pilot 的方法学改进之一。7-run 必须标为 last resort 并写明它恢复了哪个已知缺陷。

**触发条件是"可用训练窗口"，不区分窗口是被算力压缩还是被审批压缩。** 这让同一套机制同时吸收两种不确定性，也让 Timeline 从"一个可能被推翻的排期"变成"一个已经建模了不确定性的计划"。

**tier 必须在看到任何测试结果之前冻结。**

**连带**：RP 的统计段**不能写死 "three seeds"**，那只在 tier-13 成立。写法见 13.4。

### 13.2 learnability 的证据结构

4.3 说了"C3 的 Metric A 比 C1/C2 高不能作为核心证据"，但没说那**算什么**，也没说绝对值该拿什么作参照。补齐如下。

**（a）controls 的分数是同指标的经验参照，不是下限。**

措辞：*empirical reference points without repair-aware target supervision*。

**不要叫 lower bound / floor / 分母**——这在本项目里是会被自己的数据证伪的：detection recall 不惩罚多余结构，而 pilot 已证明模型倾向过度生成，所以 control **完全可能在 recall 上反超** repair-aware。

**（b）差值是 manipulation check，不是 learnability 证据，且信息量不对称。**

| 观察到的差值 | 信息量 |
|---|---|
| 大 | **低**。预期之中，只确认监督生效 |
| 小或反向 | **高**。要么监督没起作用，要么 controls 在靠输入复制／过度生成拿分 |

**实用后果**：Results 里不要把这个差值放在标题位置；但如果它意外地小，必须显著报出并单独分析。

**（c）绝对值的解读用同指标参照，不跨指标。**

- 结构合法性、整图 F1 → 与**同一模型**在流利 PMB 上的同指标表现比；
- repair detection / content / attachment → 报 bounded absolute rates + 置信区间 + 错误分布，配 marked/unmarked 作任务内难度梯度；
- **不要**拿流利整图 F1 去给 repair-detection recall 定标——两者不是同一个 estimand，0.9 和 0.9 之间没有可通约关系。

### 13.3 正文与附录的分流（RP 级 vs thesis 级）

第十节的限制清单是按**最终论文**写的。RP 只需论证设计成立且可行，因此切成两层。

**RP 正文必须写（不写会触发返修）**：

- normalisation-target 为何只在 9B；**cost 三路分解只在 9B 可得**，2B/27B 只有总体 intervention effect；
- accepted-meaning（Metric B）的**定义域**：只对句内改口有定义，挑战集按定义不计算。闸门一划的界线恰好是"改口止于本句"vs"改口之上还建了语篇"——**一个从句同时交代指标定义域和挑战集边界**；
- anaphora 证据规模（核心主张只压在这一类上）；
- fluent-only 是**重训**，不是复用 pilot checkpoint；
- 9,447 缩减划分 + 评分器副本 ⟹ pilot 数字仅作诊断参照。**一句封两条**：

  > All main-study conditions, including the fluent-only baselines, are trained on the same reduced partition and scored with a single current toolkit. The completed pilot used the standard PMB partition and a frozen scorer snapshot, so its figures provide diagnostic context rather than directly comparable baselines.

  **位置放在 Pilot → Model 的过渡处**，不要塞进训练超参数段。

**留给 thesis，RP 不付版面**：exact 检验的完整说明、content/position 双读的诊断解释、Metric A 与 B 非独立的跨话轮双重处罚例子、精确率退化规则的展开、标记词的排除理由史、每源最多 2 条的说明。

**判据**：会让某个**比较不可解释**或某个**主张无依据**的 → 进 RP；只是精度的 → 留给 thesis。

### 13.4 RP 统计段必须交代的五件事

第九节的内容此前从没被提取成"正文要写什么"。压成一段，必须包含：

1. **seed 聚合与 tier**（不写死 three seeds）；
2. 分析单位是**源句**，同源变体不作独立观测；
3. 连续量 Wilcoxon／二值量 McNemar；
4. **Holm 按三族分别校正**（收益／代价／泛化），**尺寸不入任何族**（单 seed、只描述、不检验 ⟹ 不需要校正）；
5. anaphora 用 **exact** McNemar 与 **exact** binomial CI（n≈40，渐近版不可靠）；
6. 推断对象是 **item variation**。

**最后一条是全局限定，不能塞进单一 tier 分支。** 塞进去会生成一个错误的反向暗示——"多 run 的 tier 就估计了训练变异"。跨 seed 平均后做条目检验，**任何 tier 都没有**完整传播训练随机性。参考写法：

> Across all replication tiers, paired tests operate on source-item scores for each condition, averaged across runs when replication is available, and therefore quantify item-level variation. Run-level condition means and standard deviations describe training variability but do not propagate it into the paired tests. Under the last-resort single-run tier, no run-level description of training variability is available.

### 13.5 Data 节必须同步修改（跨节依赖）

**Model 的四个评测集有三个、问题四的两条 holdout 论证，此前在 RP 正文里没有任何前置。** Data 不改，Model 就是在引用不存在的设计。

**三处写错了的（与本文档当前决定冲突，优先级最高）**：

| 现文 | 依据 | 改成 |
|---|---|---|
| "thresholds **will be calibrated**" | 5.6 | NLI 降为保守启发式；**plausibility 单独写**，只说规则在语料冻结前确定 |
| "remaining work will add … **tense, aspect**" | 5.3 | 时体需多概念转换，列为后续扩展 |
| quotas over "… and **reparandum position**" | 10.5 | 只按替换类型；并说明位置不作采样轴的理由 |

**必须补的**：marked/unmarked 两版共享同一 target（测试配对、训练不配对，带标记约占三成）；marker 封闭集合 + 留出一个；同一 WordNet 词对不跨划分；hard negatives 按文档留出 → **9,447**，所有条件共用；challenge 四类（含两类控制），第三类给规模；流利数据作为负例保留的理由；配额是**上限**不是产量承诺；同一流程也应用于 dev/test；funnel 表后说明 9,552 与 9,447 的关系。

**必须收紧**：删掉语料构造段里 "allowing errors to propagate between stages" + "By contrast"——那是不会被检验的比较性优势断言，且原文 "Evaluating **it**" 的 *it* 指的是规则转换，并没有声明"不与 detect-then-parse 基线比较"。改为明写**本研究不比较 inference architectures**。

**必须删除**：旧的 `\subsection{Data}` 整节（与 `NEW Data` 并存）。

### 13.6 Timeline 必须重写

它是 RP 里唯一让读者**验算可行性**的地方，不同步改就等于没算过账。

冲突项：Gemma-2 → Qwen3.5；约 4 次 → 暂定 13 次；"re-evaluate fluent-only" → 全部重训；无 normalisation-target；无 smoke run；无硬负例冻结；仍用旧的 "repair/core recall" 命名；缺 false-alarm 聚合与 anaphora 打分器（都是**要新写的代码**）。

**日期基准**：今天是 2026-09-20，原表 8/24–9/6 与 9/7–9/20 两格**已成过去**，不能继续全用未来时。

**smoke run 写成 feasibility gate**，不是普通条目：

> The smoke runs form a feasibility gate for the preferred 13-run design. If the measured runtime exceeds the available training window, replication at 9B will be reduced before any model result is inspected, while retaining the fluent-only and repair-aware conditions at all three sizes and the 9B normalisation-target control.

### 13.7 RP 通过后的执行顺序

1. **冻结并存档硬负例文档清单** → 生成 9,447 缩减划分；
2. 从三个条件的训练数据**与 RSR 源句池**中一致排除；
3. 2B / 9B / 27B 三个 smoke run（含终止符回归测试）；
4. 据实测资源在 13 / 10 / 7 中**冻结 tier**；
5. 完成语料与评分代码；
6. 开始正式微调。

第 1 步不依赖任何新信息，但阻塞后面全部——它是唯一可以立刻解除的前置依赖。

> **7.1 要求清单在"任何 smoke run 或检查点选择之前"冻结。** 真实理由偏向**存档纪律**而非模型污染（smoke run 的 adapter 本来就要丢弃）：要防的是名单在过程中反复调整，最后说不清训练集究竟排除了什么。

### 13.8 待决项（不要丢）

| 项 | 状态 | 怎么解决 |
|---|---|---|
| **plausibility 阈值能否校准** | 未知。5.6 只处置了 NLI | 那份 150 条标注**分别记录了**语义对比与语境合理性两维；149 否／1 是是**前者**的结果。把同一份标注按**第二维**重新汇总一次即可判定。不需要新标注 |
| **挑战集第三类的最终规模** | 现有 40 条，建议 80–100 | RP 写"现有约 40 条 + 冻结前扩充"——陈述已有状态加意图，交 40 或交 90 文本都成立。**不要写 80–100 然后交 40** |
| **最终 replication tier** | 待 smoke run | 见 13.1 |

---

## 附：容易记错的几件事

写论文或改代码时，这几条最容易出错，单独列出来：

1. **不要写 "without introducing a new separator"** —— `CORRECTION` 确实是本项目加进 SBN 工具的。
2. **不要写 "rare in pre-training"** —— 我们数的是 PMB 的任务监督数据，不是 Qwen 的预训练语料，后者无从推断。
3. **不要用 "orthogonal"** —— 两个指标会互相影响，用 "complementary but interacting"。
4. **不要把 oracle-cleaned 叫 "upper bound"** —— 叫 reference condition。
5. **不要把 Metric B 说成 "core recall"** —— 它是图层面的剥离后比对，不是 triple 归属划分。
6. **不要写 NLI "was calibrated"** —— 150 条里只有 1 个正例，根本没校准成。
7. **不要说挑战集"只跑 Metric A"** —— 它也跑结构合法性和整图 SMATCH++，只有 Metric B 不跑。
8. **不要把 "C3 的 Metric A 高于 C1" 当成核心发现** —— C1 从没见过 `CORRECTION`，这是同义反复。
9. **不要用 Metric B 论证"保留 reparandum 有用"** —— Metric B 先把改口结构剥掉再比，它测的是**代价**。收益看指代边命中率。
10. **不要让 oracle-cleaned 顶替 pipeline 对比** —— 它用了推理时拿不到的信息。要么做预测式 pipeline，要么按 10.9 写明没做。
11. **不要把 Wilcoxon 配 Cohen's $d_z$** —— 秩检验配 rank-biserial correlation。
12. **不要说"已完整估计训练随机性"** —— 跨 seed 平均后做条目检验，估计的主要是测试样本变异；只能写 *averaged over three trained models*。
13. **不要写"证明了模型学的是结构而非模板"** —— 只排除了三种具体捷径（标记词、词对、并置），措辞见 10.8。
14. **不要把 Qwen3.5 称作"最新/最强"** —— Qwen3.8 更新。选它的理由是**只有它有完整的稠密尺寸梯子**。
15. **不要在看到测试结果之后再定检查点选择规则** —— 标准是失败惩罚后的整图 dev F1，各条件对着自己的目标分布算，已预先定死（见 6.3）。
16. **不要从 train/dev 里取硬负例** —— 那是测记忆不是测泛化。必须训练前留出，并同时排除出 RSR 源句池（见 7.1）。
17. **不要在纯负例集上算精确率／F1** —— 会退化。纯负例集报**误报率**，精确率要在正负**混合**集上算。
18. **不要写"C2 在结构上不可能生成被撤回的材料"** —— 它可以从输入复制。写 *receives no target supervision to preserve the withdrawn referent*。
19. **不要把尺寸放进 Holm 校正族** —— 单 seed、只作描述，不做检验就不需要校正。
20. **不要说端到端优于"pipeline 方法"** —— 只优于**一个具体实例化的零样本 detect-then-parse pipeline**。而且检测器要跑在 Qwen3.5 上，否则混入家族差异。
21. **不要写"C1 和 C2 的 Metric A 必然为 0"** —— 写 *expected to obtain little repair-structure credit; their scores function as controls*。它们的分是测出来的，不是定义出来的。
22. **不要同时报误报率和 specificity** —— 互为补数，选**误报率**（与流利 PMB 口径一致）。
23. **不要说训练用的是标准 PMB 划分** —— 留出 105 篇之后是 **9,447** 篇的 custom partition，绝对值与他人数字不再直接可比（见 10.10）。
24. **不要把硬负例留出说成"几乎免费"** —— 1.1% 的数据代价很小，但它改动了标准划分，这是要在论文里交代的。
25. **不要把 controls 的分数叫 lower bound / floor / 分母** —— recall 不惩罚多余结构 + 模型倾向过度生成 ⟹ control 可能反超。叫 *empirical reference points without repair-aware target supervision*（见 13.2）。
26. **不要把核心层 2,200 条说成产量承诺** —— 5.3 写的是"**配额是上限**"。写 *targets up to …; quotas are caps rather than minimum yields*。
27. **不要把 NLI 的校准失败推广到 plausibility 过滤器** —— 5.6 只处置了 NLI。plausibility 的状态未定，既不能写"已校准"也不能写"已取消"（见 13.8）。
28. **不要把"未完整传播训练随机性"这条限定写进单一 tier 分支** —— 它对所有 tier 成立。写进分支会生成"多 run 就估计了训练变异"的反向暗示（见 13.4）。
29. **不要把 normalisation-target parser 叫 pipeline baseline** —— 它是 RSR 输入直接生成 SBN 的端到端解析器，只是目标是流利 SBN。
30. **不要把 4.4 的"优先给 2B 加 seed"当成缩减路径** —— 那是扩张优先级（13→17）。降级降的是 **9B 的 seed 数**（见 13.1）。
