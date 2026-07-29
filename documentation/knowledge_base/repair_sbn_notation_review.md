# `repair_sbn_notation.qmd` 逐例审查报告

审查对象：`documentation/knowledge_base/repair_sbn_notation.qmd`（Covered Cases 全部 13 个 SBN 例子）
审查日期：2026-07-29

> **状态：本报告的结论已全部落地到 `repair_sbn_notation.qmd`。** 本文件保留为审查记录 —— 它记的是「为什么这么改」以及每条判断背后的 gold 证据，`.qmd` 只承载结论。两处待拍板项中，跨轮次 `CORRECTION` 已按 `<1` 定案（§8.1），义项复核（§8.2）仍未做。

---

## 0. 怎么读这份报告

| 章节 | 内容 |
|---|---|
| §1 | 方法与判据 |
| §2 | 设计裁决记录 —— 需要补进 `.qmd` 的 Mechanism 一节 |
| §3 | 「reparandum 论元不写满」的语料与影响验证（回答你提的两个问题） |
| §4 | **逐例结论**：A 硬性错误 / B 静默错误 / C 惯例偏离 |
| §5 | 需要**新增**到 `.qmd` 的解释段落（Fundamental Rules 原文不动） |
| §6 | 已完成的代码改动 |
| §7 | **12 段修正版 SBN**，可直接粘回 `.qmd` |
| §8 | 仍需你拍板的两处 |
| §9 | 复现命令 |

结论一句话：**13 个例子里 5 个 parser 直接报错、4 个能解析但索引落到错节点（静默错误，最危险）、其余多处偏离 PMB gold 惯例。** 修正版已全部实测通过。

---

## 1. 方法与判据

两条独立证据链：

1. **实际解析**：把每个例子喂进仓库里真正的 parser（`data/pmb-5.1.0/src/sbn/sbn_smatch.py` + `sbn_spec.py`），dump 出全部节点、边、盒归属与 Penman 串。**不靠肉眼数索引。**
2. **语料核对**：每个构式对照 PMB 5.1.0 `data/pmb-5.1.0/split/en/train/gold.sbn`（9552 条记录）确认惯例。

四条判据：

- (a) 不抛 `SBNError`；
- (b) 每条 role/operator 边的目标节点 = `%` 注释里写的那个词；
- (c) `BOX_BOX_CONNECT` 的盒编号符合预期嵌套；
- (d) 没有意外的 `CONSTANT` 节点 —— **越界索引会被 parser 静默降级成常量而不报错**，这正是 §4.B 那类错误能藏住的原因。

补一条给 §7 用的：(e) Penman 串里所有倒置角色都已归一成 `-of` 形式。

---

## 2. 设计裁决记录（补进 `.qmd` 的 Mechanism 一节）

**`CORRECTION` 是一个 token、承担两个任务，靠结构消歧，绝不在序列里写 "sentence/discourse" 之类的标签**（那违背 SBN 的设计哲学）：

| | intra-turn self-repair | cross-turn repair / correction |
|---|---|---|
| 结构特征 | **有** sibling `CONJUNCTION` | **无** sibling `CONJUNCTION` |
| `CORRECTION` 的作用 | quarantine reparandum | 遵循 L&A：否定 π1、commit π2 |
| `CONJUNCTION` 的作用 | 把 repair 及其后续内容 merge 回 `CORRECTION` 上一层的盒子 | 不适用 |

**消歧规则**：*repair-CONJUNCTION ⇔ 与一个 `CORRECTION` 共享父盒*。任何触碰 `CONJUNCTION` 语义的工具都按这条判。

**方向自洽性（实测确认）**：parser 生成的盒间边是 `target_box → new_box`（`sbn_smatch.py:220-227`）。跨轮次场景下 target = π1（被否定）、new = π2（被断言），与 L&A `Correction(π1,π2) ≡ ¬K_π1 ∧ K_π2` 方向一致，无需任何额外约定。

---

## 3. 「reparandum 论元不写满」的两项验证

### 3.1 PMB gold 有无先例 —— **有，且直接对应，不必扩到 silver**

| 记录 | 句子 | 证据 |
|---|---|---|
| `p62/d1986` | "I also did not call." | `call.v.03 Agent Time` —— 及物动词，受话人没说出来就**完全不给 role 边** |
| `p79/d0205` | "If you sing, we'll sing with you." | 前件盒 `sing.v.02 Agent -1 Time +1`，后件盒 `sing.v.02 Agent -2 Time -1 Co-Agent +1`。**同一谓词、从属盒中 role 集更瘦**，与 retracting repair 形状完全一致 |
| `p09/d3007` | "I returned the books I borrowed from the library, and I borrowed some new ones." | 第二个 `borrow.v.01` 省掉 `Source`，尽管 library 已在上文出现 |
| `p54/d0066` / `p02/d2070` | "He promised to return and yet he didn't." | `return.v.01 Theme -3`（无 Time）vs `Theme -2 Time -1` |

全语料统计：

- 同一 record 内同一动词 synset 以**不同 role 集**重复出现：14 例
- role 数 ≤ 1 的动词 token：177 个（0 个 role 12 个，1 个 role 165 个）
- `put.v.01` 共 41 例，其中 3 例 role 集不完整（`Agent Destination Theme` ×2、`Destination Theme Time` ×1）

### 3.2 对 LM 训练与后续解析的影响 —— **方案 2 优于 movement**

以 Retracting Repair 实测两种写法：

| | 方案 2（瘦 reparandum） | 方案 1（movement 前移） |
|---|---|---|
| 图中边数 | 24 | 26 |
| max \|index\| | 3 | 5 |
| Penman 串长 | 545 | 579 |

**支持方案 2：**

- **语言学上它才是对的。**「you will put, I mean, you will drop the ball on the table」里，ball 和 table 从未在 `put` 的计划下被说出来。给 `put` 挂 `Theme`/`Destination` 是标注者的重构，不是说话人说的内容。
- **保住表层语序。** movement 把 ball/table 提到动词之前，破坏 SBN 赖以对齐的语序惯例 —— 而那正是 seq2seq 模型利用的信号。
- **避开索引膨胀。** movement 单调推高 |index|，一旦越过 9 就撞上已知的 `INDEX_PATTERN` 单位数 bug（`sbn_spec.py:217` 的 `r"((-|\+|\<|\>)\d)"` 没有 `+` 量词，`-11` 被静默截断成 `-1`）。本例最大到 5 尚未触发，长句会。
- **残留正指标不跨边界。** 修正后 `drop.v.01 Theme +1 Destination +2` 的目标全在自己所在的 CONJUNCTION 盒内，与 gold 一致（`p01/d3366` "The ship left every Monday."：`leave.v.01 … Time +2 … CONJUNCTION <2 time.n.08 DayOfWeek monday`）。

**代价（须在评估时对冲）：** reparandum 盒 role 更少 → 该盒贡献的 Smatch 三元组更少 → 模型完全漏掉 CORRECTION 盒时受到的惩罚变轻。**建议评估时单独报 CORRECTION 盒的三元组召回，不要只看整体 F1。**

---

## 4. 逐例结论

### 4.A 硬性错误 —— parser 直接报错（5 个例子）

| 位置 | 错误 | parser 实际报错 | 修正 |
|---|---|---|---|
| 动词自我修复 v1/v2 | `EUQ` | `SBNError: Invalid token found 'EUQ'` | `EQU` |
| 动词自我修复 v1/v2 | `chase` / `hunt` 缺 synset id | 后续 token 解析错位 | `chase.v.01` / `hunt.v.01` |
| Negation | `ThemeOf-1` 缺空格 | `SBNError: Invalid token found 'ThemeOf-1'` | `ThemeOf -1` |
| Adjunct | `Day "Monday"` | `SBNError: Invalid token found 'Day'` | `DayOfWeek monday`。gold 16 次，**小写裸常量、不加引号** |
| 篇章回指 | `Co-ThemeOf` | `SBNError: Invalid token found 'Co-ThemeOf'` | 已改脚本，见 §6 |
| **全文** | 独立成行的 `% Utterance 1`、`% Discourse boundary` | `SBNError: Invalid token found '%'` | 必须写成 `%%% …`（`SBNSpec.COMMENT_LINE`） |
| **全文** | 行尾**空**注释标记，如 `order.v.01 Agent -2 Time -1          %` | `SBNError: Invalid token found '%'` | 删掉。原文档共 41 处 |

> 关于 `%` 的完整规则：`SBNSpec.COMMENT` 是三字符串 `" % "`，`%` 只有在**前后都有空白**时才被认作注释标记。行首的 `%` 后面有空格但前面没有；行尾空标记前面有空格但后面没有 —— 两者都会被当成 SBN token 而中止解析。只有 `内容 % 注释文字` 这一种形态是安全的。
>
> 这一条是把整份 `.qmd` 的 sbn 块逐块喂 parser 才暴露的：单看例子内部逻辑不会发现，因为它与索引、盒结构都无关。

> `DayOfWeek` 这条你在 `verify_conjunction.py` 里已经修对过一次（`repair-3`），`.qmd` 里回退了。

### 4.B 静默错误 —— 能解析，但索引落到错节点（最危险）

**B1. SV 句：`Agent` 边整条消失**

```
sneeze.v.01 Agent -1 Time -1
```
`Agent -1` 和 `Time -1` 都指向 `time.n.08`。图是 DiGraph，后写覆盖先写，实测边表只剩：

```
sneeze.v.01  --Time-->  time.n.08
```

`Agent` 边不见了，**而且不报任何错**。应为 `Agent -2 Time -1`。

**B2. 动词自我修复 v1：role 挂到了错误的概念上**

```
CORRECTION <1 Agent -3 Theme -2
```
分隔符只吃一个 connector token，剩下的 role 落到**前一个概念 `time.n.08`** 上；`Agent -3` 越界后被降级成一个叫 `"-3"` 的常量节点。实测：

```
time.n.08  --Agent-->  -3            ← 常量节点，不是概念
time.n.08  --Theme-->  cat.n.01
```

角色不能挂在分隔符上（Manual §3.5.6：分隔符后面只跟一个 connector）。**建议整段删掉 v1，只保留 v2。** v2 的索引全部正确，且原文那句「若 SBN 允许不写满语义角色」的悬念，§3.1 已给出肯定答案。

**B3. Donkey 句：条件句结构塌掉**

```
NEGATION <2
```
实测生成 `BOX0 ==NEGATION==> BOX2` —— 两个**并列**否定盒挂在 BOX0 下，而不是嵌套。应为 `NEGATION <1`，得到 `BOX1 ==NEGATION==> BOX2`。

这条你也在 `verify_conjunction.py` 里标注过 `[CORRECTED: NEGATION <1, not <2]`，`.qmd` 回退了。

**B4. Forwarding Repair：四处错**

| 原文 | 实际解析成 | 应为 |
|---|---|---|
| `drive.v.03 … Theme +3`（Josh 的） | 指向**第二个 `drive.v.03`**（自指） | 该论元在 reparandum 里根本没说出来 → 删掉 |
| `drive.v.03 … Theme +2`（Marsha 的） | 指向 `old.a.02`（形容词） | `Theme +2` 指向 `car.n.01` |
| `old.a.02 Value +1` + `car.n.01 Attribute -1` | 两条方向相反、互相矛盾的边 | 单边 `old.a.02 AttributeOf +1` |
| `car.n.01 … Destination +1` | `Destination` 挂在**车**上 | 挂在 drive 事件上（gold：`go.v.01 … Destination +2 city.n.01`） |

**B5. 篇章回指：`dessert.n.01` 零边悬空**

末尾的 `dessert.n.01` 没有任何 role/operator 边，"desserts" 对表示毫无贡献。gold 的对应构式是系词 + `Co-Theme`：

```
entity.n.01 be.v.01 Theme -1 Time +1 Co-Theme +2 time.n.08 EQU now fish.n.01     % Those are not fish.
… be.v.08 Theme -1 Time +1 Co-Theme +2 … person.n.01 Role +1 doctor.n.01        % My parents are both doctors.
```

（双 `ANA` 本身没问题 —— Bos 已批准，不动。）

### 4.C 惯例偏离 —— 能解析但偏离 gold

| 原文 | 改成 | 依据 |
|---|---|---|
| `time.n.01 TPR now` ×3 | `time.n.08 TPR now` | gold 时态指称一律 `time.n.08`（9310 次）；`time.n.01`（45 次）是可数「次数」，配 `Quantity` |
| `male.n.01 EQU -6` | `male.n.02 EQU …` | gold `male.n.02` 4414 次 vs `male.n.01` 1 次 |
| `person.n.01 Name "Joe"` | `male.n.02 Name "Joe"` | 下文 "he" 已定性别；gold 2188 次 `male.n.02 Name` |
| `gift.n.01 PartOf "birthday"` | `gift.n.01 Of +1` + `birthday.n.01` | `PartOf` 是概念—概念关系，**不能取字符串** |
| `buy.v.01 … Manner +3`（as a gift） | `Attribute +3` | gold："She hired him as a programmer" = `hire.v.01 … Attribute +3 … programmer.n.01`；`Manner` 在 gold 配方式副词 / 交通方式 |
| `favourite.a.02` | `favorite.a.02` | gold 15 : 7 |
| "well" 未表示（你挂着的那条） | `Manner +3` + `well.r.01` | gold 原样例 `p94/d2703` "I play the piano well."：`play.v.07 Agent -1 Time +1 Theme +2 Manner +3 time.n.08 EQU now piano.n.01 well.r.10` |
| `car.n.01 Name "Ferrari"` | *（软提示，未改）* | gold 把 "Ferrari" 一律给 `company.n.01` |
| 介词例的 "went" vs `run.v.01` | 句子改 "I ran to, I mean, from the school" | 按你的裁决，以 SBN 为准 |

---

## 5. 需要新增到 `.qmd` 的解释（Fundamental Rules 原文一字不改）

Bos 论文原文保持原样，在其**后面**加一个 Clarifications 小节，承载下面四条。

### 5.1 线性索引计数规则（目前 KB 最大的缺口）

**`Role -n` / `Role +n` 数的是全序列中**所有**前置概念，盒子从不重置、不限制、不划分计数域。** 已在 PMB 5.1.0 gold 上验证：92 个「线性 vs register」两模型判读不同的案例中，线性 92/92 正确，零反例；全语料线性只有 1/19873 无法消解（且那条本身是 gold 标注错误 `p82/d3242`）。参考实现同样如此（`sbn_smatch.py:253-256`，`target_idx = _active_synset_id + idx`，计数器从不被建盒动作触碰）。

**推论：合成数据可以放心穿过 CORRECTION 盒线性计数，`CONJUNCTION` 之后不需要任何 off-by-one 修正。**

顺带澄清一个易混点：**尖括号索引盒子，正负号索引概念。** `Proposition >1` / `CONTINUATION <0` 是盒指针（`sbn_smatch.py:300-302`），不是概念索引 —— 这才是「`Proposition >1` 的目标明明在两格以外索引却是 +1」的真正解释。

### 5.2 可及性的操作化补注（对原文的解释，非修改）

- 原文说的「negation 之外」，在操作上是**祖先盒**。**sibling 盒的指称同样不可及**，即使它在线性顺序上更靠前。
- **正指标可以穿进 `CONJUNCTION` 盒**（它是 merge 不是从属，gold 大量使用），**但不得穿进或穿出 `CORRECTION` 盒**。
- 两条修复正指标越界的手段，按优先级：(1) reparandum 只写实际说出来的部分（§3）；(2) role 倒置（`ThemeOf` / `TimeOf` / `Co-ThemeOf`）；(3) movement 前移 —— 只在前两者都不适用时用。

### 5.3 `CORRECTION` 盒对跨轮次回指是透明的 —— 应写成正面性质

篇章回指例（Bos 已批准）里，`entity.n.01 ANA -2` 指向的 reparandum 正被隔离在 `CORRECTION` 盒中。这与「把 `CORRECTION` 类比成 `NEGATION`」的直觉相冲突 —— 但**语言事实站在它这边**：「My favourite dessert is banana bread, actually, cherry pie. — They are both good desserts.」完全合法，说明被 quarantine 的 reparandum 仍然留在话语指称域里。

**这正是 `CORRECTION` 区别于 `NEGATION` 的关键性质，值得在文档里明写成一条正面规则**，而不是留给读者自己撞上去当成矛盾。

（数据提示，非错误：gold 中「单个概念带两个 `ANA`」出现 0 次。合成数据里用这个装置会引入训练分布外的模式，值得在评估时单独留意。）

### 5.4 `CONJUNCTION` 盒的「尾巴」效应

`CONJUNCTION` 之后直到下一个分隔符的**全部**内容都留在该盒里。例如主语自我修复例中，`play.v.01`、`time.n.08`、`tennis.n.01`、`well.r.01` 全部落在 BOX2 而非 BOX0。因为 merge，语义上等价于母盒 —— **但图结构不同、Smatch 分数不同**，所以必须作为固定惯例写死，不能让不同标注者各写各的。

### 5.5 `ThemeOf` / `TimeOf` 是本项目扩展，不是 PMB 惯例

全语料 gold 中实际使用的倒置角色只有：

| 角色 | 次数 |
|---|---|
| `AttributeOf` | 1802 |
| `PartOf` | 572 |
| `SubOf` | 120 |
| `ColourOf` | 80 |
| `MadeOf` | 38 |
| `InstanceOf` | 37 |
| `ContentOf` | 4 |
| `CauserOf` | 2 |
| `FeatureOf` | 1 |
| `AgentOf` | 1 |
| **`ThemeOf`** | **0** |
| **`TimeOf`** | **0** |
| **`MannerOf`** | **0** |

靠它们规避正指标是合理的设计，但**文档必须声明这是项目扩展**。同时说明：`to_penman_string`（`sbn_smatch.py:691-698`）**只对 `INVERTIBLE_ROLES` 集合里的成员**做 `Of → -of` 归一化 —— 不在该集合里的倒置角色，Smatch 会当成一条完全不同的边。

---

## 6. 已完成的代码改动

按你的裁决（`Co-Theme` 属 VerbNet-LIRICS 体系、合法，倒置形式应加进解析脚本而非改例子）：

| 文件 | 改动 |
|---|---|
| `data/pmb-5.1.0/src/sbn/sbn_spec.py` | `INVERTIBLE_ROLES` 加 `"Co-ThemeOf"`；`ROLES` 加 `"Co-ThemeOf"` |
| `colloquium_prep/pilot_eval/sbn_lib/sbn_spec.py` | 同上（两份拷贝原本 byte-identical，改后仍一致） |

**两个集合都必须加**：只加进 `ROLES` 会解析成功，但 Penman 输出 `:Co-ThemeOf`，Smatch 不做归一，与 gold 的 `:Co-Theme` 判为不同边。加进 `INVERTIBLE_ROLES` 后实测输出 `:Co-Theme-of`，无残留。

**回归检查**：`python3 verify_conjunction.py` 的 4 个 repair + 3 个 PMB gold universal 例子全部照常通过。`Co-ThemeOf` 在 gold 中出现 0 次，因此不影响任何 baseline 分数。

---

## 7. 修正版 SBN

以下 12 段全部实测通过判据 (a)–(e)：解析成功、每条边落在预期节点、盒嵌套正确、无越界降级常量、无零边悬空概念、Penman 中倒置角色全部归一。

### 7.1 SV — "The boy sneezed, I mean, coughed"

```sbn
boy.n.01                       % The boy
time.n.08 TPR now
    CORRECTION <1
sneeze.v.01 Agent -2 Time -1   % sneezed, I mean,
    CONJUNCTION <2
cough.v.01 Agent -3 Time -2    % coughed
```

### 7.2 SVO 主语自我修复 — "Josh, no, Mary plays tennis well"

```sbn
entity.n.01
    CORRECTION <1
male.n.02 Name "Josh" EQU -1                  % Josh, no,
    CONJUNCTION <2
female.n.02 Name "Mary" EQU -2                % Mary
play.v.01 Agent -3 Time +1 Theme +2 Manner +3 % plays
time.n.08 EQU now
tennis.n.01                                   % tennis
well.r.01                                     % well
```

### 7.3 动词自我修复 — "The cat chases, actually, hunts the rat"

> 原 v1（`CORRECTION <1 Agent -3 Theme -2`）删除，只保留此式。

```sbn
cat.n.01                            % The cat
time.n.08 EQU now
    CORRECTION <1
chase.v.01 Agent -2 Time -1         % chases, actually,
    CONJUNCTION <2
hunt.v.01 Agent -3 Theme +1 Time -2 % hunts
rat.n.01                            % the rat
```

> `chase.v.01` 不带 `Theme` 是**正确**的：rat 未在 chases 的计划下说出。gold 先例 `p62/d1986 call.v.03 Agent Time`。

### 7.4 宾语自我修复 — "I ordered a banana bread, I mean, a cherry pie"

```sbn
person.n.01 EQU speaker        % I
time.n.08 TPR now
order.v.01 Agent -2 Time -1    % ordered
    CORRECTION <1
banana_bread.n.01 ThemeOf -1   % a banana bread, I mean,
    CONJUNCTION <2
cherry_pie.n.01 ThemeOf -2     % a cherry pie
```

### 7.5 否定 — "I didn't order a banana bread, I mean, a cherry pie"

```sbn
person.n.01 EQU speaker        % I
time.n.08 TPR now
    NEGATION <1                % didn't
order.v.01 Agent -2 Time -1    % order
    CORRECTION <1
banana_bread.n.01 ThemeOf -1   % a banana bread, I mean,
    CONJUNCTION <2
cherry_pie.n.01 ThemeOf -2     % a cherry pie
```

> 盒结构实测：`BOX0 →NEGATION→ BOX1`、`BOX1 →CORRECTION→ BOX2`、`BOX1 →CONJUNCTION→ BOX3`。repair 正确 merge 回**否定盒**而非母盒。

### 7.6 Adjunct — "The class is on Monday, no, on Tuesday"

```sbn
class.n.01                                % The class
time.n.08 EQU now
be.v.01 Theme -2 Time -1                  % is on
    CORRECTION <1
time.n.08 DayOfWeek monday TimeOf -1      % Monday, no,
    CONJUNCTION <2
time.n.08 DayOfWeek tuesday TimeOf -2     % on Tuesday
```

### 7.7 介词替换 — "I ran to, I mean, from the school"

> Bos 的 dummy concept + equality 方案，`event.v.01`（非法 synset）换成 gold 中 1315 次的通用占位符 `entity.n.01`（`event.n.01` 在 gold 中 0 次）。

```sbn
person.n.01 EQU speaker              % I
time.n.08 TPR now
run.v.01 Theme -2 Time -1            % ran
school.n.02                          % the school
    CORRECTION <1
entity.n.01 EQU -2 Destination -1    % to, I mean,
    CONJUNCTION <2
entity.n.01 EQU -3 Source -2         % from
```

> 两个 dummy 都 `EQU` 同一跑动事件：CORRECTION 盒里它带 `Destination`（被否定），merge 盒里带 `Source`（被断言）。全负指标、全部指向 superordinate BOX0，零可及性违规。

### 7.8 Donkey — "If a farmer owns a donkey, he beats, I mean, feeds it."

```sbn
    NEGATION <1                   % If
farmer.n.01                       % a farmer
own.v.01 Pivot -1 Theme +1        % owns a
donkey.n.01                       % donkey,
    NEGATION <1                   % [唯一改动：<2 → <1]
entity.n.01 EQU -1                % it
    CORRECTION <1
beat.v.01 Agent -4 Patient -1     % he beats, I mean,
    CONJUNCTION <2
feed.v.01 Agent -5 Patient -2     % feeds it
```

### 7.9 Retracting Repair — "Bill said you will put, I mean, you will drop the ball on the table"

```sbn
male.n.02 Name "Bill"                              % Bill
say.v.01 Proposition >1 Agent -1 Time +1           % said
time.n.08 TPR now
    CONTINUATION <0
person.n.01 EQU hearer                             % you
time.n.08 TSU now                                  % will
    CORRECTION <1
put.v.01 Agent -2 Time -1                          % put, I mean,
    CONJUNCTION <2
drop.v.01 Agent -3 Time -2 Theme +1 Destination +2 % you will drop
ball.n.01                                          % the ball
table.n.02                                         % on the table
```

> **Positive Index Alert 解除。** `put` 不再有 `Theme +2 / Destination +3` —— 那两个论元在 reparandum 里根本没说出来。`drop` 的正指标全部落在自己所在的 CONJUNCTION 盒内，不跨任何边界。

### 7.10 Forwarding Repair — "Josh drove, no, Marsha drove the old car to the church"

```sbn
entity.n.01
    CORRECTION <1
male.n.02 Name "Josh" EQU -1                         % Josh
time.n.08 TPR now
drive.v.03 Agent -2 Time -1                          % drove, no,
    CONJUNCTION <2
female.n.02 Name "Marsha" EQU -4                     % Marsha
time.n.08 TPR now
drive.v.03 Agent -2 Time -1 Theme +2 Destination +3  % drove
old.a.02 AttributeOf +1                              % the old
car.n.01                                             % car
church.n.02                                          % to the church
```

### 7.11 篇章回指 — "A: My favourite dessert is banana bread, actually, cherry pie. B: They are both good desserts"

> 需先完成 §6 的 `Co-ThemeOf` patch（已完成）。

```sbn
person.n.01 EQU speaker                    % A: My
favorite.a.02 Experiencer -1 Stimulus +1   % favourite
dessert.n.01                               % dessert
time.n.08 EQU now
be.v.02 Theme -2 Time -1                   % is
    CORRECTION <1
banana_bread.n.01 Co-ThemeOf -1            % banana bread, actually,   (reparandum)
    CONJUNCTION <2
cherry_pie.n.01 Co-ThemeOf -2              % cherry pie                (repair)
    CONTINUATION <1
entity.n.01 ANA -2 ANA -1                  % B: They   (双 ANA，Bos 已批准)
time.n.08 EQU now                          % are
be.v.02 Theme -2 Time -1 Co-Theme +2
good.a.01 AttributeOf +1                   % both good
dessert.n.01                               % desserts
```

三处改动：

1. `be.v.02 Co-Theme +2` → 两个 `Co-ThemeOf`，让 banana_bread 与 cherry_pie 用同一种写法，且不再有正指标穿进 CORRECTION 盒；
2. 补系词，修掉悬空的 `dessert.n.01`（gold 构式 "Those are not fish"）；
3. `favourite` → `favorite`。

实测：双 `ANA` 两条边都在、无覆盖（目标不同所以不会互相覆盖，与 §4.B1 的情形不同）；Penman 输出 `:Co-Theme-of`。

> 若要表示 "both"，gold 的做法是在 entity 上加 `Quantity 2`（"My parents are both doctors"）—— 可选。

### 7.12 跨轮次 — "A: Joe bought a Ferrari, I mean, a Jaguar, as a birthday gift. B: No, he bought it as a wedding gift."

```sbn
male.n.02 Name "Joe"                     % Joe
time.n.08 TPR now
buy.v.01 Agent -2 Time -1 Attribute +3   % bought
    CORRECTION <1                        % I mean,  [句内：有 sibling CONJUNCTION]
car.n.01 Name "Ferrari" ThemeOf -1       % a Ferrari  (reparandum)
    CONJUNCTION <2
car.n.01 Name "Jaguar" ThemeOf -2        % a Jaguar   (merge 回 BOX0)
gift.n.01 Of +1                          % as a
birthday.n.01                            % birthday gift
    CORRECTION <1                        % No,  [跨轮次：无 sibling CONJUNCTION]
male.n.02 EQU -7                         % he
buy.v.01 Agent -1 Theme -4 Attribute +1  % bought it
gift.n.01 Of +1                          % as a
wedding.n.01                             % wedding gift
```

> 跨轮次 `CORRECTION <1` 实测生成 `BOX2 →CORRECTION→ BOX3`，即 π1 = A 的 repair 盒（含 birthday gift）、π2 = B 的断言 —— 与 L&A 方向一致，且 B 否定的正是 `Attribute` 那部分。见 §8.1。

---

## 8. 仍需你拍板的两处

### 8.1 跨轮次 `CORRECTION` 指向哪个盒

| 选项 | 效果 | 论据 |
|---|---|---|
| `<1`（当前） | π1 = A 的 CONJUNCTION/repair 盒 | B 否定的正是 "as a birthday gift"，而 `gift.n.01` 就在那个盒里；且 gold 的篇章关系惯例是指向紧邻上文语境 |
| `<3` | π1 = BOX0，A 的整个语段 | 若你认为 "No" 否定的是 A 的整句而非某个成分。gold 有非相邻 connector 先例：`p60/d2992` 的 `CONJUNCTION <3` / `NEGATION <3` |

两者在 merge 之后真值条件相同（CONJUNCTION 盒的内容本来就并入 BOX0），但**图结构不同 → Smatch 分数不同**，合成数据必须统一成一条规则。

### 8.2 义项复核

`class.n.01`、`be.v.01`、`drive.v.03`、`church.n.02`、`old.a.02`、`well.r.01` 的义项号我按文档「义项仅作示意」的声明未改。若这些例子要进合成数据，义项需单独过一遍 WordNet —— 尤其 `church.n.02`（= 教堂礼拜仪式，不是建筑物）和 `class.n.01`（= 具有共同属性的一类事物，不是课程/课时）。

---

## 9. 复现命令

### 9.1 检查单个例子

```bash
cd colloquium_prep/pilot_eval/sbn_lib
python3 -c "
import sys; sys.path.insert(0,'.')
from sbn_smatch import SBNGraph
from sbn_spec import SBN_EDGE_TYPE
G = SBNGraph().from_string(open('/tmp/example.sbn').read())
tok = {n:d['token'] for n,d in G.nodes(data=True)}
for u,v,d in G.edges(data=True):
    print(d['type'], tok.get(u,u), d.get('token'), tok.get(v,v))
print(G.to_penman_string())
"
```

判据见 §1。特别注意判据 (d)：**越界索引会被静默降级成常量节点**，所以要显式检查有没有形如 `-3` 的 `CONSTANT` 节点。

### 9.2 patch 回归检查

```bash
cd data/pmb-5.1.0/src/sbn && python3 verify_conjunction.py
```

4 个 repair + 3 个 PMB gold universal 例子应全部无错误通过。

### 9.3 本报告用到的 gold 查询

```bash
cd data/pmb-5.1.0/split/en/train

# 时态指称的 synset
grep -c "time\.n\.08" gold.sbn        # 9310
grep -c "time\.n\.01" gold.sbn        # 45

# DayOfWeek 的常量形式（小写裸词）
grep -o "DayOfWeek [^ ]*" gold.sbn | sort | uniq -c

# 实际使用的倒置角色
grep -oE "\b[A-Z][A-Za-z-]*Of\b" gold.sbn | sort | uniq -c | sort -rn

# 方式副词 "well"
grep -B1 "well\.r\.[0-9]*" gold.sbn

# "as a X" 用的是哪个 role
grep -A1 -E "^[A-Z].* as an? [a-z]+" gold.sbn

# 论元不写满的先例
grep -A2 "^p62/d1986$" gold.sbn       # I also did not call.
grep -A2 "^p79/d0205$" gold.sbn       # If you sing, we'll sing with you.
grep -A2 "^p09/d3007$" gold.sbn       # …I borrowed from the library, and I borrowed some new ones.

# 复数谓语名词构式
grep -A1 "^Those are not fish" gold.sbn
grep -A1 "^My parents are both doctors" gold.sbn

# 非相邻 connector
grep -A2 "^p60/d2992$" gold.sbn       # CONJUNCTION <3 / NEGATION <3
```
