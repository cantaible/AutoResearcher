## 调研范围与方法说明

本次调研聚焦 **2026 年 3 月 1 日至 3 月 31 日** 期间“发布”的大模型，覆盖四类对象：

1. **基础大语言模型**
2. **图像生成模型**
3. **视频生成模型**
4. **Agent / Coding 模型**

同时重点关注两类对象：

- **头部厂商官方发布**：优先采用厂商官网、官方博客、开发者文档、技术报告、官方新闻稿
- **LMArena（Arena）榜单相关模型**：优先采用 Arena 官方 leaderboard、更新日志、官方博客与官方数据集

需要特别说明三点：

- **“发布”口径不完全等同于“论文公开”**。本报告优先纳入“正式发布 / 开放访问 / API 上线 / 产品级预览开放”的模型。
- **部分模型在 3 月仅为 preview / alpha / beta / 榜单上线**，这类会明确标注为“预览”或“测试”。
- **并非所有厂商都会公开参数量、训练数据规模**；遇到官方未披露的情况，会明确写“未披露”。

---

## 一、2026 年 3 月确认发布/开放的大模型总表

下表优先列出**在 2026 年 3 月可被较高置信度确认发布或开放**、且与本次研究范围直接相关的模型。

| 模型 | 日期 | 类别 | 厂商 | 发布形态 | 主要亮点 | 技术规格 | LMArena情况 |
|---|---|---:|---|---|---|---|---|
| GPT-5.4 | 2026-03-05 | 基础 LLM / Agent / Coding | OpenAI | 正式发布 | 推理、编码、Agent 工作流统一；支持深度检索；视觉理解增强 | API 最高 100 万 token 上下文 | 3 月文本榜前列，gpt-5.4-high 位列前 3 之一 [1][2][3] |
| GPT-5.4 mini | 2026-03-17 | 基础 LLM / Agent / Coding | OpenAI | 正式发布 | 低延迟、多模态、工具调用、适合高频任务 | 40 万 token 上下文 | 作为 GPT-5.4 家族成员进入 Arena 更新日志 [1][4][2] |
| GPT-5.4 nano | 2026-03-17 | 轻量模型 / Agent / Coding | OpenAI | 正式发布 | 最低成本，适合分类、抽取、轻量编码代理 | 官方未披露参数量 | 家族成员进入 Arena 更新体系 [1][4] |
| Gemini 3.1 Flash Live | 2026-03-26 | 实时多模态语音模型 | Google | 正式发布 | 实时音频对话、多语言、复杂函数调用、SynthID 水印 | 官方未披露参数；支持 200+ 国家地区使用 | 非传统文本榜主力，但属于 Google 3 月重要模型发布 [5] |
| Veo 3.1 Lite | 2026-03 下旬 | 视频生成 | Google | 付费预览 | 低成本视频生成，720p/1080p，4/6/8 秒 | 官方未披露参数量 | Google Veo 系列在 Arena 视频榜长期高位 [6][7] |
| Phi-4-Reasoning-Vision-15B | 2026-03 | 多模态推理 / Vision | Microsoft | 发布到 Foundry | 视觉+多步推理，适合实时交互与 Agent 场景 | 15B 参数 | 非 LMArena 主文本榜头部，但属 3 月微软重要模型 [8] |
| Leanstral | 2026-03 | Code / Agent | Mistral | 正式发布 / 开源 | 面向 Lean 4 的形式化证明代码代理 | 0.6B 参数 | 非 Arena 主榜头部，但属代码代理重要新模型 [9] |
| Voxtral TTS | 2026-03 | 语音生成 | Mistral | 正式发布 / 开源 | 多语言情感 TTS，低延迟 | 4B 参数 | 不属于 Arena 核心文本榜对象 [10] |
| Mistral OCR | 2026-03 | 文档理解 / OCR | Mistral | 正式发布 | 文档理解 API，结构化抽取强 | 官方称准确率 94.89% | 与 Arena 文档/视觉赛道相关 [11] |
| Transcribe | 2026-03-26 | 语音识别 | Cohere | 正式发布 / 开源 | 企业级低延迟 ASR，14 种语言 | 2B 参数 | 非 LMArena 核心主榜对象 [12] |
| Runway Characters | 2026-03-09 | 视频 Agent / 实时角色 | Runway | 正式发布 | 单图生成实时视频角色，支持声音、知识库、动作 | 基于 GWM-1 | 非 Arena 主文本榜；与视频/Agent 赛道相关 [13] |
| Midjourney V8 Alpha | 2026-03-17 | 图像生成 | Midjourney | Alpha 测试 | 提升速度、细节、文本渲染，支持 2K 渲染 | 未披露参数 | 预览性质，不宜按正式版处理 [14] |
| Qwen3.5-Max-Preview | 2026-03-18 | 基础 LLM | 阿里巴巴 | Preview | 高性能通用模型，在 Arena 首秀 | 官方未完整披露参数；Qwen 3.5 系列采用 MoE 路线 | Arena 约为全球前列，官方称中国第一 [15][16] |
| Qwen-Agent | 2026-03-05 | Agent 框架 / Agent 模型生态 | 阿里巴巴 | 正式发布 | 原生函数调用、代码解释器、RAG、MCP、浏览器自动化 | 支持 4B 级移动端运行 | 非 Arena 单模型榜对象，但与 Agent 能力高度相关 [17] |
| Qwen3.5-Omni | 2026-03-29 | 全模态模型 | 阿里巴巴 | 正式开放 | 文本、图像、音频、视频统一处理 | 256K 上下文；支持长音视频；113 种 ASR、36 种 TTS | 3 月末发布，榜单体现可能滞后 [18] |
| GLM-5-Turbo | 2026-03 中旬 | Agent / Coding | 智谱 Z.ai | 正式发布 | 长链路 Agent 任务优化，强调工具调用稳定性 | 20 万上下文、12.8 万输出 token；参数未披露 | 未见官方 Arena 头部排名快照 [19] |
| LongCat-Flash-Prover | 2026-03-24 | Code / Prover | 美团 | 正式开源 | 数学定理证明，Lean4 环境表现突出 | 560B MoE；MiniF2F-Test 97.1% | 不属于 Arena 主榜常规对象 [20][21] |
| Dreamina Seedance 2.0 | 2026-03（规模化上线） | 视频生成 | 字节跳动 / CapCut | 正式开放付费体验 | 文本/图像/参考视频输入，最长 15 秒，多比例，含水印与 C2PA | 官方未披露参数 | Arena Image-to-Video 榜第一 [22][23] |
| MiniMax M2.7 | 2026-03 | Agent / Coding | MiniMax | 正式发布 | 强调自我进化、复杂任务执行、多人代理协同 | 官方未完整披露参数 | 未见官方 Arena 精确名次快照 [24] |
| Cursor Composer 2 | 2026-03-19 | Coding / Agent | Cursor | 正式发布 | 长上下文、多步骤编码代理、复杂工程任务 | 20 万上下文；基座相关信息来自官方材料与社区说明 | 不属于 Arena 主榜标准收录对象 [25][26] |

---

## 二、基础大语言模型：3 月重点发布情况

### 1. OpenAI：GPT-5.4 系列是 3 月最重要的基础模型发布之一

OpenAI 在 2026 年 3 月发布了 **GPT-5.4**，并在 3 月 17 日进一步发布 **GPT-5.4 mini / nano**。[1][2][4]

#### 核心信息
- **发布日期**：GPT-5.4 为 3 月 5 日；mini 和 nano 为 3 月 17 日
- **类别**：基础大语言模型，同时覆盖 Agent 与 Coding
- **能力重点**：
  - 推理、编码、代理工作流统一
  - 支持更深的网页检索与工具生态
  - 视觉理解、文档解析增强
  - 在专业文档密集型任务中准确率显著提升
- **技术规格**
  - GPT-5.4 API 最高支持 **100 万 token 上下文**
  - GPT-5.4 mini 支持 **40 万 token 上下文**
  - 参数量官方未披露

#### LMArena 表现
Arena 文本榜 2026 年 3 月快照中，**gpt-5.4-high** 位于前列，约为前 3 名之一；同时 3 月和 4 月初的 Arena 更新日志也记录了 GPT-5.4 家族进入榜单体系。[2][3][27]

这意味着 GPT-5.4 不仅是“3 月发布”，也是“3 月 Arena 重要上榜新模型”。

---

### 2. 阿里巴巴：Qwen3.5-Max-Preview 与 Qwen3.5-Omni 构成 3 月最强中国厂商发布组合之一

阿里在 3 月有三条非常关键的发布线：[15][17][18]

#### Qwen3.5-Max-Preview
- **发布日期**：2026-03-18
- **类别**：基础 LLM
- **定位**：旗舰预览版通用模型
- **亮点**：
  - 在 Arena 首次亮相即取得全球前列成绩
  - 面向高难文本、推理与综合能力场景
- **技术规格**：
  - 官方未在该预览版页面完整公开参数
  - Qwen 3.5 路线整体强调 native multimodal agents 与 MoE 架构

#### Qwen3.5-Omni
- **发布日期**：2026-03-29
- **类别**：全模态模型
- **亮点**：
  - 同时处理文本、图像、音频、视频
  - 面向“原生全模态 Agent”
- **技术规格**：
  - **256K 上下文**
  - 可处理超 10 小时音频、约 400 秒 720p 视听内容
  - **113 种语音识别语言**
  - **36 种语音合成语言**

#### Qwen-Agent
- **发布日期**：2026-03-05
- **类别**：Agent 框架 / Agent 能力生态
- **亮点**：
  - 原生函数调用
  - 代码解释器
  - RAG
  - MCP 支持
  - 浏览器自动化
  - 移动端可运行 4B 级模型

#### LMArena 表现
Qwen3.5-Max-Preview 在 3 月 Arena 中表现突出，公开传播口径普遍指向**全球前五附近、中国厂商第一梯队**；Arena 官方更新日志也显示 Qwen 系列在该阶段持续进入榜单。[15][16][27]

---

### 3. Google：Gemini 3.1 Flash Live 属于 3 月重要多模态语言/语音模型发布

Google 在 3 月最明确的新模型发布之一是 **Gemini 3.1 Flash Live**。[5]

#### 核心信息
- **发布日期**：2026-03-26
- **类别**：实时语音/多模态模型
- **亮点**：
  - 更自然实时的语音对话
  - 支持复杂函数调用
  - 多语言覆盖广
  - 生成音频自带 SynthID 水印
- **技术指标**
  - 官方给出 ComplexFuncBench Audio 分数 **90.8%**
  - 参数量未披露

#### 关于 Arena
Gemini 3 / 3.1 系列在 Arena 文本榜、视觉榜、搜索榜均处于长期头部位置。虽然 Flash Live 本身更偏“实时语音交互模型”，但 Google 家族在 3 月 Arena 仍是最稳定的头部阵营之一。[3][28][29][30]

---

### 4. 微软：Phi-4-Reasoning-Vision-15B 是 3 月值得纳入的多模态基础模型

微软在 3 月公开了 **Phi-4-Reasoning-Vision-15B**。[8]

#### 核心信息
- **日期**：2026 年 3 月
- **类别**：多模态推理模型
- **亮点**：
  - 视觉理解 + 多步推理结合
  - 面向实时交互和 Agent 场景
- **技术规格**：
  - **15B 参数**
- **Arena情况**：
  - 不是 Arena 综合文本榜顶级主力，但属于头部厂商 3 月正式公开的重要模型

---

## 三、图像生成模型：3 月发布与可纳入清单

### 1. Midjourney V8 Alpha

Midjourney 在 3 月 17 日开启 **V8 Alpha** 测试。[14]

#### 信息要点
- **发布日期**：2026-03-17
- **类别**：图像生成
- **状态**：**Alpha 测试，不是正式版**
- **亮点**：
  - 速度约提升 5 倍
  - 细节质量和文本渲染提升
  - 支持更多纵横比、风格引用、moodboard
  - 新增本地 2K 渲染

#### 研究判断
如果按“3 月有新版本开放测试”口径，应纳入；  
如果按“正式发布”口径，应标注为 **预览/测试版**，不能与正式 GA 模型完全等同。

---

### 2. 微软 MAI-Image-2：3 月热度很高，但正式发布时间不在 3 月

这是本次调研中最需要澄清的一个点。[31][32]

- 业界在 3 月已有不少关于 **MAI-Image-2** 排名和能力的讨论
- 但微软官方正式宣布时间是 **2026-04-09**
- 因此它**不应列入“2026 年 3 月正式发布模型清单”**

#### 但它仍值得作为背景说明
- 类别：文本生成图像
- 特点：写实感、光影、肤色和文字生成能力增强
- 在 Arena 图像生成榜中处于很高位置，微软官方口径称为全球前三之一

---

### 3. 3 月图像生成赛道结论

严格按“3 月正式发布/开放”计算，图像生成赛道中最明确、头部且可纳入的对象其实不多：

- **Midjourney V8 Alpha**：可纳入，但须标注为 alpha
- 其他如 MAI-Image-2：**不纳入 3 月正式发布清单**
- Black Forest Labs、Stability AI 等头部厂商在 3 月更多是已有产品延续，并无明确 3 月新旗舰模型首发 [33][34]

---

## 四、视频生成模型：3 月是高活跃月份

### 1. Google Veo 3.1 Lite

Google 在 3 月下旬推出 **Veo 3.1 Lite**。[6][7]

#### 要点
- **类别**：视频生成
- **发布形态**：付费预览
- **亮点**：
  - 成本显著低于 Veo 3.1 Fast
  - 支持 720p / 1080p
  - 支持 4 秒、6 秒、8 秒时长
  - 支持横版和竖版
- **Arena情况**：
  - Veo 系列长期在 Text-to-Video 榜高位
  - Arena 3 月初快照中，Google 的 **veo-3.1-audio-1080p** 位居第一 [30]

这说明 Google 既有新发布，也继续在 Arena 视频赛道保持头部。

---

### 2. 字节跳动 / CapCut：Dreamina Seedance 2.0

这是 3 月视频生成领域最值得重点写入的模型之一。[22][23]

#### 核心信息
- **发布时间**：2026 年 3 月规模化开放
- **类别**：视频生成
- **厂商**：字节跳动 / CapCut
- **亮点**：
  - 支持文本、图像、参考视频输入
  - 生成最长 **15 秒**视频
  - 支持 **6 种比例**
  - 强调版权与安全控制
  - 内置不可见水印与 **C2PA** 内容凭证
- **技术规格**：
  - 参数未公开
- **LMArena情况**：
  - 在 **Image-to-Video** 榜单位列 **第 1**
  - 评分约 **1449±11**

这是少数同时满足“3 月开放”和“Arena 赛道头部”的视频模型。

---

### 3. OpenAI：Sora 相关变化

OpenAI 在 3 月没有明确“全新 Sora 模型首发”，但有重要平台变动：[35][36]

- **2026-03-13**：Sora 1 在美国退役，切换到 **Sora 2**
- 视频与图像生成功能进一步整合入 ChatGPT / 开发者生态
- Sora 2 支持视频编辑、扩展、角色创建等能力

#### Arena 表现
Arena 的 Text-to-Video 榜中，**sora-2-pro** 在 3 月快照中位居前列（约第 4）。[30]

#### 研究判断
- **不宜将 Sora 2 算作“3 月首次发布”**
- 但应作为 3 月视频生成市场的重要产品演进写入背景

---

### 4. Runway Characters：视频生成与实时角色 Agent 的交叉产品

Runway 于 3 月 9 日发布 **Runway Characters**。[13]

#### 要点
- **类别**：视频 Agent / 实时生成角色
- **亮点**：
  - 单张图片生成实时数字角色
  - 支持语音、知识库、动作和个性
  - 面向客服、培训、品牌内容等场景
- **技术基础**：基于 **GWM-1**

它更偏“视频交互 Agent”而不是传统 T2V 模型，但在 3 月的头部厂商产品里很有代表性。

---

## 五、Agent 与 Coding 模型：3 月是密集发布期

### 1. OpenAI：GPT-5.4 系列同时是 Agent/Coding 头部新模型

GPT-5.4 在设计上不只是基础 LLM，而是明显面向 Agent 与 Coding 工作流。[2][4]

#### 代表性能力
- 工具调用
- 长上下文
- 深度检索
- 文档与图像理解
- 高质量代码生成
- mini / nano 覆盖不同延迟与成本档位

因此在本研究中，GPT-5.4 应同时归入：
- 基础 LLM
- Agent 模型
- Coding 模型

---

### 2. 阿里：Qwen-Agent 是 3 月 Agent 生态的重要发布

相比“单一模型”，Qwen-Agent 更像是一个 **Agent 能力框架层**。[17]

#### 关键价值
- 原生函数调用
- 代码执行
- RAG
- MCP
- 浏览器自动化

在 2026 年 3 月这个时间点，Agent 竞争已不只看基座模型，开始转向“模型+工具+协议+执行框架”。Qwen-Agent 很符合这一趋势。

---

### 3. 智谱：GLM-5-Turbo 是典型的 Agent-first 模型

GLM-5-Turbo 明确是为长流水线、多工具协作任务优化。[19]

#### 官方可确认信息
- **发布时间**：3 月中旬
- **上下文**：**200K**
- **最大输出**：**128K**
- **定位**：面向 OpenClaw 等 Agent 生态的 API 模型
- **特点**：
  - 工具调用稳定性
  - 多步骤执行
  - 长链路任务表现

这使其在分类上更适合归入：
- Agent 模型
- Coding/Tool-use 模型  
而非单纯通用聊天模型。

---

### 4. Mistral：Leanstral 是 3 月最有辨识度的代码代理模型之一

Leanstral 的价值不在参数规模，而在其任务明确性。[9]

#### 关键点
- **类别**：代码代理 / 形式化证明
- **参数量**：**0.6B**
- **面向场景**：Lean 4、形式化数学证明、软件验证
- **状态**：开源

这类模型体现了 2026 年代码模型的一个趋势：  
**从通用代码补全，转向面向特定开发工作流的专业代理。**

---

### 5. 美团：LongCat-Flash-Prover 是 3 月代码/证明模型的重量级发布

LongCat-Flash-Prover 在参数规模和目标任务上都极具代表性。[20][21]

#### 核心信息
- **发布日期**：2026-03-24
- **类别**：代码/Prover 模型
- **参数量**：**560B MoE**
- **亮点**：
  - 自动形式化
  - 草图生成
  - 定理证明
  - 在 Lean4 环境表现强
- **指标**：
  - **MiniF2F-Test 97.1%**
  - PutnamBench 解决率 **41.5%**

它不是 Arena 主流偏好榜中的常规对象，但在“代码生成/证明”这个细分方向上，重要性很高。

---

### 6. Cursor Composer 2

Cursor 于 3 月 19 日发布 **Composer 2**。[25][26]

#### 核心信息
- **类别**：Coding / Agent
- **亮点**：
  - 更强的长上下文编码代理能力
  - 多步骤工程任务
  - 复杂重构与调试
- **技术规格**：
  - **20 万上下文**
- **发布形态**：
  - 属于 Cursor 生态内产品模型，而非通用 API 大模型

这类模型也说明：  
2026 年 3 月的代码模型竞争，已经不仅来自基础模型厂商，也来自 IDE/开发工具厂商。

---

## 六、LMArena 榜单：2026 年 3 月的头部格局

本节重点回答“**当时 LMArena leaderboard 榜单上排名的模型**”这一需求。

### 1. 文本总榜（Text Arena）

根据 Arena 文本榜 2026 年 3 月快照，头部格局大致如下：[28]

1. **claude-opus-4-6-thinking**
2. **gemini-3.1-pro-preview**
3. **gpt-5.4-high**

这说明在 3 月末：
- Anthropic 仍在文本总体偏好榜领跑
- Google Gemini 3.1 Pro 紧随其后
- OpenAI GPT-5.4 刚发布即进入顶级梯队

#### 与本次“3 月发布”直接相关的交集
- **GPT-5.4**：强相关，且属于 3 月新发布
- **Gemini 3.1 Pro Preview**：3 月榜单头部，但并非 3 月首次发布
- **Claude Opus / Sonnet 4.6**：榜单头部，但发布早于 3 月

---

### 2. 视觉 / OCR 榜（Vision Arena）

视觉理解与 OCR 榜 3 月头部模型如下：[29]

1. **claude-opus-4-6**
2. **gemini-3-pro**
3. **gemini-3.1-pro-preview**
4. **claude-opus-4-6-thinking**
5. **claude-sonnet-4-6**

#### 观察
- 3 月视觉榜是 **Anthropic + Google 双强格局**
- OpenAI GPT-5.4 虽然有视觉增强，但在视觉榜上不是前五最强主导者
- 微软、阿里等在视觉赛道有进展，但 3 月公开榜单头部仍由 Anthropic / Google 占据

---

### 3. 搜索榜（Search Arena）

截至 2026-03-31 的 Search Arena 头部如下：[30][37]

1. **grok-4.20-multi-agent-beta-0309**
2. **claude-opus-4-6-search**
3. **gemini-3.1-pro-grounding**
4. **grok-4.20-beta1**
5. **grok-4-1-fast-search**

#### 观察
- 搜索增强模型中，**xAI Grok 4.20** 系列在 3 月非常强势
- 但 xAI 3 月并未找到同等明确度的官方“全新基础模型正式发布”信息
- 这说明：**榜单头部模型不一定都在 3 月新发，但必须纳入榜单分析**

---

### 4. Text-to-Video 榜

Arena 3 月视频榜快照显示：[30]

1. **Google veo-3.1-audio-1080p**
2. 其他 Google / 头部视频模型变体
3. …
4. **OpenAI sora-2-pro**

#### 观察
- Google Veo 系列在 3 月视频偏好榜中占优
- OpenAI Sora 2 仍然是强势竞争者
- 视频榜与文本榜最大的不同是：产品化视频模型的排名与“是否在 3 月首发”关联较弱，很多是持续迭代产品

---

### 5. Image-to-Video 榜

**Dreamina Seedance 2.0** 在 Image-to-Video 榜位列第一，是本次调研中“3 月发布 + Arena 榜首”最明确的案例之一。[23]

这使它在视频赛道的研究权重非常高。

---

### 6. 文档、视频编辑等新赛道

Arena 3 月还上线/强化了更多细分赛道：[27][38]

- **文档竞技场**：第一名为 **Claude Opus 4.6**
- **视频编辑竞技场**：第一名为 **xAI Grok-Imagine-Video**

这反映出 2026 年 3 月 Arena 已不再只是聊天大模型榜，而是形成：
- 文本
- 搜索
- 视觉/OCR
- 文档
- WebDev
- 文本转视频
- 图像转视频
- 视频编辑  
等多赛道评估体系。

---

## 七、头部厂商在 2026 年 3 月的“有发布 / 无发布”结论

### 1. 明确有重要模型发布的头部厂商
- **OpenAI**：GPT-5.4、GPT-5.4 mini、GPT-5.4 nano [2][4]
- **Google**：Gemini 3.1 Flash Live、Veo 3.1 Lite [5][6]
- **Microsoft**：Phi-4-Reasoning-Vision-15B [8]
- **Alibaba / Qwen**：Qwen3.5-Max-Preview、Qwen-Agent、Qwen3.5-Omni [15][17][18]
- **Mistral**：Leanstral、Voxtral TTS、Mistral OCR [9][10][11]
- **Cohere**：Transcribe [12]
- **Runway**：Runway Characters [13]
- **ByteDance / CapCut**：Dreamina Seedance 2.0 [22]
- **MiniMax**：M2.7 [24]
- **Cursor**：Composer 2 [25]

### 2. 3 月榜单强势，但未见 3 月新模型正式发布
- **Anthropic**：3 月无明确新主模型发布；但 Claude Opus 4.6 / Sonnet 4.6 在多榜单持续领先 [39][40]
- **Meta**：Llama 4 发布时间早于 3 月；3 月无新旗舰模型 [41]
- **xAI**：Grok 4.20 系列在榜单和搜索赛道强势，但 3 月公开发布信息不如榜单表现那样清晰 [30][37]
- **DeepSeek**：3 月未见新模型首发 [42]
- **Stability AI**：3 月更多是已有模型与产品功能更新，无明确新旗舰首发 [33]
- **Black Forest Labs**：3 月未见新模型首发，较重要发布在 1 月 [34]

---

## 八、关键分析：2026 年 3 月的大模型发布趋势

### 1. “基础模型发布”正在变成“模型家族发布”
3 月最典型的例子是 **GPT-5.4 + mini + nano**。  
厂商不再只发一个主模型，而是同时覆盖：
- 顶级性能
- 低延迟
- 低成本
- 多模态
- Agent 调用

阿里的 Qwen 系列也体现了同一趋势：  
**Max-Preview + Agent + Omni**，分别对应不同产品层。

---

### 2. Agent 与 Coding 已不再是附属能力，而是核心竞争轴
从 OpenAI、Qwen、GLM、Cursor、Mistral、LongCat 可以看出：

- 工具调用
- 浏览器操作
- RAG / MCP
- 代码执行
- 多步骤规划
- 形式化证明

这些都不再是边缘能力，而是主发布卖点。  
也就是说，2026 年 3 月的模型竞争，正在从“谁会聊天”转向“谁能完成工作流”。

---

### 3. 视频生成赛道比图像生成赛道更活跃
在 3 月：
- Google 有 **Veo 3.1 Lite**
- 字节有 **Dreamina Seedance 2.0**
- OpenAI 有 **Sora 2** 持续强化
- Runway 有 **Characters**

相较之下，图像赛道的头部正式新发模型更少，更多是预览版或已有模型延续。

---

### 4. LMArena 已从单一聊天榜进化为多赛道评测系统
对本次调研最重要的启示是：

如果只看文本总榜，会漏掉很多真正重要的产品变化。  
例如：
- 搜索榜里 xAI 很强
- 视频榜里 Google / OpenAI / 字节竞争激烈
- Image-to-Video 榜里 Dreamina Seedance 2.0 登顶
- 文档榜里 Claude Opus 4.6 优势明显

因此，对 2026 年 3 月模型格局的判断，必须采用 **多榜单交叉视角**，不能只看文本聊天榜。

---

## 九、建议采用的最终结论口径

如果需要把本次研究整理成决策层可直接使用的“短结论”，最稳妥的结论是：

### 1. 2026 年 3 月最重要的新模型发布
- **OpenAI GPT-5.4 系列**
- **Alibaba Qwen3.5-Max-Preview / Qwen3.5-Omni / Qwen-Agent**
- **Google Gemini 3.1 Flash Live / Veo 3.1 Lite**
- **ByteDance Dreamina Seedance 2.0**
- **Mistral Leanstral / Mistral OCR**
- **Zhipu GLM-5-Turbo**
- **LongCat-Flash-Prover**
- **Cursor Composer 2**
- **MiniMax M2.7**
- **Runway Characters**

### 2. 2026 年 3 月 LMArena 的主导格局
- **文本总榜**：Anthropic、Google、OpenAI 三强
- **视觉榜**：Anthropic、Google 占优
- **搜索榜**：xAI、Anthropic、Google 强势
- **视频榜**：Google Veo、OpenAI Sora、字节 Dreamina 竞争激烈
- **图像到视频榜**：字节 **Dreamina Seedance 2.0** 领先

### 3. 不应误纳入 3 月正式发布清单的模型
- **MAI-Image-2**：官方正式发布时间为 2026-04-09，不属于 3 月正式发布 [31][32]
- **Sora 2**：3 月有重要产品迁移，但不属于 3 月首次发布 [35][36]
- **Claude 4.6 / Sonnet 4.6**：榜单很强，但发布时间早于 3 月 [40]
- **Llama 4**：发布时间早于 3 月 [41]

---

## 十、按类别整理的精简清单

### 基础大语言模型
- GPT-5.4 / mini / nano
- Qwen3.5-Max-Preview
- Qwen3.5-Omni
- Gemini 3.1 Flash Live
- Phi-4-Reasoning-Vision-15B

### 图像生成模型
- Midjourney V8 Alpha（alpha）
- 其他 3 月头部图像模型更多是已有模型延续，非正式首发

### 视频生成模型
- Veo 3.1 Lite
- Dreamina Seedance 2.0
- Runway Characters（交叉视频/Agent）

### Agent & Coding 模型
- GPT-5.4 系列
- Qwen-Agent
- GLM-5-Turbo
- Leanstral
- LongCat-Flash-Prover
- Cursor Composer 2
- MiniMax M2.7

---

### 来源

[1] OpenAI Model Release Notes: https://help.openai.com/en/articles/9624314-model-release-notes?utm_source=syndication&pubDate=20260309  
[2] Introducing GPT-5.4 - OpenAI: https://openai.com/index/introducing-gpt-5-4/  
[3] LLM Leaderboard - Best Text & Chat AI Models Compared: https://lmarena.ai/leaderboard/text/overall-no-style-control  
[4] Introducing GPT-5.4 mini and nano - OpenAI: https://openai.com/index/introducing-gpt-5-4-mini-and-nano/  
[5] Google AI Updates March 2026 / Gemini 3.1 Flash Live: https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-march-2026/  
[6] Google Veo 3.1 Lite announcement / product update: https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-march-2026/  
[7] Text-to-Video Leaderboard - LMArena: https://lmarena.ai/leaderboard/text-to-video  
[8] Introducing Phi-4-Reasoning-Vision to Microsoft Foundry: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-phi-4-reasoning-vision-to-microsoft-foundry/4499154  
[9] Leanstral - Mistral AI: https://mistral.ai/news/leanstral  
[10] Voxtral TTS - Mistral AI: https://mistral.ai/news/voxtral-tts  
[11] Mistral OCR - Mistral AI: https://mistral.ai/news/mistral-ocr  
[12] Transcribe - Cohere: https://cohere.com/blog/transcribe  
[13] Introducing Runway Characters: https://runwayml.com/news/introducing-runway-characters  
[14] Midjourney V8 Alpha announcement: https://updates.midjourney.com/v8-alpha/  
[15] Qwen Research / Qwen3.5 updates: https://qwen.ai/research  
[16] Qwen3.5-Max-Preview / Qwen ecosystem update: https://qwen.ai/blog?id=qwen3.5  
[17] Qwen-Agent framework update: https://www.simplenews.ai/news/alibabas-qwen-team-releases-official-agent-framework-with-native-function-calling-and-mcp-support-2db6  
[18] Qwen3.5-Omni: Scaling Up, Toward Native Omni-Modal AGI: https://qwen.ai/blog?id=qwen3.5-omni  
[19] GLM-5-Turbo / Z.ai release notes: https://docs.z.ai/release-notes/new-released  
[20] LongCat-Flash-Prover GitHub: https://github.com/meituan-longcat/LongCat-Flash-Prover  
[21] LongCat-Flash-Prover Technical Report: https://github.com/meituan-longcat/LongCat-Flash-Prover/blob/main/LongCat_Flash_Prover_Technical_Report.pdf  
[22] Dreamina Seedance 2.0 - CapCut Newsroom: https://www.capcut.com/newsroom/dreamina-seedance-2  
[23] Image-to-Video Leaderboard - LMArena: https://lmarena.ai/leaderboard/image-to-video  
[24] MiniMax M2.7 news: https://www.minimax.io/news/minimax-m27-en  
[25] Introducing Composer 2 - Cursor Forum: https://forum.cursor.com/t/introducing-composer-2/155288  
[26] Cursor Composer 2 resources: https://cursor.com/resources/Composer2.pdf  
[27] Leaderboard Changelog - Arena: https://news.lmarena.ai/leaderboard-changelog/  
[28] Text Arena leaderboard: https://lmarena.ai/leaderboard/text/overall-no-style-control  
[29] Vision AI Leaderboard - OCR: https://lmarena.ai/leaderboard/vision/ocr  
[30] March 2026 Arena Updates: https://arena.ai/blog/march-2026-arena-updates/  
[31] Introducing MAI-Image-2 - Microsoft AI: https://microsoft.ai/news/introducing-mai-image-2/  
[32] Microsoft announces new MAI models in Foundry: https://microsoft.ai/news/today-were-announcing-3-new-world-class-mai-models-available-in-foundry  
[33] Stability AI news / product updates: https://stability.ai/news  
[34] FLUX.2 [klein] - Black Forest Labs: https://blackforestlabs.ai/blog/flux2-klein-towards-interactive-visual-intelligence  
[35] Sora Demo - OpenAI Developers: https://developers.openai.com/showcase/sora-sample-app  
[36] OpenAI model/product release notes including Sora transition: https://help.openai.com/en/articles/9624314-model-release-notes?utm_source=syndication&pubDate=20260309  
[37] Search AI Leaderboard - LMArena: https://lmarena.ai/leaderboard/search/overall-add-style-control  
[38] Arena Blog: https://news.lmarena.ai/  
[39] Anthropic Newsroom: https://www.anthropic.com/news  
[40] Introducing Claude Sonnet 4.6 - Anthropic: https://www.anthropic.com/news/claude-sonnet-4-6  
[41] The Llama 4 herd - Meta AI: https://ai.meta.com/blog/llama-4-multimodal-intelligence/  
[42] DeepSeek official site / release status: https://cdn.deepseek.com/