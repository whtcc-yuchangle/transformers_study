# transformers_study

基于 HuggingFace `transformers` 生态的 NLP 深度学习项目，涵盖从基础 Pipeline 到分布式训练与大模型微调的完整学习路径。

## 下载说明

HuggingFace 官网直接访问需要梯子，可通过国内镜像网站下载模型和数据集：

```shell
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 模型下载
huggingface-cli download --resume-download gpt2 --local-dir gpt2

# 数据集下载
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

模型统一从本地路径（如 `/tmp/pretrainmodel/`）加载，不从 HuggingFace Hub 在线获取。

## 项目结构

```
├── 01-introduction/              # Transformers 库简介
├── 02-pipeline/                  # Pipeline 管道 API（10 个示例）
│   ├── 01_pipeline_show_task     # Pipeline 支持的任务总览
│   ├── 02_text-classification    # 文本分类
│   ├── 03_from_pretrained        # 从预训练模型加载
│   ├── 04_QuestionAnswer         # 阅读理解问答
│   ├── 05_CV_object_detect       # 目标检测
│   ├── 06_pipeline_details       # Pipeline 内部机制详解
│   ├── 07_facebook_detr-resnet   # DETR 目标检测
│   ├── 08_image-classification   # 图像分类
│   ├── 09_visual-question-answering  # 视觉问答
│   └── 10_automatic-speech-recognition  # 语音识别
├── 03-tokenizer/                 # 分词器：基础、快慢分词器、特殊 Token
├── 04-model/                     # 模型：PyTorch 从零搭建 + 预训练模型基本用法
├── 05-datasets/                  # Datasets 库：基础用法、音频、视觉、NLP、文本分类
├── 06-evaluate/                  # Evaluate 库：基本用法、Evaluator、分类评估
├── 07-trainer/                   # Trainer API：文本分类实战
├── 08-text-classification/       # 文本分类深入
├── 09-token-classification/      # 命名实体识别（NER）
├── 10-question_answering/        # 抽取式问答（CMRC 2018 数据集）
├── 11-multiple_choice/           # 多选题（C3 数据集）
├── 12-sentence_similarity/       # 句子相似度（跨模型 + 双模型）
├── 13-retrieval_chatbot/         # 检索式聊天机器人
├── 14-causal_language_model/     # 因果语言模型训练
├── 15-masked_language_model/     # 掩码语言模型训练
├── 16-text_summarization/        # 文本摘要（通用 + ChatGLM 模型）
├── 17-generative_chatbot/        # 生成式聊天机器人
├── 18-peft_bitfit/               # PEFT：BitFit 方法
├── 19-peft_prompt_tuning/        # PEFT：Prompt Tuning
├── 20-peft_lora_tuning/          # PEFT：LoRA 微调（含 LoHa、LoKr 变体 + 聊天机器人）
├── 21-peft_p_tuning/             # PEFT：P-Tuning
├── 22-peft_prefix_tuning/        # PEFT：Prefix Tuning
├── 23-peft_ia3/                  # PEFT：IA3（聊天机器人 + HuggingFace 原生方法）
├── 24-16bits/                    # 混合精度训练（16位）：ChatGLM3-6B LoRA + Llama-3.2-1B
├── 25-8bits/                     # 8位量化：ChatGLM3 8-bit 推理
├── 26-districuted_dp/            # 分布式训练：torch DDP + HuggingFace Trainer DDP
├── 27-accelerate_ddp/            # 分布式训练：HuggingFace Accelerate
├── builtins_transformer/         # 从零实现 Transformer（编码器-解码器，G2P 任务）
├── whtcc_01_Qwen3_SFT_Study/     # Qwen3 系列 SFT & LoRA 微调实战
└── data/                         # 共享数据集文件
```

## 学习路线

### 第一阶段：基础入门（01 ~ 07）

从 Pipeline 管道快速上手，逐步深入 Tokenizer、Model、Datasets、Evaluate 等核心组件，最终通过 Trainer API 完成一个完整的文本分类任务。

### 第二阶段：任务实战（08 ~ 17）

覆盖 NLP 主流任务方向：文本分类、命名实体识别、阅读理解、多选题、句子相似度、检索式对话、因果/掩码语言模型、文本摘要、生成式对话。

### 第三阶段：参数高效微调（18 ~ 23）

系统学习 PEFT（Parameter-Efficient Fine-Tuning）六大方法：
- **BitFit**：仅微调偏置项
- **Prompt Tuning**：在输入层添加可学习的软提示
- **LoRA**：低秩适配（含 LoHa、LoKr 变体）
- **P-Tuning**：在嵌入层添加可学习的连续提示
- **Prefix Tuning**：在每一层添加可学习的前缀向量
- **IA3**：通过三个可学习向量缩放注意力及前馈网络

### 第四阶段：性能优化（24 ~ 27）

- **混合精度训练（16-bit）**：使用 `fp16` 加速大模型训练，降低显存占用
- **8-bit 量化**：使用 `bitsandbytes` 实现 8-bit 模型推理与训练
- **分布式训练**：掌握 torch DDP、HuggingFace Trainer DDP 和 Accelerate 三种多卡训练方式

### 第五阶段：深入原理（builtins_transformer）

用纯 PyTorch 从零实现 "Attention Is All You Need" 论文中的完整 Transformer 架构，包括位置编码、多头注意力、前馈网络、编码器-解码器等全部组件，在字素转音素（G2P）任务上完成训练。

### 第六阶段：实战应用（whtcc_01_Qwen3_SFT_Study）

对 Qwen3 系列模型（1.7B 密集模型、8B 密集模型、30B-A3B MoE 混合专家模型）进行 SFT 监督微调和 LoRA 低秩适配，涵盖训练与推理全流程。

## 环境依赖

项目无 `requirements.txt`，核心依赖库如下：

- `transformers` — HuggingFace 模型库
- `datasets` — 数据集加载与处理
- `evaluate` — 评估指标
- `peft` — 参数高效微调
- `accelerate` — 分布式训练加速
- `torch` — PyTorch 深度学习框架
- `modelscope` — 部分模型使用魔搭社区加载

## 常用模型与数据集

**模型**：`chinese-roberta-wwm-ext`、ChatGLM3-6B、Llama-3.2-1B、Qwen3-1.7B/8B/30B-A3B

**数据集**：ChnSentiCorp（中文情感分类）、CMRC 2018（中文阅读理解）、C3（中文多选题）、NLPCC 2017（中文摘要）、alpaca_data_zh（中文指令数据）

## 分布式训练

```bash
# torchrun DDP（两卡训练）
torchrun --nproc-per-node=2 04_ddp_trainer_demo.py

# Accelerate
accelerate config
accelerate launch 01_ddp.py
```

训练脚本在启智平台使用 V100*2、A100 等 GPU 完成验证。

## 致谢

感谢启智社区 (https://openi.pcl.ac.cn) 提供的 GPU 算力支持。

如果本平台对您的科研工作提供了帮助，可在论文致谢中加入：
- 英文版：Thanks for the support provided by OpenI Community (https://openi.pcl.ac.cn).
- 中文版：感谢启智社区提供的技术支持 (https://openi.pcl.ac.cn).

欢迎在以下开源项目中提交引用本平台的成果信息：https://openi.pcl.ac.cn/OpenIOSSG/references
