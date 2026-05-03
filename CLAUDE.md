# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作提供指导。

## 项目概述

这是一个基于 HuggingFace `transformers` 生态的 NLP 学习项目，内容以 Jupyter Notebook (`.ipynb`) 为主，按循序渐进的方式组织。包含少量独立的 Python 脚本用于分布式训练，以及一个从零实现的 Transformer 模型。

## 环境与依赖

项目无 `requirements.txt` 或 `pyproject.toml`。核心依赖为：
- `transformers`、`datasets`、`evaluate`、`peft`、`accelerate`、`torch`
- 模型从**本地路径**加载（如 `/tmp/pretrainmodel/`），不从 HuggingFace Hub 在线加载
- 下载模型和数据集使用国内镜像：
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  huggingface-cli download --resume-download gpt2 --local-dir gpt2
  huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
  ```
- 代码在启智平台使用 V100*2 和 A100 等 GPU 进行训练

## 课程结构

| 阶段 | 目录 | 内容 |
|---|---|---|
| 入门 | `01-introduction/` ~ `07-trainer/` | Pipeline 管道、Tokenizer 分词器、Model 模型、Datasets 数据集、Evaluate 评估、Trainer 训练器 |
| 进阶 | `08-text-classification/` ~ `17-generative_chatbot/` | 各任务专项训练：文本分类、NER、问答、多选题、句子相似度、检索式聊天机器人、因果语言模型、掩码语言模型、文本摘要、生成式聊天机器人 |
| PEFT | `18-peft_bitfit/` ~ `23-peft_ia3/` | 参数高效微调：BitFit、Prompt Tuning、LoRA（含 LoHa/LoKr）、P-Tuning、Prefix Tuning、IA3 |
| 优化 | `24-16bits/`、`25-8bits/`、`26-districuted_dp/`、`27-accelerate_ddp/` | 混合精度训练（16位）、量化（8位）、分布式训练（torch DDP、Trainer DDP、Accelerate） |
| 从零实现 | `builtins_transformer/` | 纯 PyTorch 实现完整 Transformer（编码器-解码器），在 G2P（字素转音素）任务上训练 |
| 实战应用 | `whtcc_01_Qwen3_SFT_Study/` | Qwen3 系列模型（1.7B、8B、30B-A3B MoE）的 SFT 和 LoRA 微调 |

## 通用模式

- 主要语言：中文（注释、文档、模型、数据集均为中文）
- 常用数据集：ChnSentiCorp（情感分类）、CMRC 2018（阅读理解）、C3（多选题）、NLPCC 2017（摘要）、alpaca_data_zh（指令微调）
- 常用模型：`chinese-roberta-wwm-ext`、ChatGLM3-6B、Llama-3.2-1B、Qwen3 系列
- 几乎所有 Notebook 头部都设置了 `HF_ENDPOINT` 环境变量指向 `https://hf-mirror.com`
- 模型从 `/tmp/pretrainmodel/` 或类似本地路径加载
- `data/` 目录存放共享数据集，被各章节复制使用

## 分布式训练命令

`26-districuted_dp/` 和 `27-accelerate_ddp/` 中的 Python 脚本在文件开头注释中包含启动命令：

- **torchrun DDP**：`torchrun --nproc-per-node=2 <脚本名>.py`
- **Accelerate**：先 `accelerate config` 配置，再 `accelerate launch <脚本名>.py`

## 关键独立文件

- `builtins_transformer/config.py` — 从零实现 Transformer 的全部超参数（编码器/解码器维度、注意力头数、训练参数等，作为 `HP` 类的属性）
- `builtins_transformer/model.py` — 完整 Transformer 实现：PositionalEncoding、MultiHeadAttention、FeedForward、Encoder、Decoder 层
- `builtins_transformer/train.py` — G2P 任务的训练循环，含 TensorBoard 日志
- `builtins_transformer/dataloaders.py` — G2P（字素转音素）数据的自定义 Dataset/DataLoader/collate_fn
- `builtins_transformer/utils.py` — 字素/音素词汇表及 char2id/id2char 映射
- `10-question_answering/cmrc_eval.py` — CMRC 2018 评估指标（F1、EM 分数，结合中文分词）
