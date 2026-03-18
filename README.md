---

# 🎵 基于 RNN 的中文歌词生成器（PyTorch）

> 🧠 使用循环神经网络（RNN）生成中文歌词
> 🎤 数据集：周杰伦歌词
> 🔥 支持训练 + 文本生成

---

## 📌 项目简介

本项目基于 PyTorch 实现一个**字符级中文歌词生成模型**，通过训练 RNN 学习歌词的语言模式，实现：

👉 输入开头词 → 自动生成完整歌词

适用于：

* NLP 入门项目
* RNN / 序列模型学习
* 文本生成任务实践

---

## 🧠 模型原理

### 🔁 RNN 语言模型

核心思想：

```text
给定前面的词 → 预测下一个词
```

模型学习：

```text
P(w_t | w_1, w_2, ..., w_{t-1})
```

---

## 🏗️ 模型结构

```text
输入词索引
    ↓
Embedding（词向量）
    ↓
RNN（序列建模）
    ↓
Linear（映射到词表）
    ↓
Softmax（概率分布）
    ↓
预测下一个词
```

---

## 📂 项目结构

```bash
.
├── data/
│   └── jaychou_lyrics.txt     # 🎵 歌词数据集
│
├── RNN_AI歌词生成器案例.py     # 🚀 主程序（训练 + 预测）
│
├── model.pth                 # 💾 训练好的模型（可选）
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

---

## ⚙️ 环境依赖

```bash
pip install torch jieba
```

---

## 🚀 快速开始

### 1️⃣ 准备数据

将歌词文件放入：

```bash
data/jaychou_lyrics.txt
```

---

### 2️⃣ 训练模型

```bash
python RNN_AI歌词生成器案例.py
```

训练过程中输出：

```text
epoch:1, loss:xxx
epoch:2, loss:xxx
...
```

训练完成后：

```bash
model.pth
```

---

### 3️⃣ 生成歌词

修改代码中的：

```python
evaluate('手牵手', 50)
```

运行后输出：

```text
手牵手一起走在风中...
```

---

## 🧩 核心模块解析

---

### 1️⃣ 数据处理（词表构建）

👉 使用 `jieba` 分词

```python
words = jieba.lcut(line)
```

生成：

* `unique_words`：词表
* `word_to_index`：词→索引
* `corpus_idx`：文本序列

✔ 将文本转为数字（模型可处理）

---

### 2️⃣ 数据集构建

自定义 Dataset：

```python
class LyricsDataset(torch.utils.data.Dataset)
```

输入输出：

| 输入 x         | 目标 y         |
| ------------ | ------------ |
| [w1, w2, w3] | [w2, w3, w4] |

👉 本质：**下一个词预测**

---

### 3️⃣ 模型结构

```python
Embedding → RNN → Linear
```

#### 📌 Embedding

```python
nn.Embedding(vocab_size, 128)
```

#### 📌 RNN

```python
nn.RNN(128, 256, 1)
```

#### 📌 输出层

```python
nn.Linear(256, vocab_size)
```

---

### 4️⃣ 训练流程

关键步骤：

```python
loss = CrossEntropyLoss
optimizer = Adam
```

训练流程：

1. 前向传播
2. 计算loss
3. 反向传播
4. 更新参数

---

### 5️⃣ 文本生成（核心亮点🔥）

生成逻辑：

```text
输入一个词 → 预测下一个词 → 再作为输入 → 循环
```

代码核心：

```python
word_idx = torch.argmax(output)
```

👉 每次选择概率最大的词（贪心策略）

---

## 📊 模型参数

| 参数            | 值   |
| ------------- | --- |
| embedding_dim | 128 |
| hidden_size   | 256 |
| num_layers    | 1   |
| seq_length    | 32  |
| batch_size    | 5   |
| epochs        | 50  |

---

## 💡 项目亮点

### ⭐ 1. 完整 NLP 流程

✔ 数据预处理
✔ 模型训练
✔ 文本生成

---

### ⭐ 2. 自定义 Dataset

* 动态切片序列
* 支持批量训练

---

### ⭐ 3. 序列建模能力

* 使用 RNN 捕捉上下文关系
* 实现语言模型

---

### ⭐ 4. 中文处理能力

* 使用 `jieba` 分词
* 支持中文语料训练

---
