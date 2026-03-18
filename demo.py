import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
# ===================== 1. 数据准备（简化版，确保文件存在） =====================
def build_vocab():
    # 确保数据文件存在，这里提供一段示例歌词（也可替换为本地文件）
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # 写入示例歌词到文件（避免文件不存在报错）
    demo_lyrics = """
    手牵手 一步两步三步四步 望着天
    看星星 一颗两颗三颗四颗 连成线
    背对背默默许下心愿 看远方的星 是否听得见
    手牵手 漫步在海边的沙滩
    风吹过 你的长发拂过我的脸
    """
    with open('./data/jaychou_lyrics.txt', 'w', encoding='utf-8') as f:
        f.write(demo_lyrics)

    unique_words, all_words = [], []
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line.strip())  # 去除换行符，避免空词
        if not words:
            continue
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # 补充特殊字符（避免索引错误）
    if ' ' not in unique_words:
        unique_words.append(' ')
    word_count = len(unique_words)
    word_to_index = {w: i for i, w in enumerate(unique_words)}

    # 构建语料索引
    corpus_idx = []
    for words in all_words:
        tmp = [word_to_index[w] for w in words]
        tmp.append(word_to_index[' '])  # 分隔符
        corpus_idx.extend(tmp)
    return unique_words, word_to_index, word_count, corpus_idx


# ===================== 2. 数据集类 =====================
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars=32):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count = len(corpus_idx)
        self.number = max(1, self.word_count // self.num_chars)  # 至少1个样本

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(idx * self.num_chars, self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1:end + 1]
        return torch.tensor(x), torch.tensor(y)


# ===================== 3. RNN 模型 =====================
class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        self.ebd = nn.Embedding(unique_word_count, embedding_dim=64)  # 简化维度，加快训练
        self.rnn = nn.RNN(64, 128, 1, batch_first=True)  # 改用batch_first，更直观
        self.out = nn.Linear(128, unique_word_count)

    def forward(self, inputs, hidden):
        embd = self.ebd(inputs)  # [batch, seq_len, 64]
        output, hidden = self.rnn(embd, hidden)  # [batch, seq_len, 128]
        output = self.out(output.reshape(-1, output.shape[-1]))  # [batch*seq_len, vocab_size]
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128)  # [num_layers, batch, hidden_dim]


# ===================== 4. 训练函数（简化版） =====================
def train_model():
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    dataset = LyricsDataset(corpus_idx, num_chars=8)  # 缩短序列长度，加快训练
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TextGenerator(word_count)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # 增大学习率，快速收敛

    epochs = 20  # 减少训练轮数，快速测试
    print("开始训练...")
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0.0
        iter_num = 0

        for x, y in dataloader:
            hidden = model.init_hidden(x.shape[0])
            output, hidden = model(x, hidden)

            # 调整y的形状，匹配output
            y = y.reshape(-1)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iter_num += 1

        avg_loss = total_loss / iter_num if iter_num > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs} | 耗时: {time.time() - start:.2f}s | 平均损失: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), './lyrics_generator_demo.pth')
    return model, unique_words, word_to_index


# ===================== 5. 生成歌词函数 =====================
def generate_lyrics(start_word, gen_length=20):
    # 加载词典和模型
    unique_words, word_to_index, word_count, _ = build_vocab()
    model = TextGenerator(word_count)
    try:
        model.load_state_dict(torch.load('./lyrics_generator_demo.pth'))
    except:
        print("未找到训练好的模型，先训练...")
        model, _, _ = train_model()

    model.eval()  # 切换到评估模式
    hidden = model.init_hidden(1)

    # 检查起始词是否在词典中
    if start_word not in word_to_index:
        print(f"起始词{start_word}不在词典中，改用第一个词：{unique_words[0]}")
        start_word = unique_words[0]
    word_idx = word_to_index[start_word]
    generate_idx = [word_idx]

    # 生成歌词
    with torch.no_grad():  # 禁用梯度计算，加快速度
        for _ in range(gen_length):
            # 输入形状：[batch=1, seq_len=1]
            output, hidden = model(torch.tensor([[word_idx]]), hidden)
            word_idx = torch.argmax(output, dim=1).item()  # 取概率最大的词
            generate_idx.append(word_idx)

    # 转换为文字
    generated_lyrics = ''.join([unique_words[idx] for idx in generate_idx])
    return generated_lyrics


# ===================== 6. 运行Demo =====================
if __name__ == '__main__':
    # 1. 训练模型（首次运行需要）
    train_model()

    # 2. 生成歌词
    start_word = "手牵手"
    generated = generate_lyrics(start_word, gen_length=30)
    print("\n生成的歌词：")
    print(generated)