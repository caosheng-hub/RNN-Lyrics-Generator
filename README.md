# RNN-Lyrics-Generator
基于PyTorch实现的周杰伦歌词生成器，采用RNN（可扩展为LSTM/GRU）构建序列生成模型，兼顾工程化实现与学术性分析，适用于NLP/深度学习方向的算法实习展示。

## 🌟 项目亮点
1. **工程化规范**：模块化拆分代码，添加类型提示/文档字符串，符合工业界代码规范；
2. **可复现性**：提供完整的环境依赖、数据和训练脚本，一键复现训练/推理；
3. **学术性思考**：包含模型设计、实验分析等深度内容；
4. **扩展性**：模型支持替换为LSTM/GRU、调整嵌入维度/隐藏层维度，适配不同场景；
5. **可视化分析**：训练损失曲线、生成效果对比，体现数据分析能力。

## 🛠 技术栈
- 框架：PyTorch（2.0+）
- 分词：jieba
- 可视化：matplotlib、Jupyter
- 工程规范：PEP8、类型提示、模块化设计

## 🚀 快速开始
### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/caosheng-hub/RNN-Lyrics-Generator.git
cd RNN-Lyrics-Generator

# 安装依赖
pip install -r requirements.txt
