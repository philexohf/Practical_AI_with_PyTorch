import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        """
        初始化位置编码模块。

        参数:
        - d_model: 模型的维度。
        - max_len: 最大序列长度，默认为 1000。
        - dropout: Dropout 的概率，默认为 0.1。
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))  # 修正这里
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区

    def forward(self, X):
        """
        前向传播。

        参数:
        - X: 输入张量，形状为 (batch_size, seq_len, d_model)。

        返回:
        - 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        X = X + self.pe[:, :X.size(1)]  # 将位置编码加到输入上
        return self.dropout(X)
    

# 2. 多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, valid_lens=None):
        batch_size, seq_len, d_model = X.shape

        # 线性变换并分头
        queries = self.transpose_qkv(self.W_q(X), self.num_heads)  # (batch_size * num_heads, seq_len, head_dim)
        keys = self.transpose_qkv(self.W_k(X), self.num_heads)     # (batch_size * num_heads, seq_len, head_dim)
        values = self.transpose_qkv(self.W_v(X), self.num_heads)   # (batch_size * num_heads, seq_len, head_dim)

        # 计算点积注意力
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.head_dim)  # (batch_size * num_heads, seq_len, seq_len)
        attention_weights = self.masked_softmax(scores, valid_lens)  # (batch_size * num_heads, seq_len, seq_len)

        # 加权求和
        attention_output = torch.bmm(self.dropout(attention_weights), values)  # (batch_size * num_heads, seq_len, head_dim)

        # 恢复形状
        attention_output = self.transpose_output(attention_output, self.num_heads)  # (batch_size, seq_len, d_model)

        # 输出线性变换
        output = self.W_o(attention_output)  # (batch_size, seq_len, d_model)
        return output

    def transpose_qkv(self, X, num_heads):
        batch_size, seq_len, d_model = X.shape
        head_dim = d_model // num_heads
        X = X.reshape(batch_size, seq_len, num_heads, head_dim)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(-1, seq_len, head_dim)
        return X

    def transpose_output(self, X, num_heads):
        batch_size_num_heads, seq_len, head_dim = X.shape
        batch_size = batch_size_num_heads // num_heads
        X = X.reshape(batch_size, num_heads, seq_len, head_dim)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(batch_size, seq_len, -1)
        return X

    def masked_softmax(self, X, valid_lens):
        if valid_lens is None:
            return F.softmax(X, dim=-1)
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        mask = torch.arange(shape[-1], device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = -1e6
        return F.softmax(X, dim=-1)


# 3. 前馈神经网络
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        return self.linear2(self.dropout(F.relu(self.linear1(X))))


# 4. Encoder 层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, valid_lens=None):
        # 多头自注意力
        attention_output = self.attention(X, valid_lens)
        X = X + self.dropout(attention_output)
        X = self.norm1(X)

        # 前馈神经网络
        ffn_output = self.ffn(X)
        X = X + self.dropout(ffn_output)
        X = self.norm2(X)
        return X


# 5. Decoder 层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attention2 = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attention_output = self.attention1(X, tgt_mask)
        X = X + self.dropout(attention_output)
        X = self.norm1(X)

        # 编码器-解码器注意力
        attention_output = self.attention2(X, src_mask)
        X = X + self.dropout(attention_output)
        X = self.norm2(X)

        # 前馈神经网络
        ffn_output = self.ffn(X)
        X = X + self.dropout(ffn_output)
        X = self.norm3(X)
        return X


# 6. Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        # 解码器
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出
        output = self.linear(decoder_output)
        return output


# 7. 测试 Transformer 模型
if __name__ == "__main__":
    # 定义超参数
    src_vocab_size = 100256  # 源语言词汇表大小
    tgt_vocab_size = 100256  # 目标语言词汇表大小
    d_model = 512          # 模型维度
    num_heads = 8          # 多头注意力头数
    num_layers = 6         # 编码器和解码器层数
    d_ff = 2048            # 前馈神经网络隐藏层维度
    dropout = 0.1          # Dropout 概率
    max_len = 1000          # 最大序列长度

    # 实例化 Transformer 模型
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)

    # 创建随机输入
    src = torch.randint(0, src_vocab_size, (32, 10))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (32, 10))  # (batch_size, tgt_seq_len)

    # 前向传播
    output = transformer(src, tgt)
    print("Transformer 输出形状:", output.shape)  # (batch_size, tgt_seq_len, tgt_vocab_size)
    