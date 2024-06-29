import math
import torch
import pandas as pd
from torch import nn


# 定义一个函数，用于显示热图
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    # 获取矩阵的行数和列数
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    # 创建一个图形和一组子图轴
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, 
                                 sharey=True, squeeze=False)
    
    # 遍历每一行的子图轴和对应的矩阵
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        # print(i, row_axes, row_matrices)
        # 遍历每一行的子图轴和对应的矩阵
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # print(j, ax, matrix)
            # 在子图轴上显示矩阵的热图
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            # 如果是最后一行，则设置x轴标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 如果是第一列，则设置y轴标签
            if j == 0:
                ax.set_ylabel(ylabel)
            # 如果提供了标题，则设置子图的标题
            if titles:
                ax.set_title(titles[j])
                
    # 为整个图形添加颜色条
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    # 显示图形
    plt.show()
    

# 掩蔽softmax函数，用于在softmax操作中屏蔽无效的数据
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
        X:3D张量
        valid_lens:1D张量, valid_lens.shape == torch.Size([X.shape[0]])
        valid_lens:2D张量, valid_lens.shape == torch.Size([X.shape[0], X.shape[1]])
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    
    else:
        shape = X.shape
        
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = X.reshape(-1, shape[-1])
        arange_X = torch.arange((X.size(-1)), dtype=torch.float32, device=X.device)
        # 将序列arange_X与valid_len进行比较，生成一个布尔类型的掩码
        mask = arange_X[None, :] < valid_lens[:, None]
        # print(f'MASK:\n{mask} \n X:\n{X}')
        X[~mask] = -1e6
        
        # 对掩蔽后的X应用softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # 初始化dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens=None):
        # 查询、键和值的维度应相同
        d = queries.shape[-1]
        # 计算查询和键的点积，然后除以d的平方根进行缩放
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 应用掩蔽softmax获取注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 应用dropout并计算加权值
        return torch.bmm(self.dropout(self.attention_weights), values)


# 用于变换查询、键、值的形状以适应多头注意力的函数
def transpose_qkv(X, num_heads):
    # 将输入X变换为适合多头注意力的形状
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 重新排列维度
    X = X.permute(0, 2, 1, 3)
    # 展平前两个维度以进行批处理操作
    return X.reshape(-1, X.shape[2], X.shape[3])

# 用于变换多头注意力的输出形状的函数
def transpose_output(X, num_heads):
    # 将多头输出变换回原始形状
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # 重新排列维度
    X = X.permute(0, 2, 1, 3)
    # 展平最后两个维度
    return X.reshape(X.shape[0], X.shape[1], -1)


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        # 设置多头的数量
        self.num_heads = num_heads
        # 初始化点积注意力模块
        self.attention = DotProductAttention(dropout)
        # 初始化查询、键、值的线性变换层
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        # 初始化输出的线性变换层
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        
    def forward(self, queries, keys, values, valid_lens):
        # 将查询、键、值通过各自的线性层，并按照多头的方式进行变换
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        print("将查询、键、值通过各自的线性层，并按照多头的方式进行变换:\n")
        print(f'Q.shape = {queries.shape}, K.shape = {keys.shape}, V.shape = {values.shape}')
        
        # 如果提供了有效长度，则在多头注意力中应用掩蔽
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        # 计算注意力输出
        output = self.attention(queries, keys, values, valid_lens)
        print("注意力输出:\n")
        print(output.shape)
        # 将多头的输出变换回原始形状
        output_concat = transpose_output(output, self.num_heads)
        print("将多头的输出变换回原始形状:\n")
        print(output_concat.shape)
        # 通过输出的线性层
        return self.W_o(output_concat)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        # 初始化方法
        super(PositionalEncoding, self).__init__()  # 调用父类的初始化方法
        self.dropout = nn.Dropout(dropout)  # 创建一个Dropout层
        self.P = torch.zeros((1, max_len, num_hiddens))  # 初始化一个全0的位置编码矩阵，大小为(1, max_len, num_hiddens)
        # 计算位置编码
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 使用正弦函数为0,2,4,...维度的位置上填充位置编码
        self.P[:, :, 0::2] = torch.sin(X)
        # 使用余弦函数为1,3,5,...维度的位置上填充位置编码
        self.P[:, :, 1::2] = torch.cos(X)
        
    def forward(self, X):
        # 前向传播方法
        # 将位置编码加到输入X上
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # 应用Dropout并返回结果
        return self.dropout(X)


# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# 残差连接和层规范化(Add&Norm)
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class Encoder(nn.Module):
    """The base encoder interface for the encoder--decoder architecture.
    """
    # 类构造函数，初始化 Encoder 类的实例
    def __init__(self):
        # 调用父类 nn.Module 的构造函数
        super().__init__()

    # forward 函数定义了编码器的前向传播逻辑
    def forward(self, X, *args):
        # 这里 X 代表输入数据
        # *args 表示可以传入任意数量的位置参数
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self):
        # 初始化方法，在这里可以初始化解码器的参数和层
        super().__init__()

    # init_state方法用于初始化解码器的状态。
    # 这个方法应该在每个具体的解码器类中被实现（override）。
    # enc_all_outputs参数代表编码器的所有输出，可以用于初始化解码器的状态。
    # *args允许传入额外的参数，这提供了灵活性，以适应不同的解码器实现。
    def init_state(self, enc_all_outputs, *args):
        # NotImplementedError将被触发，因为这个方法需要在子类中具体实现。
        raise NotImplementedError

    # forward方法定义了解码器的前向传播逻辑。
    # X代表输入到解码器的序列数据。
    # state代表解码器的状态，它可能包含编码器的输出、隐藏状态等信息。
    # 这个方法也应该在每个具体的解码器类中被实现。
    def forward(self, X, state):
        # NotImplementedError将被触发，因为这个方法需要在子类中具体实现。
        raise NotImplementedError


class AttentionDecoder(Decoder):
    def __init__(self):
        # 初始化方法，可以添加针对注意力解码器特有的初始化代码
        super().__init__()

    # attention_weights属性用于获取解码器在处理序列时产生的注意力权重
    # 这个属性的具体实现应该在子类中提供
    @property
    def attention_weights(self):
        # NotImplementedError将被触发，因为具体的实现需要在子类中完成
        raise NotImplementedError
