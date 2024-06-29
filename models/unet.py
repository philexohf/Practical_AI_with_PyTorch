import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    UNet网络类，继承自PyTorch的nn.Module。
    构造函数初始化网络结构，包括编码器和解码器部分。
    
    参数:
    in_channels (int): 输入图像的通道数，默认为1。
    num_classes (int): 分割任务的类别数，默认为1，适用于二分类问题。
    bilinear (bool): 是否使用双线性插值进行上采样，默认为True。如果为False，则使用转置卷积。
    """

    def __init__(self, in_channels=1, num_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)  # 输入层，对输入进行双卷积操作

        # 编码器部分：逐步下采样，同时增加通道数
        self.down1 = Down(64, 128)  # 第1层下采样
        self.down2 = Down(128, 256)  # 第2层下采样
        self.down3 = Down(256, 512)  # 第3层下采样
        factor = 2 if bilinear else 1  # 根据是否使用双线性插值确定因子
        self.down4 = Down(512, 1024 // factor)  # 第4层下采样，考虑上采样时的通道数

        # 解码器部分：逐步上采样，同时减少通道数
        self.up1 = Up(1024, 512 // factor, bilinear)  # 第1层上采样
        self.up2 = Up(512, 256 // factor, bilinear)  # 第2层上采样
        self.up3 = Up(256, 128 // factor, bilinear)  # 第3层上采样
        self.up4 = Up(128, 64, bilinear)  # 第4层上采样
        self.out = OutConv(64, num_classes)  # 输出层，生成最终的分割图

    def forward(self, x):
        """
        前向传播函数，定义了数据通过网络的流程。
        
        参数:
        x (Tensor): 输入的张量，即待分割的图像。
        
        返回:
        logits (Tensor): 分割任务的输出，即每个像素点对应每个类别的分数。
        """
        # 编码器部分：逐步下采样
        x1 = self.inc(x)  # 输入层
        x2 = self.down1(x1)  # 第1层下采样
        x3 = self.down2(x2)  # 第2层下采样
        x4 = self.down3(x3)  # 第3层下采样
        x5 = self.down4(x4)  # 第4层下采样

        # 解码器部分：逐步上采样，并与编码器的特征图进行拼接
        x = self.up1(x5, x4)  # 第1层上采样
        x = self.up2(x, x3)  # 第2层上采样
        x = self.up3(x, x2)  # 第3层上采样
        x = self.up4(x, x1)  # 第4层上采样

        # 输出层：生成最终的分割图
        logits = self.out(x)

        return logits  # 返回分割图的预测结果


class DoubleConv(nn.Module):
    """
    定义一个具有两次卷积的模块，每次卷积后跟一个实例归一化层和ReLU激活函数。
    这种结构通常用于UNet网络中以增强特征提取能力。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        构造函数初始化DoubleConv模块。
        
        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数。
        mid_channels (int, optional): 中间特征图的通道数。如果未指定，默认与out_channels相同。
        """
        super().__init__()
        
        # 如果没有指定mid_channels，则将其设置为out_channels
        if mid_channels is None:
            mid_channels = out_channels
        
        # 构建具有两次卷积的序列模型
        self.double_conv = nn.Sequential(
            # 第一次卷积：输入通道为in_channels，输出通道为mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 实例归一化层，归一化维度为mid_channels
            nn.InstanceNorm2d(mid_channels),
            # ReLU激活函数，inplace=True表示在原地进行操作，减少内存使用
            nn.ReLU(inplace=True),
            # 第二次卷积：输入通道为mid_channels，输出通道为out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 实例归一化层，归一化维度为out_channels
            nn.InstanceNorm2d(out_channels),
            # 另一个ReLU激活函数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数，将输入x通过double_conv序列模型进行处理。
        
        参数:
        x (Tensor): 输入的张量，即特征图。
        
        返回:
        Tensor: 经过两次卷积和两次激活后的输出张量。
        """
        return self.double_conv(x)  # 将输入x通过double_conv序列模型进行处理并返回结果


class Down(nn.Module):
    """
    定义一个下采样模块，首先应用最大池化层，然后应用DoubleConv模块。
    这种结构通常用于UNet网络中的编码器部分，用于逐步降低特征图的空间维度，同时增加通道数。
    """

    def __init__(self, in_channels, out_channels):
        """
        构造函数初始化Down模块。
        
        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 经过下采样操作后输出特征图的通道数。
        """
        super().__init__()

        # 构建下采样序列模型，首先应用最大池化层，然后是DoubleConv模块
        self.maxpool_conv = nn.Sequential(
            # 最大池化层，kernel_size=2，用于将特征图的高度和宽度减半
            nn.MaxPool2d(kernel_size=2),
            # DoubleConv模块，用于进一步提取特征，输入通道为in_channels，输出通道为out_channels
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        前向传播函数，将输入x通过maxpool_conv序列模型进行处理。
        
        参数:
        x (Tensor): 输入的张量，即特征图。
        
        返回:
        Tensor: 经过下采样和DoubleConv处理后的输出张量。
        """
        return self.maxpool_conv(x)  # 将输入x通过maxpool_conv序列模型进行处理并返回结果


class Up(nn.Module):
    """
    定义一个上采样模块，首先应用上采样操作，然后应用DoubleConv模块。
    这种结构通常用于UNet网络中的解码器部分，用于逐步增加特征图的空间维度，同时减少通道数。
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        构造函数初始化Up模块。
        
        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 经过上采样操作后输出特征图的通道数。
        bilinear (bool): 是否使用双线性插值进行上采样，默认为True。如果为False，则使用转置卷积。
        """
        super().__init__()

        # 根据bilinear参数选择上采样方法
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # DoubleConv模块，输入通道为in_channels，输出通道为out_channels，中间通道为in_channels//2
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            # DoubleConv模块，输入通道为in_channels，输出通道为out_channels
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        前向传播函数，将输入的特征图x1上采样并与x2进行拼接，然后通过DoubleConv模块。
        
        参数:
        x1 (Tensor): 来自解码器的上采样特征图。
        x2 (Tensor): 来自编码器对应层的特征图，用于与上采样的特征图拼接。
        
        返回:
        Tensor: 经过上采样、拼接和DoubleConv处理后的输出张量。
        """
        x1 = self.up(x1)  # 将x1上采样

        # 计算需要填充的尺寸以匹配x2的尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行填充以匹配x2的尺寸
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 沿着通道维度将x2和填充后的x1进行拼接
        x = torch.cat((x2, x1), dim=1)

        # 将拼接后的特征图通过DoubleConv模块
        return self.conv(x)


class OutConv(nn.Module):
    """
    定义一个输出卷积层模块，通常用于网络的最后，将特征图转换为最终的预测结果。
    这个模块只包含一个一维卷积层（kernel_size=1），用于在通道维度上进行转换。
    """

    def __init__(self, in_channels, out_channels):
        """
        构造函数初始化OutConv模块。
        
        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数，通常是类别数。
        """
        super(OutConv, self).__init__()

        # 构建一个一维卷积层，用于将输入特征图的通道数从in_channels转换为out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        前向传播函数，将输入x通过一维卷积层进行处理。
        
        参数:
        x (Tensor): 输入的张量，即特征图。
        
        返回:
        Tensor: 经过一维卷积层处理后的输出张量，通常表示每个像素点对应每个类别的预测结果。
        """
        return self.conv(x)  # 将输入x通过一维卷积层进行处理并返回结果
