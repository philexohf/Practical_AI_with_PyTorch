import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

df = pd.read_csv("./datasets/strawberry/strawberry_clf.csv")
print(df.head())

# 提取特征和标签，特征列是'SSC', 'TA', 'Esters'，标签列是'premium'
x_data = df[['SSC', 'TA', 'Esters']].values
y_data = df['premium'].values.reshape(-1, 1)  # 将标签转换为适合模型的格式

print(x_data.shape)
print(y_data.shape)

# 将特征和标签转换为PyTorch张量，用于模型训练和测试
X = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32)

# 创建TensorDataset，它是一个包含了特征和标签的数据集
dataset = TensorDataset(X, y)

# 使用random_split函数随机划分数据集为训练集和测试集
# 这里我们按照80%的数据作为训练集，20%的数据作为测试集
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# 创建训练集和测试集的数据加载器，用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取输入特征的维度
input_size = x_data.shape[1]
# 设置训练的轮数
num_epochs = 6000

# 构建逻辑回归模型，使用线性层后接Sigmoid激活函数
model = nn.Sequential(
    nn.Linear(input_size, 1),  # 线性层，将输入特征映射到一个输出
    nn.Sigmoid()  # Sigmoid激活函数，用于二分类问题
)

print(model)

# 定义二元交叉熵损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降

# 训练模型的循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 初始化总损失累加器
    
    # 遍历训练数据加载器中的数据
    for data, labels in train_loader:
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(data.view(-1, input_size))  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        
        total_loss += loss.item()  # 累加损失值

    # 计算并打印平均损失
    avg_loss = total_loss / len(train_loader)
    if (epoch+1) % 500 == 0 or epoch == num_epochs - 1:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.5f}')

# 测试模型的准确性
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算
    correct = 0
    total = 0
    
    # 遍历测试数据加载器中的数据
    for data, labels in test_loader:
        outputs = model(data.view(-1, input_size))  # 前向传播
        predicted = (outputs > 0.5).float()  # 应用阈值0.5进行分类
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确预测的数量

    # 打印测试集上的准确率
    print(f'Accuracy: {100 * correct / total:.2f}%')
