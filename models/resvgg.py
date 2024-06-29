import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义转换操作
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载数据集
image_dataset = datasets.ImageFolder('E:/DeepLearningTutorial20240603/datasets/Potato Plant Diseases/PotatoPlants', train_transform)

# 划分数据集
split_size = int(0.8 * len(image_dataset))
train_dataset, val_dataset = random_split(image_dataset, [split_size, len(image_dataset) - split_size])

# 为测试集设置变换操作
val_dataset.dataset.transform = val_transform

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


class ResVGG(nn.Module):
    def __init__(self):
        super(ResVGG, self).__init__()
        self.features1 = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BN
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BN
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64),  # BN
        )
        
        self.features2 = nn.Sequential(
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BN
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BN
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),  # BN
        )
                
        self.features3 = nn.Sequential(
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BN
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BN
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BN
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),  # BN
        )
                
        self.features4 = nn.Sequential(
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512),  # BN
        )
                
        self.features5 = nn.Sequential(
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BN
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.shortcut5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512),  # BN
        )
                
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 3)  # 输出层，3分类任务
        )

    def forward(self, x):
        output1 = self.features1(x)
        x1 = self.shortcut1(x)
        output1 += x1
        
        output2 = self.features2(output1)
        x2 = self.shortcut2(output1)
        output2 += x2
        
        output3 = self.features3(output2)
        x3 = self.shortcut3(output2)
        output3 += x3
        
        output4 = self.features4(output3)
        x4 = self.shortcut4(output3)
        output4 += x4
        
        output5 = self.features5(output4)
        x5 = self.shortcut5(output4)
        output5 += x5
        
        output = torch.flatten(output5, 1)  # 展开特征图
        output = self.classifier(output)
        return output


model = ResVGG()
model.to(device)

# 定义超参数
learning_rate = 0.001
num_epochs = 5

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if __name__ == '__main__':
    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 打印每个批次的进度和损失
            if (i + 1) % 5 == 0:  # 每10个批次打印一次
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}")
            
        # 计算并打印平均损失
        avg_loss = total_loss / total_step
        print(f"Epoch [{epoch+1}/{num_epochs}] finished with avg loss: {avg_loss}")

    # 测试模型
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy: {(correct / total) * 100:.2f}%")
