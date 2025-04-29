import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from opensarwake_dataset import OpenSARWakeDataset
import pickle  # <-- 加上这个保存用

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 数据集和数据加载器
dataset = OpenSARWakeDataset(
    image_dir=r'C:\Users\astro\Desktop\OpenSARWake_1.0\train\images',
    label_dir=r'C:\Users\astro\Desktop\OpenSARWake_1.0\train\labels',
    transform=transform
)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型
class SimpleOBBDetector(nn.Module):
    def __init__(self):
        super(SimpleOBBDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 6)

    def forward(self, x):
        return self.backbone(x)

model = SimpleOBBDetector().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
loss_list = []

for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels.view(labels.size(0), -1))  # 标签reshape
    loss.backward()
    optimizer.step()

    loss_value = loss.item()
    print(f"Loss: {loss_value}")
    loss_list.append(loss_value)

    if i > 10:  # 只训练10步
        break

# 保存 loss_list
with open('loss_list.pkl', 'wb') as f:
    pickle.dump(loss_list, f)

print("✅ 训练完成，loss_list.pkl 文件已保存！")

