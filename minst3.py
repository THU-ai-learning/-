import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
# 检查是否有可用的GPU，若有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置seaborn的风格
sns.set_style('whitegrid')
# 定义数据预处理的转换操作
# 将图像数据转换为张量，并进行归一化处理，这里的均值和标准差是MNIST数据集的经验值
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),
                         (0.3081,))
])

# 下载并加载训练数据集
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           # 意味着数据加载器每次会取出 64 个样本数据
                                           shuffle=True,
                                           # 通过随机打乱数据，可以确保模型在每次训练周期中以不同的顺序处理数据，
                                           num_workers=4)
# 下载并加载测试数据集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层
        # 输入通道数为1（因为MNIST图像是单通道灰度图），输出通道数为32，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # 第一个卷积层：320 个参数

        # 第二个卷积层
        # 输入通道数为32（上一层卷积的输出通道数），输出通道数为64，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 第二个卷积层：18496 个参数
        # 第一个Dropout层，以0.25的概率随机将神经元置为0，防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        # 第二个Dropout层，以0.5的概率随机将神经元置为0，进一步防止过拟合
        self.dropout2 = nn.Dropout(0.5)
        # 第一个全连接层
        # 输入特征数量为经过前面卷积、池化等操作后展平的特征数量（这里计算得出是9216），输出特征数量为128
        self.fc1 = nn.Linear(9216, 128)  # 第一个全连接层：1179776 个参数
        # 第二个全连接层，将128个特征映射到10个类别（对应MNIST的10个数字类别）
        self.fc2 = nn.Linear(128, 10)  # 第二个全连接层：1290 个参数

    def forward(self, x):
        # 前向传播过程

        # 输入数据经过第一个卷积层
        x = self.conv1(x)
        # 使用ReLU激活函数增加非线性，使模型能够学习更复杂的模式
        x = nn.functional.relu(x)

        # 经过第二个卷积层
        x = self.conv2(x)
        # 再次使用ReLU激活函数
        x = nn.functional.relu(x)

        # 进行最大池化操作，池化核大小为2x2，步长默认与核大小相同，这样可以减小数据维度，提取主要特征
        x = nn.functional.max_pool2d(x, 2)

        # 应用第一个Dropout层，在训练过程中随机丢弃部分神经元，防止过拟合
        x = self.dropout1(x)

        # 将经过前面操作后的多维特征张量展平为一维向量，方便输入全连接层
        x = torch.flatten(x, 1)

        # 经过第一个全连接层
        x = self.fc1(x)
        # 使用ReLU激活函数
        x = nn.functional.relu(x)

        # 应用第二个Dropout层
        x = self.dropout2(x)

        # 经过第二个全连接层，输出最终的预测结果（10个类别对应的得分）
        x = self.fc2(x)

        return x


# 创建模型实例并移动到GPU（如果可用），这样模型的计算会在GPU上加速执行（如果使用GPU）
model = Net().to(device)

# 定义损失函数，这里使用交叉熵损失函数，常用于多分类问题，衡量预测结果与真实标签之间的差异
criterion = nn.CrossEntropyLoss()

# 定义优化器，使用Adam优化器，它自适应地调整学习率，传入模型的可学习参数以及学习率参数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练的轮数，即整个训练数据集完整遍历的次数
epochs = 15

# 用于记录训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

# 用于早停机制，记录验证集（这里用测试集代替）准确率不再提升的轮次数
patience = 5
counter = 0
# 记录最优准确率
best_accuracy = 0.0
# 保存模型的路径
model_path = "best_model.pth"
# 刘白123-150//训练过程
# 开始训练循环
for epoch in range(epochs):
    # 将模型设置为训练模式，启用Dropout等只在训练时起作用的操作
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和目标标签移动到指定的设备（GPU或CPU）上
        data, target = data.to(device), target.to(device)
        # 梯度清零，避免上一次迭代的梯度影响本次迭代
        optimizer.zero_grad()
        # 前向传播，通过模型得到预测输出
        output = model(data)
        # 计算损失，比较预测输出和真实目标标签之间的差异
        loss = criterion(output, target)
        # 反向传播，计算梯度，根据损失函数自动计算每个可学习参数的梯度
        loss.backward()
        # 根据计算得到的梯度，使用优化器更新模型的参数
        optimizer.step()
        # 累加当前批次的损失
        running_loss += loss.item()
        # 获取预测结果中概率最大的类别索引，作为预测的类别
        _, predicted = torch.max(output.data, 1)
        # 统计当前批次中样本的总数
        total += target.size(0)
        # 统计当前批次中预测正确的样本数量
        correct += (predicted == target).sum().item()
    # 计算平均损失，除以训练数据加载器中的批次数量
    epoch_loss = running_loss / len(train_loader)
    # 计算训练准确率，用预测正确的样本数量除以总样本数量，并转换为百分比形式
    epoch_accuracy = 100. * correct / total
    # 将当前轮次的损失添加到记录列表中
    train_losses.append(epoch_loss)
    # 将当前轮次的训练准确率添加到记录列表中
    train_accuracies.append(epoch_accuracy)
    print(
        f'轮次 {epoch + 1}/{epochs}，损失: {epoch_loss:.4f}，训练准确率: {epoch_accuracy:.2f}%')

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_accuracy = 100. * correct / total
    test_accuracies.append(test_accuracy)

    # 判断是否为目前最优准确率，若是则保存模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), model_path)
        counter = 0
    else:
        counter += 1
    # 早停机制判断，如果验证集准确率在一定轮次内不再提升，则停止训练
    if counter >= patience:
        print(f"验证集准确率在 {patience} 轮内未提升，提前停止训练。")
        break

checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
# 余学风191-213
# 绘制训练损失曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.title('训练损失随轮次变化情况')
plt.legend()
plt.show()

# 绘制训练准确率和测试准确率曲线
plt.plot(range(1, len(train_accuracies) + 1),
         train_accuracies,
         label='训练准确率')
plt.plot(range(1, len(test_accuracies) + 1),
         test_accuracies,
         label='测试准确率')
plt.xlabel('轮次')
plt.ylabel('准确率 (%)')
plt.title('准确率随轮次变化情况')
plt.legend()
plt.show()
# 混淆矩阵李丹214-233
# 计算并绘制混淆矩阵
all_predicted = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

conf_matrix = confusion_matrix(all_targets, all_predicted)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.show()

# 设置图像网格布局的行数和列数（这里设置为4x4，可以根据需要调整）
grid_rows = 4
grid_cols = 4

# 创建对应的子图布局
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
# 谭笑240往后
# 遍历每个子图位置，展示图像
for row in range(grid_rows):
    for col in range(grid_cols):
        index = row * grid_cols + col
        ax = axes[row][col]
        images, labels = next(iter(train_loader))  # 重新获取一批次数据，可获取更多不同图像
        image = images[index].squeeze().numpy()
        sns.heatmap(image, cmap='gray', ax=ax, cbar=False,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f'标签: {labels[index].item()}')
plt.suptitle("MNIST训练数据集示例图像")
plt.show()
# 随机选取一些测试集图像进行可视化展示预测结果
num_images = 10
images, labels = next(iter(test_loader))
outputs = model(images[:num_images])
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(10, 5))
for i in range(num_images):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(images[i][0].numpy(), cmap='gray')
    ax.set_title(f'Predicted: {predicted[i].item()}')
    ax.axis('off')
plt.show()
print("Current working directory:", os.getcwd())
