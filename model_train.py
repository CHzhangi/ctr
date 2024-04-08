import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 数据创建
def generate_data(num_samples=100):
    #每个样本两个特征
    X = torch.randn(num_samples, 2)
    # 使用线性关系生成标签，加上一些随机噪声
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.1 * torch.randn(num_samples)
    return X, y.view(-1, 1)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义单层线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()
        self.h1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.h2=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=self.h1(x)
        x=self.relu(x)
        x=self.h2(x)
        return x
# 训练函数
def train(model, criterion, optimizer, X, y, epochs=1000):
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        # 计算损失
        loss = criterion(outputs, y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 可视化模型拟合效果
def plot_model_fit(X, y, model):
    plt.scatter(X[:, 0], y, label='True data')
    plt.scatter(X[:, 0], model(X).detach().numpy(), label='Model prediction', color='red')
    plt.legend()
    plt.show()

# 创建数据
X, y = generate_data()

# 转换为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 创建模型、损失函数和优化器
model=LinearModel()
#model = MLP(2,10,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
train(model, criterion, optimizer, X, y)

# 可视化拟合效果
plot_model_fit(X, y, model)
