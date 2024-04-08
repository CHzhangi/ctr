import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#softmax实际上是对线性回归的参数进行训练
from tqdm import tqdm
# 创建自定义数据集
x = np.array([[1, 2, 3,4],
              [4, 5, 6,7],
              [7, 8, 9,9],
              [10, 11, 12,11],
              [13, 14, 15,19]])
x=torch.tensor(x, dtype=torch.float32)
y = np.array([0, 0, 0, 1, 1])
y=torch.tensor(y, dtype=torch.long)

class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
        #probabilities=nn.functional.softmax(logits,dim=1)
        #return probabilities
        #因为交叉熵损失函数会自动对输入模型的预测值进行softmax。因此在多分类问题中，如果使用nn.CrossEntropyLoss()，则预测模型的输出层无需添加softmax层。
        #并且torch.nn.CrossEntropyLoss()接受两种形式的标签输入，一种是类别index，一种是one-hot形式，所以[0,0,1,2,3]就可以了
input_dim = 4  # 输入维度，MNIST图像大小为28x28，展平后为784
output_dim = 2  # 输出维度，MNIST数据集有10个类别
learning_rate = 0.1
batch_size = 64
num_epochs = 10
torch.set_printoptions(sci_mode=False)
net=SoftmaxRegression(input_dim,output_dim)
model = SoftmaxRegression(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
for i in range(1000):
    prediction=net(x)
    loss=criterion(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(nn.functional.softmax(net(x)))


