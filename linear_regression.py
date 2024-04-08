import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
x=torch.linspace(-1,1,200)
x=torch.unsqueeze(x,dim=1)
y=x.pow(2)
dataset = TensorDataset(x,y)
data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
class net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        return self.predict(x)
loss=torch.nn.MSELoss()

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

net=net(n_feature=1,n_hidden=100,n_output=1)


optimizer=torch.optim.SGD(net.parameters(),lr=0.0001)
loss_func=torch.nn.MSELoss()
losses = []
for epoch in range(200):
    for x, y in data_loader:
        prediction=net(x)
        loss=loss_func(prediction,y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(loss.item())

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
print(net(torch.unsqueeze(torch.tensor([0.8]),dim=1)))