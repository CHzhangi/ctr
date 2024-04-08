import torch.nn.functional as F
import numpy as np
import torch.nn as nn
class FNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(FNN,self).__init__()
        self.hidden=nn.linear(input_size,hidden_size)
        self.output=nn.linear(hidden_size, output_size)
        
    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))
input_size = 5;
hidden_size =10;
output_size =5;
model=fnn(input_size,hidden_size,output_size)
criterion = nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(outputs,target)
    loss.backward()
    optimizer.step()