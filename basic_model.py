import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import csv
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns



if torch.cuda.is_available():  
    print("CUDA is available. Training will be on GPU.")  
else:  
    print("CUDA is not available. Training will be on CPU.")


datafile = 'housing.csv'
housing_data = np.fromfile(datafile, sep=' ')
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)
housing_data = housing_data.reshape([housing_data.shape[0] // feature_num, feature_num])
features_np = np.array([x[:13] for x in housing_data], np.float32)
labels_np = np.array([x[-1] for x in housing_data], np.float32)
features_np = np.array([x[:13] for x in housing_data], np.float32)
labels_np = np.array([x[-1] for x in housing_data], np.float32)
# data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(housing_data, columns=feature_names)
matplotlib.use('TkAgg')
sns.pairplot(df.dropna(), y_vars=feature_names[-1], x_vars=feature_names[::-1], diag_kind='kde')
plt.show()
fig, ax = plt.subplots(figsize=(15, 1)) 
corr_data = df.corr().iloc[-1]
corr_data = np.asarray(corr_data).reshape(1, 14)
ax = sns.heatmap(corr_data, cbar=True, annot=True)
plt.show()

sns.boxplot(data=df.iloc[:, 0:13])
plt.show()

features_max = housing_data.max(axis=0)
features_min = housing_data.min(axis=0)
features_avg = housing_data.sum(axis=0) / housing_data.shape[0]
#计算每一列的均值minmax也就是每个特征的

def feature_norm(input):
    f_size = input.shape
    output_features = np.zeros(f_size, np.float32)
    for batch_id in range(f_size[0]):
        for index in range(13):
            output_features[batch_id][index] = (input[batch_id][index] - features_avg[index]) / (features_max[index] - features_min[index])
    return output_features 

# 只对属性进行归一化
housing_features = feature_norm(housing_data[:, :13])
# print(feature_trian.shape)
housing_data = np.c_[housing_features, housing_data[:, -1]].astype(np.float32)
# print(training_data[0])

# 归一化后的train_data, 看下各属性的情况
features_np = np.array([x[:13] for x in housing_data],np.float32)
labels_np = np.array([x[-1] for x in housing_data],np.float32)
data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(data_np, columns=feature_names)
sns.boxplot(data=df.iloc[:, 0:13])

plt.show()

def draw_train_process(iters, train_costs):
    plt.title("training loss", fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.show()
    

class myDataset(Dataset):
    def __init__(self,csv_file):
        self.data=pd.read_csv(csv_file).iloc[:,0].str.split().apply(pd.to_numeric, errors='coerce')
        #.apply(pd.to_numeric, errors='coerce') 针对是str的
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        sample= self.data.iloc[index]
        features=torch.tensor(sample[0:-1], dtype=torch.float32)
        label=torch.tensor(sample[-1], dtype=torch.float32)
        return features,label
    
input_n=13
hidden_n=30
output_n=1


#model=nn.Sequential(
#    nn.Linear(input_n,1)
#    nn.Relu()
#)


class DNNnet(nn.Module):
    def __init__(self,input_n,hidden_n,output_n):
        super(DNNnet,self).__init__()
        self.hidden=nn.Linear(input_n,hidden_n)
        self.output=nn.Linear(hidden_n,output_n)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        return self.output(x)
criterion_DNN=nn.MSELoss()  
#loss=criterion_DNN(prediction,y)  

class LRnet(nn.Module):
    def __init__(self,input_n):
        super(LRnet,self).__init__()
        self.Linear=nn.Linear(input_n,1)

    def forward(self,x):
        x=F.relu(self.Linear(x))
        return self.sigmoid(x)
    
class SoftMaxNet(nn.Module):
    def __init__(self,in_n,hi_n,o_n):
        super(SoftMaxNet,self).__init__()
        self.hidden=nn.Linear(in_n,hi_n)
        self.output=nn.Linear(hi_n,o_n)
    
    def forward(self,X):
        X=F.relu(self.hidden(x,hidden_n))
        return self.output(x)
criterion_Soft_LR = nn.CrossEntropyLoss()
#loss=criterion_Soft_LR(prediction,y)

DNNnet=DNNnet(input_n,hidden_n,output_n)

optimizer=optim.Adam(DNNnet.parameters(),lr=0.01)
#优化器

data_dir='housing.csv'        
data_set=myDataset(data_dir)
#读取数据集

train_size = int(0.8 * len(data_set))
test_size = len(data_set) - train_size
train_set, test_set = torch.utils.data.random_split(data_set, [train_size, test_size])
#分割数据集为训练集和测试集

df = pd.read_csv(data_dir,header=None)
batch_size = 20
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# 创建训练集和测试集的数据加载器

n_epochs=10000
#迭代次数

train_num=0
train_nums=[]
losses=[]
#每次epoch都会遍历完一遍所有数据，一次train_loader会取batch_size样本
for epoch in range(n_epochs):
    for x,y in train_loader:
        prediction=DNNnet(x)
        loss=criterion_DNN(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_num+=1
        train_nums.append(train_num)
        losses.append(loss.item())
#    if epoch % 10==0:
#            print("NO", epoch, "loss is %.2f" % loss.item())
matplotlib.use('TkAgg')
#%matplotlib inline
draw_train_process(train_nums, losses)
            
with torch.no_grad():
    test_loss = 0
    correct = 0
    total = 0

    for x, y in test_loader:
        prediction = DNNnet(x)
        loss = criterion_DNN(prediction, y)
        test_loss += loss.item()

        _, predicted = torch.max(prediction.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    average_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100

    print("Test Loss: %.2f" % average_loss)
    print("Accuracy: %.2f%%" % accuracy)