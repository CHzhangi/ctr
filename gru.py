import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
value = pd.read_csv(r'dataset/A5M.txt', header=None)  # (14772, 1)
time = pd.date_range(start='200411190930', periods=len(value), freq='5min')
ts = pd.Series(value.iloc[:, 0].values, index=time)
ts_sample_h = ts.resample('H').sum()


# plt.plot(ts_sample_h)
# plt.xlabel("Time")
# plt.ylabel("traffic demand")
# plt.title("resample history traffic demand from 5M to H")
# plt.show()
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_us(B):
    dataset = ts_sample_h
    # split
    train = dataset[:int(len(dataset) * 0.7)]
    test = dataset[int(len(dataset) * 0.7):]
    m, n = np.max(train.values), np.min(train.values)

    # print(m,n)

    def process(data, batch_size, shuffle):
        load = data
        load = (load - n) / (m - n)
        seq = []
        for i in range(len(data) - 6):
            train_seq = []
            train_label = []
            for j in range(i, i + 6):
                x = [load[j]]
                train_seq.append(x)
            train_label.append(load[i + 6])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, False)
    Dte = process(test, B, False)

    return Dtr, Dte, m, n


class GRU(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 1
        self.num_directions = 1
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.gru(input_seq, (h_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred


Dtr, Dte, m, n = nn_seq_us(64)
hidden_size, num_layers = 10, 2
model = GRU(hidden_size, num_layers).to(device)
loss_function = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.5e-3)

# training
trainloss_list = []
model.train()
for epoch in tqdm(range(50)):
    train_loss = []
    for (seq, label) in Dtr:
        seq = seq.to(device)  # torch.Size([64, 80, 1])
        label = label.to(device)  # torch.Size([64, 1])
        y_pred = model(seq)
        loss = loss_function(y_pred, label)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainloss_list.append(np.mean(train_loss))
# training_loss的图
plt.plot(trainloss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("average of Training loss")
plt.show()

pred = []
y = []
model.eval()
for (seq, target) in Dte:
    seq = seq.to(device)
    target = target.to(device)
    y_pred = model(seq)
    pred.append(y_pred)
    y.append(target)

y = torch.cat(y, dim=0)
pred = torch.cat(pred, dim=0)
y = (m - n) * y + n
pred = (m - n) * pred + n  # torch.Size([179, 1])
print('MSE:', loss_function(y, pred))
# plot
plt.plot(y.cpu().detach().numpy(), label='ground-truth')
plt.plot(pred.cpu().detach().numpy(), label='prediction')
plt.xlabel("Time")
plt.ylabel("traffic demand")
plt.title("history traffic demand from 5M to H")
plt.show()