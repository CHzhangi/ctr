import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


#仅初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#获得一个初始的隐藏层的值
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

#主要的内部函数，沿着序列len维度，step by step地得到最后的隐藏层输出
def rnn(inputs,state,params):
    #print("inputs:",inputs)
    #print("inputsshape:", inputs.shape)
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    output=[]
    for X in inputs:
        #（批量大小，词表大小）
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        output.append(Y)
    return torch.cat(output,dim=0),(H,)

class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(net, self).__init__()
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.randn(hidden_size))
        self.W_hq = nn.Parameter(torch.randn(output_size, hidden_size))
        self.b_q = nn.Parameter(torch.randn(output_size))

    def forward(self, inputs, state):
        W_xh, W_hh, b_h, W_hq, b_q = self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q
        H, = state
        output = []
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh.t()) + torch.mm(H, W_hh.t()) + b_h)
            Y = torch.mm(H, W_hq.t()) + b_q
            output.append(Y)
        return torch.cat(output, dim=0), (H,)

# 示例用法
input_size = 10
hidden_size = 20
output_size = 5

model = net(input_size, hidden_size, output_size)

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.num_hiddens=num_hiddens
        self.vocab_size=vocab_size
        self.params=get_params(self.vocab_size,num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=forward_fn
    #_call__ 是一个特殊方法（special method），用于使实例对象可以像函数一样被调用
    def __call__(self, X, state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32)
        #这里forward_fn是rnn,就是核心模型
        return self.forward_fn(X,state,self.params)
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        print("y_hat:",y_hat)
        print("y_hat.shape:",y_hat.shape)
        l = loss(y_hat, y.long()).mean()
        print("y:",y.long())
        print("y.shape:", y.long().shape)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期 这里他相当于模拟了一个循环，让RNN只接受时间维度为1的一个数
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

#X = torch.arange(10).reshape((2, 5))
# （批量大小，时间步数）
#seq_le
#F.one_hot(X.T, 28).shape
# （时间步数，批量大小，词表大小）批量的样本之间独立，要按照时间步数上看，这个的第一行，下个的第一行
num_hiddens=512
#只需要传入函数名get_params,init_rnn_state
net=RNNModelScratch(len(vocab),num_hiddens,d2l.try_gpu(),get_params,init_rnn_state,rnn)
#state=net.begin_state(X.shape[0],d2l.try_gpu())

num_epochs, lr = 1, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)

print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))
exit()

#训练的时候.比如输入是 traverller然后对应每一个位置都让他输出一个值,和真实标签比如raverllere求一个损失函数,训练参数
#预测的时候呢,比如要预测traverller后面的词,会先把traverller遍历完,但是不输出,用来更新隐藏值(比初始时的隐藏值好)
#然后再通过最后一个r输出的值,一步一步的循环生产pre_len个输出,作为预测值.



