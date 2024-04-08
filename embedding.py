import torch
from torch import nn
#https://blog.csdn.net/lsb2002/article/details/132993128?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-132993128-blog-127134563.235^v39^pc_relevant_anti_vip&spm=1001.2101.3001.4242.1&utm_relevant_index=1
# 创建最大词个数为10，每个词用维度为4表示
embedding = nn.Embedding(10, 4)

# 将第一个句子填充0，与第二个句子长度对齐
in_vector = torch.LongTensor([[1, 2, 3, 4, 0, 0], [1, 2, 5, 6, 5, 7]])
inputs=[""]
out_emb = embedding(in_vector)
print(in_vector.shape)
print((out_emb.shape))
print(out_emb)
print(embedding.weight)