import torch
import numpy as np
from torch import nn
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
x = torch.tensor(data) # x= torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

print(x.shape)
print(x.dtype)
print(x)

print(x[0])
print(x[0, 0])
print(x[0, 1])
print(x[2])
print(x[:, 0])
print(x[:, 1]) #pytorch會自動對齊  
print(x + 1)
print(x * 2)
print(x + x)
print(x.sum())
print(x.float().mean())


'''
int tensor：適合 label / index
float tensor：適合神經網路運算 / mean / gradient
'''


rand = torch.rand(2, 3) #rand = torch.rand(2, 3) 是用來生成一個 2 行 3 列的隨機浮點數張量，這些數字在 [0, 1) 的範圍內均勻分布
print(rand)
print("shape:", rand.shape)
print("dtype:", rand.dtype)
print("device:", rand.device)


a = torch.arange(12)
print(a)
print(a.shape)

b = a.reshape(3, 4)
print(b)
print(b.shape)

c = a.reshape(2, 2, 3)
print(c)
print(c.shape)
#total number of elements must remain the same when reshaping, otherwise it will raise an error

#Flattening a tensor means converting a multi-dimensional tensor into a one-dimensional tensor. This is often done before feeding the data into a fully connected layer in a neural network, as fully connected layers expect one-dimensional input.
batch = torch.rand(64, 1, 28, 28)

flatten = nn.Flatten()
flat_batch = flatten(batch)

print(batch.shape)
print(flat_batch.shape)

#Argmax 主要用於分類問題中，當我們有多個類別的預測結果時，我們可以使用 argmax 來找到預測概率最高的類別索引，這樣我們就可以知道模型預測的類別是什麼。
#對於accuracy來說，我們可以使用 argmax 來找到模型預測的類別索引，然後與真實的類別索引進行比較，計算出正確預測的數量，最後除以總的樣本數量來得到準確率。
pred = torch.tensor([
    [0.1, 0.2, 3.5],
    [2.0, 0.5, 0.1],
    [0.3, 4.1, 0.2],
    [0.1, 0.4, 2.2],
])

y = torch.tensor([2, 1, 1, 2])

pred_class = pred.argmax(1)
print(pred_class)

correct_mask = pred_class == y
print(correct_mask)

correct_count = correct_mask.type(torch.float).sum().item() #correct_mask.type(torch.float) 是用來把布林值轉成浮點數，這裡是把 True 轉成 1.0，False 轉成 0.0，然後用 sum() 來計算正確的數量，最後用 item() 來把 tensor 轉成 Python 的數字，這裡是把正確的數量轉成 Python 的數字
print(correct_count)

#Autograd 是 PyTorch 中的一個自動微分引擎，它可以自動計算張量的梯度，這對於訓練神經網絡非常有用。當我們創建一個張量並設置 requires_grad=True 時，PyTorch 會開始跟踪對該張量的所有操作，以便在需要時計算梯度。
x = torch.tensor(2.0, requires_grad=True)#requires_grad=True 是用來告訴 PyTorch 這個張量需要計算梯度，這樣在進行反向傳播時，PyTorch 就會自動計算這個張量的梯度，並將其存儲在 x.grad 中

y = x ** 2
y.backward() #backward() 是用來計算張量的梯度的函數，這裡是對 y 進行反向傳播，計算 x 的梯度，然後把梯度存儲在 x.grad 中，這樣我們就可以看到 x 的梯度是多少了

print(x.grad)

n = np.ones(5)
t = torch.from_numpy(n)

print(n)
print(t)

#Arithmetic operations

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)