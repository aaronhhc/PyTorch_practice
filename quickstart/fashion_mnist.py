import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor #torchvision 的一個 transform，作用是把圖片轉成 PyTorch 可以訓練的 Tensor
'''
tensor shape is very important in deep learning, it is the basis of all operations in deep learning, and it is also the basis of all operations in PyTorch.
'''
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(len(training_data))
print(len(test_data))

image, label = training_data[0]

print(image.shape)
print(label)
'''
60000
10000
torch.Size([1, 28, 28])
9
'''

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataLoader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
'''
Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
Shape of y:  torch.Size([64]) torch.int64
'''

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #super() 是用來呼叫父類別的建構子，這裡是呼叫 nn.Module 的建構子，讓我們的 NeuralNetwork 類別繼承 nn.Module 的功能
        self.flatten = nn.Flatten() #nn.Flatten() 是一個 PyTorch 的模組，用來把多維的輸入展平為一維，這裡是把 28x28 的圖片展平為 784 的向量
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) #len(dataloader) 是 dataloader 中的 batch 數量，這裡是 10000 / 64 = 156.25，因為最後一個 batch 可能不滿 64，所以實際上是 157 個 batch
    model.eval()
    test_loss, correct = 0, 0 #test_loss 是測試的平均損失，correct 是測試的正確率，這裡是用來計算測試的平均損失和正確率的變數
    with torch.no_grad(): #torch.no_grad() 是一個上下文管理器，用來禁止 PyTorch 計算梯度，這樣可以節省記憶體和計算時間，因為在測試階段我們不需要更新模型的參數
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() #item() 是用來把 tensor 轉成 Python 的數字，這裡是把損失值轉成 Python 的數字，然後累加到 test_loss 中
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #correct 是用來計算測試的正確率的變數，這裡是用 pred.argmax(1) 來得到模型預測的類別，然後和 y 來比較，如果相等就表示預測正確，然後用 type(torch.float) 來把布林值轉成浮點數，最後用 sum() 來計算正確的數量，然後用 item() 來把 tensor 轉成 Python 的數字，最後累加到 correct 中
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataLoader, model, loss_fn, optimizer)
    test(test_dataLoader, model, loss_fn)
print("Done!")