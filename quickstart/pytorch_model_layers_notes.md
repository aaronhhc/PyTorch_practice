# PyTorch Model Layers 學習筆記

> 學習日期：2026年4月30日

---

## 1. 今日學習目標

今天主要學會如何用 PyTorch 的 `torch.nn` 模組建立簡單神經網路 layers，並理解資料經過每一層時 shape 如何改變。

這份程式碼的重點是：

| 主題 | 說明 |
|------|------|
| `nn.Flatten()` | 將圖片攤平成一維特徵向量 |
| `nn.Linear()` | 全連接層，將輸入特徵轉成輸出特徵 |
| `nn.ReLU()` | 加入非線性，讓模型能學習更複雜的關係 |
| `nn.Sequential()` | 把多個 layer 依序串起來 |
| `nn.Softmax()` | 將 logits 轉成每個類別的機率 |
| `argmax()` | 找出機率最高的類別 |

---

## 2. 匯入 torch 和 nn

```python
import torch
from torch import nn
```

| 匯入 | 用途 |
|------|------|
| `torch` | 建立 tensor、做 tensor 運算 |
| `torch.nn` | 建立 neural network layers |

`nn` 是 PyTorch 中設計神經網路最常用的模組，例如 `Linear`、`ReLU`、`Flatten`、`Softmax` 都在這裡。

---

## 3. 建立輸入圖片 Tensor

```python
input_image = torch.rand(3, 28, 28)
print("Input image shape:", input_image.shape)
```

輸出：

```text
Input image shape: torch.Size([3, 28, 28])
```

這裡建立的是隨機 tensor，數值介於 0 到 1 之間。

> 注意：在這個範例中，shape 是 `[3, 28, 28]`。因為後面使用預設的 `nn.Flatten()`，PyTorch 會把第一個維度 `3` 保留下來，當成 batch 維度，而不是 RGB channel。

---

## 4. nn.Flatten 是什麼

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print("After Flatten:", flat_image.shape)
```

`nn.Flatten()` 的預設行為是：

```python
nn.Flatten(start_dim=1, end_dim=-1)
```

也就是保留第 0 維，將第 1 維之後全部攤平。

| 原始 shape | Flatten 後 shape | 說明 |
|------------|------------------|------|
| `[3, 28, 28]` | `[3, 784]` | 保留第 0 維 `3`，將 `28 * 28` 攤平 |

```python
28 * 28 = 784
```

所以每一筆資料會變成 784 個 features。

---

## 5. nn.Linear 是什麼

```python
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print("After Linear 1:", hidden1.shape)
```

`nn.Linear` 是全連接層，也可以想成：

```text
輸入特徵 -> 線性轉換 -> 輸出特徵
```

公式概念：

```text
y = xW^T + b
```

| 參數 | 說明 |
|------|------|
| `in_features=28 * 28` | 每筆輸入有 784 個特徵 |
| `out_features=20` | 每筆輸出變成 20 個特徵 |

shape 變化：

| Layer | Shape |
|-------|-------|
| Flatten 後 | `[3, 784]` |
| Linear 1 後 | `[3, 20]` |

---

## 6. nn.ReLU 是什麼

```python
hidden1_relu = nn.ReLU()(hidden1)
print("After ReLU:", hidden1_relu.shape)
```

ReLU 是 activation function，作用是把負數變成 0，正數保留原值：

```text
ReLU(x) = max(0, x)
```

| 輸入 | ReLU 後 |
|------|---------|
| `-2` | `0` |
| `0` | `0` |
| `3` | `3` |

ReLU 不會改變 shape，只會改變 tensor 裡面的數值。

| ReLU 前 | ReLU 後 |
|---------|---------|
| `[3, 20]` | `[3, 20]` |

> 為什麼需要 ReLU？如果只有 Linear layer，模型只能學到線性關係。加入 ReLU 這類非線性函數後，模型才有能力學習更複雜的模式。

---

## 7. 第二層 Linear：輸出 10 個類別分數

```python
layer2 = nn.Linear(in_features=20, out_features=10)
logits = layer2(hidden1_relu)
print("Logits shape:", logits.shape)
```

這一層將 20 個 hidden features 轉成 10 個輸出分數。

| 參數 | 說明 |
|------|------|
| `in_features=20` | 接收上一層輸出的 20 個 features |
| `out_features=10` | 輸出 10 個類別分數 |

shape 變化：

| Layer | Shape |
|-------|-------|
| ReLU 後 | `[3, 20]` |
| Linear 2 後 | `[3, 10]` |

這裡的 `[3, 10]` 可以理解成：

| 維度 | 意義 |
|------|------|
| `3` | 3 筆資料 |
| `10` | 每筆資料有 10 個類別分數 |

---

## 8. logits 是什麼

```python
logits = layer2(hidden1_relu)
```

**logits** 是模型最後一層直接輸出的原始分數，還不是機率。

例如某一筆資料可能輸出：

```text
[0.2, -1.1, 2.4, 0.7, ...]
```

這些值：

| 特性 | 說明 |
|------|------|
| 可以是負數 | logits 沒有限制一定要大於 0 |
| 總和不一定是 1 | logits 還不是機率分布 |
| 數值越大 | 代表模型越偏向該類別 |

---

## 9. Softmax 是什麼

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

Softmax 會把 logits 轉成機率分布：

| Softmax 後特性 | 說明 |
|----------------|------|
| 每個值介於 0 到 1 | 可以當作機率理解 |
| 每一列總和為 1 | 代表所有類別機率加總為 100% |
| 最大值的位置 | 代表模型最可能預測的類別 |

為什麼是 `dim=1`？

```text
logits shape = [3, 10]
```

| dimension | 意義 |
|-----------|------|
| `dim=0` | 沿著不同資料筆數的方向算 |
| `dim=1` | 沿著 10 個類別的方向算 |

因為我們要對每一筆資料的 10 個類別分數轉成機率，所以使用 `dim=1`。

---

## 10. argmax 取得預測類別

```python
y_pred = pred_probab.argmax(1)
print("Predicted class:", y_pred)
```

`argmax(1)` 會沿著 `dim=1` 找最大值的位置。

如果某一筆資料的機率是：

```text
[0.03, 0.10, 0.71, 0.05, 0.11]
```

最大值是 `0.71`，位置是 index `2`，所以預測類別就是 `2`。

因為範例中有 3 筆資料，所以 `y_pred` 會有 3 個預測結果：

```text
Predicted class: tensor([..., ..., ...])
```

---

## 11. nn.Sequential 是什麼

目前的程式碼使用 `nn.Sequential` 將 layers 串起來：

```python
linear_seq = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10),
)
output = linear_seq(input_image)
```

`nn.Sequential` 的作用是把輸入依序送進每一層：

```text
input_image
    -> Flatten
    -> Linear(784 -> 20)
    -> ReLU
    -> Linear(20 -> 10)
    -> output logits
```

這和手動一層一層寫的結果是一樣的，只是更簡潔。

---

## 12. 整體 shape 流程

| 步驟 | 程式碼 | Shape |
|------|--------|-------|
| 輸入 | `input_image` | `[3, 28, 28]` |
| 攤平 | `nn.Flatten()` | `[3, 784]` |
| 第一層 Linear | `nn.Linear(784, 20)` | `[3, 20]` |
| ReLU | `nn.ReLU()` | `[3, 20]` |
| 第二層 Linear | `nn.Linear(20, 10)` | `[3, 10]` |
| Softmax | `nn.Softmax(dim=1)` | `[3, 10]` |
| Argmax | `argmax(1)` | `[3]` |

---

## 13. 目前程式碼的重要觀念

```python
output = linear_seq(input_image)
pred_probab = nn.Softmax(dim=1)(output)
y_pred = pred_probab.argmax(1)
```

| 程式碼 | 意義 |
|--------|------|
| `linear_seq(input_image)` | 將圖片 tensor 送進 Sequential 模型，得到 logits |
| `nn.Softmax(dim=1)(output)` | 將 logits 轉成每個類別的機率 |
| `pred_probab.argmax(1)` | 取得每筆資料機率最高的類別 |

---

## 14. 常見疑問

### Q1：為什麼 `in_features` 是 `28 * 28`，不是 `3 * 28 * 28`？

因為這份程式碼的輸入 shape 是 `[3, 28, 28]`，而預設 `nn.Flatten()` 會保留第 0 維，將後面兩個維度攤平成 `28 * 28 = 784`。

所以 Flatten 後是：

```text
[3, 784]
```

這時每筆資料的 features 是 784，因此 `in_features=28 * 28`。

### Q2：如果這個 `3` 是 RGB channel 呢？

如果要表示「一張 RGB 圖片」，比較常見的 shape 會是：

```text
[1, 3, 28, 28]
```

其中：

| 維度 | 意義 |
|------|------|
| `1` | batch size，一張圖片 |
| `3` | RGB channels |
| `28` | 高度 |
| `28` | 寬度 |

這時 Flatten 後會是：

```text
[1, 2352]
```

因為：

```python
3 * 28 * 28 = 2352
```

對應的 Linear layer 就要改成：

```python
nn.Linear(in_features=3 * 28 * 28, out_features=20)
```

### Q3：訓練時也要先 Softmax 嗎？

如果使用 `nn.CrossEntropyLoss()`，通常不需要先做 Softmax。

原因是 `CrossEntropyLoss` 會在內部處理 logits，因此訓練時通常會直接把 logits 丟進 loss function：

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

Softmax 比較常用在「想看預測機率」或「推論後解讀結果」時。

---

## 15. 完整範例程式碼

```python
import torch
from torch import nn

input_image = torch.rand(3, 28, 28)
print("Input image shape:", input_image.shape)

linear_seq = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10),
)
output = linear_seq(input_image)
pred_probab = nn.Softmax(dim=1)(output)

y_pred = pred_probab.argmax(1)
print("Predicted class:", y_pred)
```

---

## 16. 今日總結

- `nn.Flatten()`：將多維 tensor 攤平成 feature vector
- `nn.Linear()`：全連接層，改變 feature 數量
- `nn.ReLU()`：activation function，加入非線性能力
- `nn.Sequential()`：把多個 layers 依序組成模型
- `logits`：模型輸出的原始分數，還不是機率
- `nn.Softmax(dim=1)`：將每筆資料的類別分數轉成機率
- `argmax(1)`：找出每筆資料機率最高的類別
- 訓練分類模型時，如果使用 `CrossEntropyLoss`，通常直接使用 logits，不需要先 Softmax

---

> 筆記建立日期：2026年4月30日
> 主題：PyTorch Model Layers, Sequential, Linear, ReLU, Softmax
