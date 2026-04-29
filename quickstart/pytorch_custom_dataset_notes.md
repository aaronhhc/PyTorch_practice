# PyTorch Custom Dataset & DataLoader 學習筆記

> 學習日期：2026年4月29日

---

## 1. 今日學習目標

今天主要學會如何從自己的圖片資料夾與 `labels.csv` 建立 **Custom Dataset**，並用 **DataLoader** 產生 batch。這是 PyTorch 中處理自訂資料的核心技能。

---

## 2. Dataset 是什麼

Dataset 在 PyTorch 中的角色：

| 功能 | 說明 |
|------|------|
| 定義資料在哪裡 | 指定圖片路徑、CSV 檔案位置 |
| 總共有幾筆資料 | 實作 `__len__` 回傳資料數量 |
| 如何根據 index 拿出一筆資料 | 實作 `__getitem__(idx)` 回傳單筆資料 |

簡單來說，Dataset 就是「資料的存取介面」，告訴 PyTorch 怎麼讀取每一筆資料。

---

## 3. Custom Dataset 必備三個 dunder methods

Python 的 **dunder methods**（雙底線方法）是 class 與 Python 內建功能之間的特殊接口：

| 方法 | 何時被呼叫 | 範例 |
|------|------------|------|
| `__init__` | 建立物件時 | `dataset = CustomImageDataset(...)` |
| `__len__` | 呼叫 `len()` 時 | `len(dataset)` → 呼叫 `__len__` |
| `__getitem__` | 使用索引取值時 | `dataset[0]` → 呼叫 `__getitem__(0)` |

```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        # 初始化：設定檔案路徑、載入 CSV
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        # 回傳資料總數量
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # 根據 index 回傳單筆資料 (圖片, label)
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return image, label
```

---

## 4. labels.csv 與 img_dir

- **labels.csv**：儲存圖片檔名與對應的 label
- **img_dir**：圖片資料夾的路徑

```python
self.img_labels = pd.read_csv(annotations_file)
```

> 注意：`pd.read_csv` 只會讀取 CSV 檔案內容（檔名、label），**不會**先把所有圖片讀進記憶體。這樣可以節省記憶體，圖片是等到 `__getitem__` 被呼叫時才會用 `Image.open` 讀取。

---

## 5. iloc 是什麼

`iloc` 是 pandas 的索引方式，用 `,` 分隔 row 和 column：

```csv
filename,label
img_0.png,0
img_1.png,1
img_2.png,0
```

| 程式碼 | 意義 |
|--------|------|
| `self.img_labels.iloc[idx, 0]` | 取得第 idx 行的第 0 個欄位 → 圖片檔名 |
| `self.img_labels.iloc[idx, 1]` | 取得第 idx 行的第 1 個欄位 → label |

```python
img_name = self.img_labels.iloc[idx, 0]  # "img_0.png"
label = self.img_labels.iloc[idx, 1]     # 0
```

---

## 6. os.path.join

組路徑時應該用 `os.path.join`，而不是直接用字串相加：

```python
# ✅ 正確：跨平台相容
img_path = os.path.join(self.img_dir, img_name)

# ❌ 錯誤：可能出錯
img_path = self.img_dir + img_name
```

原因：
- 不同作業系統的路徑分隔符號不同（Windows 是 `\`，Linux/Mac 是 `/`）
- `os.path.join` 會自動處理，確保路徑正確

---

## 7. transform 是什麼

**transform** 是資料前處理的流程（pipeline），用來對圖片做各種處理：

```python
self.transform = transform  # 只是把轉換方法存起來
image = self.transform(image)  # 這行才是真的執行轉換
```

```python
if self.transform:
    image = self.transform(image)
```

> 為什麼要 `if self.transform`？因為當 `transform=None` 時，直接執行會報錯。加上判斷可以避免錯誤。

---

## 8. Compose, Resize, ToTensor

```python
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])
```

| 轉換 | 說明 |
|------|------|
| `Compose` | 把多個轉換串接起來，依序執行 |
| `Resize((224, 224))` | 將圖片統一調整成 224×224 大小 |
| `ToTensor()` | 將 PIL Image 轉成 PyTorch Tensor，並將像素值從 0~255 轉成 0~1 |

---

## 9. RGB 與 convert("RGB")

- **RGB**：Red, Green, Blue 三個顏色通道
  - RGB 圖片 shape：`[3, H, W]`（3 個 channel）
  - 灰階圖片 shape：`[1, H, W]`（1 個 channel）

```python
image = Image.open(img_path).convert("RGB")
```

`convert("RGB")` 的作用：
- 將灰階圖片轉成 RGB（複製 1 個 channel 到 3 個）
- 確保所有圖片都是 3 channels

> 為什麼要這樣？因為 DataLoader 組 batch 時，所有圖片的 shape 必須一致，否則會報錯。

---

## 10. DataLoader 是什麼

DataLoader 的角色是：

| 功能 | 說明 |
|------|------|
| 從 Dataset 取資料 | 一筆一筆呼叫 `__getitem__` |
| 組成 batch | 把多筆資料包成一個 batch |
| shuffle | 可以打亂資料順序 |
| batch_size | 設定每個 batch 有多少筆資料 |

```python
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

---

## 11. batch_size 與 shuffle

```python
batch_size = 2
shuffle = True
```

- **batch_size=2**：每個 batch 取 2 筆資料
- **shuffle=True**：打亂資料順序

> 範例：資料有 3 筆，batch_size=2，會分成 2 筆 + 1 筆（最後一個 batch 可能較小）

```python
Batch y: tensor([1, 0])  # 沒照 CSV 順序是正常的，因為 shuffle=True
```

---

## 12. 今天遇到的錯誤與解法

### 錯誤訊息

```
RuntimeError: stack expects each tensor to be equal size
```

### 原因

| 問題 | 說明 |
|------|------|
| 圖片大小不同 | 例如 [1, 239, 211] 和 [3, 175, 200] |
| channel 不同 | 灰階是 [1, H, W]，RGB 是 [3, H, W] |
| 結果 | DataLoader 無法將它們 stack 成一個 batch |

### 解法

```python
# 1. 統一 channel：灰階轉 RGB
image = Image.open(img_path).convert("RGB")

# 2. 統一大小：Resize
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# 3. 結果：所有圖片都變成 [3, 224, 224]
```

---

## 13. 最終成功輸出

執行後的輸出：

```
Dataset size: 3
Image shape: torch.Size([3, 224, 224])
Label: 0
Batch X shape: torch.Size([2, 3, 224, 224])
Batch y shape: torch.Size([2])
Batch y: tensor([1, 0])
```

| 輸出 | 意義 |
|------|------|
| `Dataset size: 3` | 總共有 3 張圖片 |
| `Image shape: torch.Size([3, 224, 224])` | 圖片是 3 channels、224×224 大小 |
| `Batch X shape: torch.Size([2, 3, 224, 224])` | batch 有 2 張圖片，每張 [3, 224, 224] |
| `Batch y shape: torch.Size([2])` | batch 有 2 個 label |
| `Batch y: tensor([1, 0])` | 這批的 label 是 1 和 0（順序被 shuffle 過） |

---

## 14. 完整範例程式碼

```python
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 建立 transform
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# 建立 Dataset
dataset = CustomImageDataset(
    annotations_file="custom_data/label.csv",
    img_dir="custom_data/images/",
    transform=transform
)

# 印出單筆資料
print("Dataset size:", len(dataset))
image, label = dataset[0]
print("Image shape:", image.shape)
print("Label:", label)

# 建立 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# 印出 batch
for X, y in dataloader:
    print("Batch X shape:", X.shape)
    print("Batch y shape:", y.shape)
    print("Batch y:", y)
    break
```

---

## 15. 今日總結

- ✅ **Dataset**：定義怎麼拿一筆資料（`__init__`、`__len__`、`__getitem__`）
- ✅ **DataLoader**：負責組 batch、shuffle、設定 batch_size
- ✅ **transform**：資料前處理流程
- ✅ **Resize**：統一圖片大小
- ✅ **ToTensor**：將 PIL Image 轉成 PyTorch Tensor，像素值轉成 0~1
- ✅ **convert("RGB")**：統一 channel 數量（灰階轉 RGB）
- ✅ **shuffle=True**：打亂資料順序
- ✅ **batch 裡所有圖片 shape 必須一致**，否則會報錯

---

> 📝 筆記建立日期：2026年4月29日
> 🎯 主題：PyTorch Custom Dataset & DataLoader