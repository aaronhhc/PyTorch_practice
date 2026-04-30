# PyTorch Practice Repository

A collection of PyTorch practice exercises covering fundamental concepts including Tensor operations, Custom Dataset implementation, DataLoader usage, and neural network layers.

## 📂 Project Structure

```
PyTorch_practice/
├── quickstart/
│   ├── tensor.py                  # Tensor operations tutorial
│   ├── custom_dataset.py          # Custom Dataset & DataLoader implementation
│   ├── fashion_mnist.py           # Fashion MNIST dataset example
│   ├── model_layers.py            # Neural network layers example
│   ├── pytorch_custom_dataset_notes.md  # Custom Dataset notes (Traditional Chinese)
│   ├── pytorch_model_layers_notes.md    # Model layers notes (Traditional Chinese)
│   ├── custom_data/               # Custom image dataset
│   │   ├── label.csv             # Image labels CSV file
│   │   └── images/               # Image directory
│   └── data/
│       └── FashionMNIST/         # Fashion MNIST raw data
└── .venv/                        # Python virtual environment
```

## 📖 Contents Overview

### tensor.py
Fundamental PyTorch Tensor operations:
- Tensor creation methods
- Basic arithmetic operations
- Shape manipulation

### custom_dataset.py
Custom Dataset and DataLoader implementation:
- Inherit from `Dataset` class
- Implement `__init__`, `__len__`, `__getitem__` methods
- Load CSV labels using pandas
- Image preprocessing (Resize, ToTensor, convert RGB)

### fashion_mnist.py
Fashion MNIST dataset examples:
- Load built-in dataset
- Data visualization

### model_layers.py
Neural network layers example:
- Build a simple model with `nn.Sequential`
- Use `nn.Flatten`, `nn.Linear`, and `nn.ReLU`
- Convert logits to probabilities with `nn.Softmax`
- Pick predicted classes with `argmax`

### pytorch_custom_dataset_notes.md
Learning notes in Traditional Chinese covering:
- Three required dunder methods for Custom Dataset
- labels.csv and iloc usage
- transform, Compose, Resize, ToTensor explanations
- Common errors and solutions

### pytorch_model_layers_notes.md
Learning notes in Traditional Chinese covering:
- Tensor shape flow through model layers
- Flatten, Linear, ReLU, Sequential, Softmax explanations
- Logits vs probabilities
- `argmax` prediction logic
- Common shape questions for batch and channel dimensions

## 🚀 Quick Start

```bash
# Navigate to project directory
cd quickstart

# Run custom_dataset.py
python custom_dataset.py

# Run tensor.py
python tensor.py

# Run fashion_mnist.py
python fashion_mnist.py

# Run model_layers.py
python model_layers.py
```

## 📋 Custom Dataset Format

To use your own dataset, follow this structure:

### label.csv
```csv
filename,label
img_0.png,0
img_1.png,1
img_2.png,0
```

### Directory Structure
```
custom_data/
├── label.csv
└── images/
    ├── img_0.png
    ├── img_1.png
    └── img_2.png
```

## 🛠 Requirements

- Python 3.8+
- PyTorch
- pandas
- torchvision
- PIL (Pillow)

## 📝 Documentation

For detailed learning notes, see:

- [pytorch_custom_dataset_notes.md](quickstart/pytorch_custom_dataset_notes.md)
- [pytorch_model_layers_notes.md](quickstart/pytorch_model_layers_notes.md)

---

> 🔖 Version: 1.0.0  
> 📅 Last Updated: April 30, 2026  
> ⏳ Status: In Progress
