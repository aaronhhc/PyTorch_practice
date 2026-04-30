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

'''
flatten = nn.Flatten()
flat_image = flatten(input_image)
print("After Flatten:", flat_image.shape)

layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print("After Linear 1:", hidden1.shape)

hidden1_relu = nn.ReLU()(hidden1)
print("After ReLU:", hidden1_relu.shape)

layer2 = nn.Linear(in_features=20, out_features=10)
logits = layer2(hidden1_relu)
print("Logits shape:", logits.shape)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print("Probabilities shape:", pred_probab.shape)
print("First sample probabilities:", pred_probab[0])
print("First sample probability sum:", pred_probab[0].sum())
'''

y_pred = pred_probab.argmax(1)
print("Predicted class:", y_pred)