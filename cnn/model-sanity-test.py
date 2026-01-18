import torch
from model import CraterRefiner

model = CraterRefiner()
x = torch.randn(4, 1, 96, 96)
y = model(x)

print(y.shape)
