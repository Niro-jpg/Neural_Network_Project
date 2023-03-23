import torch

a = torch.tensor([[1,0,2],[2,3,4],[3,4,5]])
print(a)
print(torch.roll(a, 0, -1))
print(torch.roll(a, 1, -1))
print(torch.roll(a, 2, -1))
print(torch.roll(a, 3, -1))