import torch
import torch.nn.functional as F
device = torch.device('cuda')
B, C, H, W = 1, 3, 128, 128
x = torch.randn(B, C, H, W, device=device)
flow = torch.tanh(torch.randn(B, H, W, 2, device=device))    
y = F.grid_sample(x, flow)
print('ok')
