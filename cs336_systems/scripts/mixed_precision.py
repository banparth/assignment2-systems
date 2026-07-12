import torch
import torch.nn as nn
from cs336_basics.nn_utils import cross_entropy



class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        print(x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        return x    

def main():
    device = torch.device("cuda")
    model = ToyModel(10, 10).to(device=device)
    x = torch.empty(10, device=device)
    target = torch.empty(1, device=device)
    
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = model(x)
        print(y.dtype)
        print(y.shape)
        loss = cross_entropy(x, y)
        loss.backward()
        
        print(model.fc1.weight.grad.dtype)

if __name__ == "__main__":
    main()