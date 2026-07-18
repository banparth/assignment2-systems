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
        print(f"feedforward1 output: {x.dtype}")
        x = self.relu(x)
        print(f"relu output: {x.dtype}")
        x = self.ln(x)
        print(f"layernorm output: {x.dtype}")
        x = self.fc2(x)
        print(f"feedforward2 output: {x.dtype}")

        return x    

def main():
    device = torch.device("cuda")
    
    # def fun():
    #     y = model(x)
    #     print(f"y typ + {y.dtype}")
    #     print(f"y shape: {y.shape}")
    #     loss = cross_entropy(y, target)
    #     print(f"target unsqueeze shape: {target.unsqueeze(-1).shape}")
    #     loss.backward()
        
    # fun()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        
        model = ToyModel(10, 10).to(device=device)
        print(f"model feedforward: {model.fc1.weight.dtype}")
        print(f"layer norm: {model.ln.weight.dtype}")
        print(f"layer norm 2: {model.ln.bias.dtype}")

        x = torch.empty(1, 10, device=device)
        print(f"x: {x.dtype}")
        target = torch.empty(1, device=device, dtype=torch.int32)

        y = model(x)
        print(f"y : {y.dtype}")
        loss = cross_entropy(y, target)
        loss.backward()
        
        print(f"model gradient: {model.fc1.weight.grad.dtype}")
        print(f"loss.dtype: {loss.dtype}")

if __name__ == "__main__":
    main()