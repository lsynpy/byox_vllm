import torch
import torch.nn as nn


# 1. Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))


# 2. Instantiate and compile the model
# We use the default "inductor" backend
model_fn = MyModel()
compiled_fn = torch.compile(model_fn)

# 3. Create dummy input and run the model to trigger compilation
inp = torch.randn(4, 10)
print("Running compiled model to trigger compilation and log IRs...")
compiled_fn(inp)
print("Run complete.")
