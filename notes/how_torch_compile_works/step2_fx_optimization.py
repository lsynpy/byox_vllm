import torch

# TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo,inductor" python step2_fx_optimization.py


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        w = self.linear.weight.mean(0)
        return torch.topk(torch.sum(self.linear(x + w).relu(), dim=-1), 3)


m = MyModule()
x = torch.randn(5, 4)

model_opt = torch.compile(m)
model_opt(x)
