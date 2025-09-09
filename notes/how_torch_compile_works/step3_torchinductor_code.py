import torch
import torch._dynamo as dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        w = self.linear.weight.mean(0)
        return torch.topk(torch.sum(self.linear(x + w).relu(), dim=-1), 3)

module = MyModule()
example_input = torch.randn(5, 4)

explain_output = dynamo.explain(module)(example_input)

print(explain_output)