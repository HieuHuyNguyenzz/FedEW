import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name: str) -> nn.Module:
    if model_name == "eMLP":
        return MLP()
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
