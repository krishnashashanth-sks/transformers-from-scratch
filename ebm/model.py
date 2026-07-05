import torch
import torch.nn as nn

class CNN_EnergyFunction(nn.Module):
  def __init__(self,image_channels,image_size):
    super().__init__()
    self.conv_layers=nn.Sequential(
        nn.Conv2d(image_channels,32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,padding=1),
        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1))
    )
    self._output_dim=128
    self.fc_layers=nn.Sequential(
        nn.Linear(self._output_dim,64),
        nn.ReLU(),
        nn.Linear(64,1)
    )
  def forward(self,x):
    x=self.conv_layers(x)
    x=torch.flatten(x,1)
    energy=self.fc_layers(x)
    return energy.squeeze(1)