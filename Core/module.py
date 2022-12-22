import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(18,48)
        self.dropout = nn.Dropout()
        self.l2 = nn.Linear(48,32)
        self.l3 = nn.Linear(32,1)
        
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x