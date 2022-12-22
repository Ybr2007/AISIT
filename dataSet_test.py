import torch

from Core import PosDataSet, PosDataSetPro
from Core import PoseNet

dataset = PosDataSetPro('Data/Train/Processed/')

net = PoseNet()
net = torch.load('./Model/net.pth')
net.eval()

for img, target in dataset:
    output = net(img).item()
    if abs(target.item() - output) > 0.3:
        print("Bad!")
    print(output, target.item())