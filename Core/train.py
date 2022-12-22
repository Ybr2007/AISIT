import torch
import torch.nn as nn
import torch.optim as optim
from Core.module import PoseNet
from Core.dataSet import PosDataSet, PosDataSetPro
import matplotlib.pyplot as plt

dataset = PosDataSetPro()
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True)

net = PoseNet()

lossFuncation = nn.BCELoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.0008)
# optimizer = optim.SGD(net.parameters(), lr=0.003, weight_decay=0.0001)

def _train(epochNum, modulePath):
    lossList = []
    net.train()
    for epoch in range(epochNum):
        for num,(data,tag) in enumerate(dataloader):
            epochLoss = []
            output = net(data)
            loss = lossFuncation(output,tag)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochLoss.append(loss.item())
        if epoch % 100 == 0:
            lossValue = min(epochLoss)
            print(lossValue)
            lossList.append(lossValue)
            torch.save(net, modulePath)
    plt.plot(lossList)
    plt.show()

def train(epochNum: int, modulePath: str, loadModule: bool):
    global net
    if loadModule:
        net = torch.load(modulePath)
    _train(epochNum, modulePath)