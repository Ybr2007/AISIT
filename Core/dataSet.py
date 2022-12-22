import os
import pickle

import torch

class PosDataSet(torch.utils.data.Dataset):
    def __init__(self, resultsPath = 'Data/Train/Processed/'):
        super().__init__()

        posDatas = pickle.load(open(os.path.join(resultsPath, 'posData.data'), 'rb'))
        negDatas = pickle.load(open(os.path.join(resultsPath, 'negData.data'), 'rb'))

        self.inputs = []
        self.targets = []

        for poseData in posDatas:
            poseData_ = []
            for posePointData in poseData:
                for pointValue in posePointData:
                    poseData_.append(pointValue)
            self.inputs.append(torch.Tensor(poseData_))
            self.targets.append(torch.Tensor([1]))

        for poseData in negDatas:
            poseData_ = []
            for posePointData in poseData:
                for pointValue in posePointData:
                    poseData_.append(pointValue)
            self.inputs.append(torch.Tensor(poseData_))
            self.targets.append(torch.Tensor([0]))

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.inputs)

class PosDataSetPro(torch.utils.data.Dataset):
    def __init__(self, resultsPath = 'Data/Train/Processed/'):
        super().__init__()

        posDatas = pickle.load(open(os.path.join(resultsPath, 'posData.data'), 'rb'))
        negDatas = pickle.load(open(os.path.join(resultsPath, 'negData.data'), 'rb'))

        self.inputs = []
        self.targets = []

        for poseData in posDatas:
            poseData_ = []
            min_, max_ = [100000] * 3, [-1] * 3
            for posePointData in poseData:
                for i, pointValue in enumerate(posePointData):
                    min_[i] = min(min_[i], pointValue)
                    max_[i] = max(max_[i], pointValue)
                    poseData_.append(pointValue)
            for i in range(len(poseData_)):
                poseData_[i] = (poseData_[i] - min_[i % 3]) / (max_[i % 3] - min_[i % 3])
            self.inputs.append(torch.Tensor(poseData_))
            self.targets.append(torch.Tensor([1]))

        for poseData in negDatas:
            poseData_ = []
            min_, max_ = [100000] * 3, [-1] * 3
            for posePointData in poseData:
                for i, pointValue in enumerate(posePointData):
                    min_[i] = min(min_[i], pointValue)
                    max_[i] = max(max_[i], pointValue)
                    poseData_.append(pointValue)
            for i in range(len(poseData_)):
                poseData_[i] = (poseData_[i] - min_[i % 3]) / (max_[i % 3] - min_[i % 3])
            self.inputs.append(torch.Tensor(poseData_))
            self.targets.append(torch.Tensor([0]))

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.inputs)

PosDataSet()