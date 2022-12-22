import os
import pickle

import cv2
from colorama import Fore
import numpy as np

import Core.pose as pose

posImagePath = 'Data/Train/Image/Pos/'
negImagePath = 'Data/Train/Image/Neg/'
resultsPath = 'Data/Train/Processed/'

posDatas = []
negDatas = []

def gamma_trans(img,gamma):
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return cv2.LUT(img,gamma_table)

def processImage(img):
    # cv2.imwrite('Data/Processed' + str(len(posDatas) + len(negDatas)) + '.jpg', img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    poseData = pose.getPosePoints(imgRGB)
    if poseData:
        poseData = [poseData[0]] + poseData[11:16]
        '''xyMin, xyMax = 100000,- 1
        zMin, zMax = 100000,- 1
        for pointData in poseData:
            for i, pointValue in enumerate(pointData):
                if i in (0, 1):
                    xyMax = max(xyMax, pointValue)
                    xyMin = min(xyMin, pointValue)
                else:
                    zMax = max(zMax, pointValue)
                    zMin = min(zMin, pointValue)
        for pointData in poseData:
            for i, pointValue in enumerate(pointData):
                if i in (0, 1):
                    pointData[i] =  (pointData[i] - xyMin) / (xyMax - xyMin)
                else:
                    pointData[i] =  (pointData[i] - zMin) / (zMax - zMin)'''
                    
        '''minValues, maxValues = [100000] * 3, [-1] * 3
        for pointData in poseData:
            for i, pointValue in enumerate(pointData):
                minValues[i] = min(minValues[i], pointValue)
                maxValues[i] = max(maxValues[i], pointValue)
        for pointData in poseData:
            for i, pointValue in enumerate(pointData):
                pointData[i] =  (pointData[i] - minValues[i%3]) / (maxValues[i%3] - minValues[i%3])'''
        return poseData
    else:
        print('Error')
        return None        

def process_(img, dataList):
    data = processImage(img)
    if data:
        dataList.append(data)
        
    img_ = cv2.flip(img, 1)
    data = processImage(img_)
    if data:
        dataList.append(data)

    img_ = gamma_trans(img, 1.2)
    data = processImage(img_)
    if data:
        dataList.append(data)

    img_ = gamma_trans(img, 0.8)
    data = processImage(img_)
    if data:
        dataList.append(data)


def process():
    print(Fore.RED + '加载并处理训练数据中...')
    print(Fore.RESET, end='')

    for path in os.listdir(posImagePath):
        img = cv2.imread(os.path.join(posImagePath, path))

        process_(img, posDatas)

    for path in os.listdir(negImagePath):
        img = cv2.imread(os.path.join(negImagePath, path))

        process_(img, negDatas)

    print(Fore.CYAN + '加载并处理完毕')
    print(Fore.RESET)
    print(f'Pos data length: {len(posDatas)}')
    print(f'Pos data size: {len(posDatas)} * {len(posDatas[0])} * {len(posDatas[0][0])}')
    print('图片数量 * 关键点数量 * 关键点坐标维度')
    print()
    print(f'Neg data length: {len(negDatas)}')
    print(f'Neg data size: {len(negDatas)} * {len(negDatas[0])} * {len(negDatas[0][0])}')
    print('图片数量 * 关键点数量 * 关键点坐标维度')
    print(Fore.RESET, end='')

    pickle.dump(posDatas, open(os.path.join(resultsPath, 'posData.data'), 'wb'))
    pickle.dump(negDatas, open(os.path.join(resultsPath, 'negData.data'), 'wb'))