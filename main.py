import time
import os

import cv2
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame

from Core import PoseNet, draw

pygame.mixer.init()

def playsound(path):
    sound = pygame.mixer.Sound(path)
    sound.set_volume(1)
    sound.play()

def opencv2QImage(img):
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img.data, width, height, bytesPerLine,
                           QImage.Format_RGB888).rgbSwapped()

    return qImg

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.mainLayout = QHBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.mainLayout)

        self.imgLabel = QLabel()

        self.mainLayout.addWidget(self.imgLabel, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint |
                            Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.capture = cv2.VideoCapture(1)

        self.net = PoseNet()
        self.net = torch.load('./Model/net.pth')
        self.net.eval()

        self.results = []

    def showImage(self, img):
        img = QPixmap.fromImage(opencv2QImage(img))
        self.imgLabel.setPixmap(img.scaledToHeight(self.imgLabel.height()))

    def paintEvent(self, a0) -> None:
        super().paintEvent(a0)
            
        success, img = self.capture.read()
        if success:
            img, poseData = draw(img)
            if poseData is None:
                self.showImage(img)
                self.update()
                return
            if poseData:
                poseData = [poseData[0]] + poseData[11:16]
                poseData_ = []
                min_, max_ = [100000] * 3, [-1] * 3
                for posePointData in poseData:
                    for i, pointValue in enumerate(posePointData):
                        min_[i] = min(min_[i], pointValue)
                        max_[i] = max(max_[i], pointValue)
                        poseData_.append(pointValue)
                for i in range(len(poseData_)):
                    poseData_[i] = (poseData_[i] - min_[i % 3]) / (max_[i % 3] - min_[i % 3])
                output = self.net(torch.Tensor(poseData_)).item()
                good = True if output >= 0.3 else False

                self.results.append(good)
                if len(self.results) > 1000:
                    del self.results[0]
                count = 0
                for result in self.results:
                    if not result:
                        count += 1

                if good:
                    cv2.putText(img, f'Good {round(output, 3)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)
                else:
                    cv2.putText(img, f'Bad {round(output, 3)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)
                    if not pygame.mixer.get_busy() and count > 20:
                        playsound('Audio/audio.mp3')
                        print('Bad!')
        self.showImage(img)
        self.update()

app = QApplication([])
app.setStyleSheet(
    '''
    * {
        background: #000000;
        color: #f3f3f3;
    }
    '''
)
mainWindow = MainWindow()
mainWindow.showFullScreen()
app.exec()