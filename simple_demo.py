import cv2
import torch

from Core.module import PoseNet
import Core.view as view

net = PoseNet()

net = torch.load('./Model/net.pth')
net.eval()

def video_demo():
    capture = cv2.VideoCapture(1)
    while True:
        success, img = capture.read()
        if success:
            img, poseData = view.draw(img)
            #poseData = pose.getPosePoints(imgRGB)
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
                output = net(torch.Tensor(poseData_)).item()
                good = True if output >= 0.3 else False

                if good:
                    cv2.putText(img, f'Good {output}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)
                else:
                    cv2.putText(img, f'Bad {output}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)

            cv2.imshow("video", img)
        c = cv2.waitKey(50)
        if c == 27:
            break

video_demo()
cv2.destroyAllWindows()
