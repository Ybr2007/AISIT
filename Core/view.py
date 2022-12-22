import cv2
import Core.pose as pose

def draw(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.getPosePoints(imgRGB)

    if results:
        for id, lm in enumerate(results):
            if id in (0, 11, 12, 13, 14, 15, 16):
                h, w, c = img.shape
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255))

    return img, results
