import mediapipe as mp

mpPose = mp.solutions.pose  # 姿态识别模型
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

def getPosePoints(imgRGB):
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        posePoints = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            posePoints.append([lm.x, lm.y, lm.z])

        return posePoints
    return None
