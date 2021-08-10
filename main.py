import cv2
import mediapipe as mp
import numpy as np
import time
import click


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.3, trackingCon=0.3):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # Change the image to black
        img[img > 0] = 0

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPositions(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    if id > 10:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 0), cv2.FILLED)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 50, (0, 0, 0), cv2.FILLED)
                        cv2.circle(img, (cx, cy), 50, (0, 255, 0), 2)
                    if id == 15:
                        cv2.circle(img, (cx, cy), 20, (0, 255, 0), 2)
                    

        return lmList

@click.command()
@click.argument('video_path')
def main(video_path, perc_resize=0.4):

    cap = cv2.VideoCapture(video_path)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        # exit loop if there was problem to get frame to display
        if not success:
            break

        img = cv2.resize(img, (int(img.shape[1] * perc_resize), int(img.shape[0] * perc_resize)))
        original_image = np.copy(img)
        img = detector.findPose(img)
        lmList = detector.findPositions(img, draw=True)

        # Concatenate the 2 videos
        concatenate_img = np.hstack((original_image, img))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(concatenate_img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', concatenate_img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
