# F_shoulder.py

import cv2
import mediapipe as mp
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def calculate_angle(a, b, c):
    radians = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    return np.degrees(radians)


reps_right = 0
reps_left = 0
up_right = False
up_left = False

# Threshold angle value for detecting incorrect form
threshold_angle = 90

def shoulder_training_logic():
    cap = cv2.VideoCapture(0)  # Open webcam (0 is usually the default webcam)
    up_right = False
    up_left = False
    reps_right = False
    reps_left = False
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280,720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                points[id] = (cx,cy)

            cv2.circle(img, points[12], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[14], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[24], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[11], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[13], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[23], 15, (255,0,0), cv2.FILLED)

            # Calculate angle between joint 14, 12, and 24 (right side)
            a1_right = np.linalg.norm(np.array([points[14][0]-points[12][0], points[14][1]-points[12][1]]))
            b1_right = np.linalg.norm(np.array([points[12][0]-points[24][0], points[12][1]-points[24][1]]))
            c1_right = np.linalg.norm(np.array([points[14][0]-points[24][0], points[14][1]-points[24][1]]))
            angle_1_right = calculate_angle(a1_right, b1_right, c1_right)

            # Calculate angle between joint 13, 11, and 23 (left side)
            a2_left = np.linalg.norm(np.array([points[13][0]-points[11][0], points[13][1]-points[11][1]]))
            b2_left = np.linalg.norm(np.array([points[11][0]-points[23][0], points[11][1]-points[23][1]]))
            c2_left = np.linalg.norm(np.array([points[13][0]-points[23][0], points[13][1]-points[23][1]]))
            angle_2_left = calculate_angle(a2_left, b2_left, c2_left)
            # Right hand reps
            if not up_right and points[14][1] + 40 < points[12][1]:
                up_right = True
            elif points[14][1] > points[12][1]:
                if up_right:
                    reps_right += 1
                    up_right = False

            # Left hand reps
            if not up_left and points[13][1] + 40 < points[11][1]:
                up_left = True
            elif points[13][1] > points[11][1]:
                if up_left:
                    reps_left += 1
                    up_left = False

            cv2.putText(img, "Reps Right: " + str(reps_right), (20,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            cv2.putText(img, "Reps Left: " + str(reps_left), (20,80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            cv2.putText(img, "Angle 1 Right: {:.2f}".format(angle_1_right), (20,120), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
            cv2.putText(img, "Angle 2 Left: {:.2f}".format(angle_2_left), (20,160), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

            # Check for incorrect form based on threshold angle
            if angle_1_right> threshold_angle:
                cv2.putText(img, "Incorrect form: Right arm angle too obtuse", (20,200), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            if angle_2_left > threshold_angle:
                cv2.putText(img, "Incorrect form: Left arm angle too obtuse", (20,240), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow("img",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

