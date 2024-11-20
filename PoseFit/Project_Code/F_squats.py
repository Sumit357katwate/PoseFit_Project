import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def angle_btn_3points(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_posture_wrong(CL, CR, DL, DR):
    reasons = []
    if CL < 170 and CR < 170 and DL < 170 and DR < 170:
        if CL < 150 and CL > 80 and DL < 150 and DL > 80 and CR < 150 and CR > 80 and DR < 150 and DR > 80:
            reasons.append("Correct Posture.")
        if CL > 150:
            reasons.append("Left Hip angle is too high.")
        if CR > 150:
            reasons.append("Right Hip angle is too high.")
        if DL > 150:
            reasons.append("Left Knee angle is too high.")
        if DR > 150:
            reasons.append("Right Knee angle is too high.")
        if CL < 80:
            reasons.append("Left Hip angle is too low.")
        if CR < 80:
            reasons.append("Right Hip angle is too low.")
    else:
        reasons.append("You are in a standing position.")
    
    if len(reasons) > 0:
        return True, reasons
    else:
        return False, []

def squats_training_logic():
    cap = cv2.VideoCapture(0)

    # Set the frame size
    frame_width = 1280
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Detect pose landmarks
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                kp = mp_pose.PoseLandmark

                # Extract landmarks and angles...
                # ...

                # Get coordinates and angle (Left Hip angle)
                p1 = [landmarks[kp.LEFT_SHOULDER.value].x, landmarks[kp.LEFT_SHOULDER.value].y]
                p2 = [landmarks[kp.LEFT_HIP.value].x, landmarks[kp.LEFT_HIP.value].y]
                p3 = [landmarks[kp.LEFT_KNEE.value].x, landmarks[kp.LEFT_KNEE.value].y]
                CL = angle_btn_3points(p1, p2, p3)  

                # Get coordinates and angle (Right Hip angle)
                p1 = [landmarks[kp.RIGHT_SHOULDER.value].x, landmarks[kp.RIGHT_SHOULDER.value].y]
                p2 = [landmarks[kp.RIGHT_HIP.value].x, landmarks[kp.RIGHT_HIP.value].y]
                p3 = [landmarks[kp.RIGHT_KNEE.value].x, landmarks[kp.RIGHT_KNEE.value].y]
                CR = angle_btn_3points(p1, p2, p3)

                # Get coordinates and angle (Left Knee angle)
                p1 = [landmarks[kp.LEFT_HIP.value].x, landmarks[kp.LEFT_HIP.value].y]
                p2 = [landmarks[kp.LEFT_KNEE.value].x, landmarks[kp.LEFT_KNEE.value].y]
                p3 = [landmarks[kp.LEFT_ANKLE.value].x, landmarks[kp.LEFT_ANKLE.value].y]
                DL = angle_btn_3points(p1, p2, p3)  

                # Get coordinates and angle (Right Knee angle)
                p1 = [landmarks[kp.RIGHT_HIP.value].x, landmarks[kp.RIGHT_HIP.value].y]
                p2 = [landmarks[kp.RIGHT_KNEE.value].x, landmarks[kp.RIGHT_KNEE.value].y]
                p3 = [landmarks[kp.RIGHT_ANKLE.value].x, landmarks[kp.RIGHT_ANKLE.value].y]
                DR = angle_btn_3points(p1, p2, p3)

                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

                # Display angles
                cv2.putText(frame, f'Left Hip Angle: {int(CL)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Right Hip Angle: {int(CR)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Left Knee Angle: {int(DL)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Right Knee Angle: {int(DR)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Check for wrong posture
                posture_wrong, reasons = is_posture_wrong(CL, CR, DL, DR)
                if posture_wrong:
                    reason_text = "Wrong posture: " + ", ".join(reasons)
                    cv2.putText(frame, reason_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Squat Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
