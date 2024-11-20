
#F_bicep.py
import cv2
import mediapipe as mp
import numpy as np

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to check body posture
def check_posture(landmarks):
    # Get coordinates for joint 11, 13, 15
    joint_11 = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y]
    joint_13 = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x,
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y]
    joint_15 = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y]

    # Get coordinates for joint 12, 14, 16
    joint_12 = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y]
    joint_14 = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y]
    joint_16 = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y]

    # Calculate angles
    angle_1 = calculate_angle(joint_11, joint_13, joint_15)
    angle_2 = calculate_angle(joint_12, joint_14, joint_16)

    return angle_1, angle_2

# Main function for counting curls
def count_curls():
    # Open webcam
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0
    stage = None
    framec = 0
    stable_counter = 0
    curr_angle1 = 0
    curr_angle2 = 0
    # Setup mediapipe instance
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            framec += 1
            stable_counter += 1
            ret, frame = cap.read()

            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
                    
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image2 = 255 * np.ones(shape=image.shape, dtype=np.uint8)
                    
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Calculate angles for posture
                angle_1, angle_2 = check_posture(landmarks)
                        
                if stable_counter < 5 and curr_angle1 != 0 and curr_angle2 != 0:
                    angle_1 = curr_angle1
                    angle_2 = curr_angle2
                else:
                    curr_angle1 = angle_1
                    curr_angle2 = angle_2
                    stable_counter = 0

                # Display angles
                cv2.putText(image, 'Left Arm: {:.2f}'.format(angle_1),
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'Right Arm: {:.2f}'.format(angle_2),
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image2, 'Left Arm: {:.2f}'.format(angle_1),
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image2, 'Right Arm: {:.2f}'.format(angle_2),
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Check posture difference
                if abs(angle_1 - angle_2) > 20:
                    cv2.putText(image, 'You are doing it wrong!',
                                (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image2, 'You are doing it wrong!',
                                (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Good posture!',
                                (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image2, 'Good posture!',
                                (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Get coordinates for counting curls
                shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y]

                # Calculate angle for curl counting
                angle = calculate_angle(shoulder, elbow, wrist)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

                # Render detections
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                            mp.solutions.pose.POSE_CONNECTIONS,
                                                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
                                                                                                        thickness=2,
                                                                                                        circle_radius=2),
                                                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230),
                                                                                                        thickness=2,
                                                                                                        circle_radius=2)
                                                            )
                mp.solutions.drawing_utils.draw_landmarks(image2, results.pose_landmarks,
                                                            mp.solutions.pose.POSE_CONNECTIONS,
                                                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
                                                                                                        thickness=2,
                                                                                                        circle_radius=2),
                                                            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230),
                                                                                                        thickness=2,
                                                                                                        circle_radius=2)
                                                            )

                # Show images
                cv2.putText(image, 'REPS: {}'.format(counter),
                            (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image2, 'REPS: {}'.format(counter),
                            (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Mediapipe Feed', image)
                cv2.imshow("Pose Only", image2)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            except:
                pass

    cap.release()
    cv2.destroyAllWindows()
