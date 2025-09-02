import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_pose    = mp.solutions.pose

# 웹캠 열기 (윈도우에서 카메라 이슈 있으면 CAP_DSHOW/MF로 변경)
# 우리 노트북 웹캠 키는거임
cap = cv2.VideoCapture(0)  # 또는 cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose(
    model_complexity=1,                 
    enable_segmentation=False,          # 배경 분리 필요 없으면 False
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    # 카메라가 열려 있는 동안
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # 셀카 찍는것처럼 좌우 반전(원하면)
        frame = cv2.flip(frame, 1)

        # BGR -> RGB 변환 + writeable False(속도 최적화)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        # 포즈 추론
        results = pose.process(img_rgb)

        # 다시 그릴 수 있게 True 설정하고 BGR로 되돌리기
        img_rgb.flags.writeable = True
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow('Mediapipe Pose', frame)

        # 키보드 q 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
