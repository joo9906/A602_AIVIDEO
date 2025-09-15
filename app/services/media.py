import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_pose    = mp.solutions.pose
mp_hands   = mp.solutions.hands

# --- 옵션 ---
SHOW_TEXT = True            # 화면에 속도 텍스트 표시
DRAW_POSE = True            # 포즈 랜드마크 그리기
DRAW_HANDS = True           # 손 랜드마크 그리기

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 또는 cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 이전 프레임의 손 좌표 저장(왼손/오른손 키로 관리)
# prev_hands = {"Left": [(x,y), ... 21개], "Right": [(x,y), ... 21개]}
prev_hands = {"Left": None, "Right": None}
prev_time = time.time()

with mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # 거울모드
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        # 추론
        results_pose  = pose.process(img_rgb)
        results_hands = hands.process(img_rgb)

        img_rgb.flags.writeable = True
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h, w, c = frame.shape

        # 시간 간격 계산 (초)
        cur_time = time.time()
        delta_t = cur_time - prev_time
        if delta_t == 0:
            delta_t = 1e-6  # 0 나눗셈 방지
        prev_time = cur_time

        # --- 포즈 그리기 (원하시면 좌표/속도 계산도 동일하게 확장 가능) ---
        if DRAW_POSE and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        # --- 손 좌표 + 속도 계산 ---
        # handedness 정보 취득 (각 손이 Left/Right 인지)
        # results_hands.multi_handedness[i].classification[0].label -> 'Left' or 'Right'
        if results_hands.multi_hand_landmarks:
            # 현재 프레임 손들을 Left/Right로 정리
            cur_hands = {"Left": None, "Right": None}

            # multi_handedness와 multi_hand_landmarks는 같은 인덱스 순서를 가짐
            for hand_idx, (hand_lms, handedness) in enumerate(
                zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)
            ):
                label = handedness.classification[0].label  # 'Left' 또는 'Right'
                # 좌표 추출 (픽셀 단위)
                pts = []
                for idx, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts.append((cx, cy))

                cur_hands[label] = pts

                # 랜드마크 그리기
                if DRAW_HANDS:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # 이제 Left/Right 기준으로 속도 계산
            for label in ["Left", "Right"]:
                cur_pts = cur_hands[label]
                prev_pts = prev_hands[label]

                if cur_pts is not None:
                    for idx, (cx, cy) in enumerate(cur_pts):
                        # 콘솔에 현재 픽셀 좌표 출력(원하시면 주석 해제)
                        # print(f"{label} hand, landmark {idx}: ({cx}, {cy})")

                        if prev_pts is not None and len(prev_pts) == len(cur_pts):
                            vx = (cx - prev_pts[idx][0]) / delta_t  # px/s
                            vy = (cy - prev_pts[idx][1]) / delta_t  # px/s
                            speed = (vx**2 + vy**2) ** 0.5         # 크기

                            # 콘솔 출력
                            print(f"{label} {idx}: pos=({cx},{cy}), vel=({vx:.1f},{vy:.1f}) px/s, |v|={speed:.1f}")

                            # 화면 표시
                            if SHOW_TEXT:
                                cv2.putText(frame, f"{int(vx)},{int(vy)}",
                                            (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.35, (0, 0, 255), 1, cv2.LINE_AA)
                                # 속도 크기(선택): 점 옆에 더 작게
                                cv2.putText(frame, f"|v|={int(speed)}",
                                            (cx+10, cy+12), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.3, (50, 50, 255), 1, cv2.LINE_AA)

            # 현재 프레임 손 좌표를 prev로 저장
            prev_hands = cur_hands
        else:
            # 손이 안보이면 이전값 리셋(원한다면 유지도 가능)
            prev_hands = {"Left": None, "Right": None}

        cv2.imshow('Mediapipe Pose + Hands + Velocity (px/s)', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
