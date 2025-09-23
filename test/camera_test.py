# webcam_record_predict.py
import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path
import os
import json
import pickle
from dotenv import load_dotenv
from model_route import ModelPath as MP

# --- ENV: MODEL_PATH / LABEL_ENCODER_PATH ---
load_dotenv()

# --- Your pipeline helpers ---
from change_to_model_input import change_np_array_from_df
from test_code import pad_or_trim

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ARM_IDS = [11, 13, 15, 12, 14, 16]
HAND_IDS = [17, 19, 21, 18, 20, 22]
FACE_IDS = [n for n in range(0, 11)]

# --- Output path ---
OUTPUT_JSON = Path(__file__).resolve().parent / "data" / "mp_annotation.json"
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# --- UX timing ---
COUNTDOWN_SEC = 3
RECORD_SEC = 3   # 자동 녹화 시간(초)

def build_base():
    return {"frames": []}

def extract_annotation_to_json(result_pose, result_hand):
    frame_data = {}
    pose_landmarks, left_hand_landmarks, right_hand_landmarks = [], [], []

    if result_pose.pose_landmarks:
        for p in result_pose.pose_landmarks.landmark:
            pose_landmarks.append({"x": p.x, "y": p.y, "z": p.z, "w": float(p.visibility)})
        frame_data["pose"] = pose_landmarks

    if result_hand.multi_hand_landmarks and result_hand.multi_handedness:
        for idx, lm in enumerate(result_hand.multi_hand_landmarks):
            label = result_hand.multi_handedness[idx].classification[0].label  # "Left"/"Right"
            pts = [{"x": p.x, "y": p.y, "z": p.z, "w": 0.0} for p in lm.landmark]
            if label == "Left":
                left_hand_landmarks = pts
            else:
                right_hand_landmarks = pts

    frame_data["left"] = left_hand_landmarks if left_hand_landmarks else []
    frame_data["right"] = right_hand_landmarks if right_hand_landmarks else []
    return frame_data

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없음.")

    json_buf = build_base()
    started = False
    start_time = time.time()
    record_start_time = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)  # selfie view
            vis = frame.copy()
            H, W = vis.shape[:2]

            elapsed = int(time.time() - start_time)
            if not started:
                if elapsed < COUNTDOWN_SEC:
                    cv2.putText(vis, str(COUNTDOWN_SEC - elapsed), (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                else:
                    cv2.putText(vis, "START", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                    started = True
                    record_start_time = time.time()

            if started:
                if (time.time() - record_start_time) >= RECORD_SEC:
                    cv2.putText(vis, "Recording done", (100, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.imshow("SELF CAMERA", vis)
                    cv2.waitKey(500)
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_pose = pose.process(rgb)
                result_hand = hands.process(rgb)

                if result_hand.multi_hand_landmarks:
                    for hlm in result_hand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(vis, hlm, mp_hands.HAND_CONNECTIONS)

                if result_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(vis, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    pose_lm = result_pose.pose_landmarks.landmark
                    for idx in ARM_IDS:
                        x, y = int(pose_lm[idx].x * W), int(pose_lm[idx].y * H)
                        cv2.putText(vis, str(idx), (x+8, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                frame_info = extract_annotation_to_json(result_pose, result_hand)
                json_buf["frames"].append(frame_info)

            cv2.imshow("SELF CAMERA", vis)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_buf, f, ensure_ascii=False, indent=2)
    print(f"[저장 완료] {OUTPUT_JSON} | frames: {len(json_buf['frames'])}")

    MODEL_PATH = MP.MODEL_PATH
    LABEL_ENCODER_PATH = MP.LABEL_ENCODER_PATH
    if not MODEL_PATH or not LABEL_ENCODER_PATH:
        raise RuntimeError("MODEL_PATH 또는 LABEL_ENCODER_PATH 환경변수가 설정되지 않았습니다.")

    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    np_data = change_np_array_from_df(annotation_data, use_buckets=True)  # (L,195)
    X_fixed = pad_or_trim(np_data, target_len=60).astype("float32")      # (60,195)

    sample = np.expand_dims(X_fixed, axis=0)  # (1,60,195)
    # 모델이 채널 차원을 기대하면 아래 한 줄 주석 해제
    # sample = np.expand_dims(sample, axis=-1)  # (1,60,195,1)

    probs = model.predict(sample, verbose=0)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_words = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    print("X 형태:", np_data.shape)
    print("Top-3 예측:")
    for word, p in zip(top3_words, top3_probs):
        print(f"{word}: {p:.4f}")

if __name__ == "__main__":
    main()
