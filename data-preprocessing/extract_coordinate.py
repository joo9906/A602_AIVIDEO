"""
비디오에서 좌표 추출하기
"""

import os
import re
import zipfile
import shutil
import csv, cv2, json
import mediapipe as mp
from typing import Optional

mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

ARM_IDS = [11, 13, 15, 12, 14, 16]
HAND_IDS = [17, 19, 21, 18, 20, 22]
FACE_IDS = [n for n in range(0, 11)]


def process_with_view(
    video_path: str,
    out_coords_path: str,
    out_format: str = "json",           # "csv" | "json"
    every_n: int = 1,                  # 프레임 샘플링(2~3이면 더 빠름)
    resize_factor: float = 1.0,        # 손이 작으면 1.5~2.0 추천

    # FPS 고정 옵션
    fps_override: Optional[float] = None,   # 예: 30.0 → 타임스탬프/라이터 모두 30fps로 고정

    # Mediapipe thresholds
    hand_min_det: float = 0.6,
    hand_min_trk: float = 0.6,
    pose_min_det: float = 0.5,
    pose_min_trk: float = 0.5,
    pose_model_complexity: int = 1,

    # JSON output options
    word_id: Optional[int] = None,
    person_id: Optional[int] = None
):

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_for_time = fps_override if (fps_override and fps_override > 0) else src_fps  # time_s 계산용

    # 모델 준비하기
    # 전체 자세 관련
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_model_complexity,
        enable_segmentation=False,
        min_detection_confidence=pose_min_det,
        min_tracking_confidence=pose_min_trk
    )
    # 손 관련 21개
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=hand_min_det,
        min_tracking_confidence=hand_min_trk
    )

    # 출력 준비
    if out_format.lower() == "csv":
        f = open(out_coords_path, "w", newline="", encoding="utf-8")
        wr = csv.writer(f)
        wr.writerow(["video","frame","time_s","width","height","part",
                     "hand_label","jid","x","y","z","x_px","y_px","visibility"])
        writer_coords = wr
        json_data = None # JSON 포맷이 아닐 경우 초기화
         
    elif out_format.lower() == "json":
        json_data = {
            "wordId": word_id,
            "personId": person_id,
            "frames": []
        }
        f = None # JSON 포맷일 경우 파일 핸들 초기화
        writer_coords = None # JSON 포맷일 경우 writer 초기화
    else:
        raise ValueError("out_format은 'csv' 또는 'json'만 지원")

    # 루프
    frame_idx = 0
    try:
        # 프레임 반복 처리
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % every_n != 0:
                frame_idx += 1
                continue

            if resize_factor != 1.0:
                frame_bgr = cv2.resize(frame_bgr, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
            H, W = frame_bgr.shape[:2]

            # 색상 변환
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            # Mediapipe 추론 - RGB 프레임을 Pose/Hands 모델에 넣어서 랜드마크를 검출한다.
            res_pose  = pose.process(rgb)
            res_hands = hands.process(rgb)

            # -------- 좌표 수집 --------
            frame_data = {}
            if out_format.lower() == "json":
                pose_landmarks = []
                left_hand_landmarks = []
                right_hand_landmarks = []

                # Pose landmarks
                if res_pose.pose_landmarks:
                    for jid, p in enumerate(res_pose.pose_landmarks.landmark):
                         pose_landmarks.append({"x": p.x,"y": p.y,"z": p.z,"w": float(p.visibility)})
                    frame_data["pose"] = pose_landmarks

                # Hand landmarks
                if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
                    for idx, lm in enumerate(res_hands.multi_hand_landmarks):
                        label = res_hands.multi_handedness[idx].classification[0].label  # "Left"/"Right"
                        hand_pts = []
                        for jid, p in enumerate(lm.landmark):
                            hand_pts.append({"x": p.x,"y": p.y,"z": p.z,"w": 0.0}) # Hand landmarks don't have visibility
                        if label == "Left":
                            left_hand_landmarks = hand_pts
                        else:
                            right_hand_landmarks = hand_pts

                if left_hand_landmarks:
                    frame_data["left"] = left_hand_landmarks
                if right_hand_landmarks:
                    frame_data["right"] = right_hand_landmarks

                json_data["frames"].append(frame_data)


            # -------- 좌표 저장 (CSV) --------
            # 비디오 이름, 프레임 번호, 시간(초), 해상도, 손/팔/얼굴 여부, 좌표값(x, y, z), 픽셀 좌표
            if out_format.lower() == "csv":
                base = dict(
                    video=os.path.basename(video_path),
                    frame=frame_idx,
                    time_s=frame_idx/max(1e-6, fps_for_time),
                    width=W, height=H
                )
                left_pts = right_pts = None

                if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
                    for idx, lm in enumerate(res_hands.multi_hand_landmarks):
                        label = res_hands.multi_handedness[idx].classification[0].label  # "Left"/"Right"

                        pts=[]
                        for jid, p in enumerate(lm.landmark):
                            pts.append({"id": jid,"x": p.x,"y": p.y,"z": p.z,"x_px": p.x*W, "y_px": p.y*H})

                        # 왼 오 구분
                        if label=="Left":
                            left_pts=pts

                        else:
                            right_pts=pts


                # 팔, 얼굴, 손 좌표
                arm_pts = face_pts = pose_hand_pts = None
                if res_pose.pose_landmarks:
                    # 팔
                    pts=[]
                    for jid in ARM_IDS:
                        p = res_pose.pose_landmarks.landmark[jid]
                        pts.append({"id":jid,"x": p.x, "y":p.y, "z":p.z, "x_px":p.x*W, "y_px":p.y*H, "visibility":float(p.visibility)})

                    arm_pts=pts

                    # 얼굴
                    pts=[]
                    for jid in FACE_IDS:
                        p = res_pose.pose_landmarks.landmark[jid]
                        pts.append({"id":jid,"x": p.x, "y":p.y, "z":p.z, "x_px":p.x*W, "y_px":p.y*H, "visibility":float(p.visibility)})

                    face_pts=pts

                    # 손
                    pts=[]
                    for jid in HAND_IDS:
                        p = res_pose.pose_landmarks.landmark[jid]
                        pts.append({"id":jid,"x": p.x, "y":p.y, "z":p.z, "x_px":p.x*W, "y_px":p.y*H, "visibility":float(p.visibility)})

                    pose_hand_pts=pts

                if left_pts:
                    for p in left_pts:
                        writer_coords.writerow([base["video"],base["frame"],base["time_s"],W,H,"hand","Left",
                                                p["id"],p["x"],p["y"],p["z"],p["x_px"],p["y_px"],""])
                if right_pts:
                    for p in right_pts:
                        writer_coords.writerow([base["video"],base["frame"],base["time_s"],W,H,"hand","Right",
                                                p["id"],p["x"],p["y"],p["z"],p["x_px"],p["y_px"],""])
                if arm_pts:
                    for p in arm_pts:
                        side = "Left" if p["id"] in [11,13,15] else "Right"
                        writer_coords.writerow([base["video"],base["frame"],base["time_s"],W,H,"pose_arm", side,
                                                p["id"],p["x"],p["y"],p["z"],p["x_px"],p["y_px"],p["visibility"]])
                if pose_hand_pts:
                    for p in pose_hand_pts:
                        side = "Left" if p["id"] in [17,19,21] else "Right"
                        writer_coords.writerow([base["video"],base["frame"],base["time_s"],W,H,"pose_hand",side,
                                                p["id"],p["x"],p["y"],p["z"],p["x_px"],p["y_px"],p["visibility"]])
                if face_pts:
                    for p in face_pts:
                        writer_coords.writerow([base["video"],base["frame"],base["time_s"],W,H,"face","",
                                                p["id"],p["x"],p["y"],p["z"],p["x_px"],p["y_px"],p["visibility"]])

            frame_idx += 1

    finally:
        cap.release()
        pose.close(); hands.close()
        if out_format.lower() == "csv" and f is not None:
             f.close()
             
        elif out_format.lower() == "json" and json_data is not None:
            with open(out_coords_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=None, ensure_ascii=False) # indent=None for JSONL like output

    return dict(coords=out_coords_path, annotated=None)