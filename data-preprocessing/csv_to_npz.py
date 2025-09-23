"""
추출한 좌표 데이터를 모델 학습에 필요한 데이터로 변경하는 함수 모음
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

from real_ksl_range import extract_frames

# 프로젝트 설정 파일 사용하지 않음
ARM_IDS = [11, 13, 15, 12, 14, 16] # L-어깨/팔꿈치/손목, R-어깨/팔꿈치/손목 고정 순서
HAND_IDS = [17, 19, 21, 18, 20, 22]
FACE_IDS = [n for n in range(0, 11)]
NUM_JOINTS = 65
NUM_COORDS = 3
BUCKETS = (30, 45, 60, 75, 90)
SEQ_LEN = 60
FEATURE_DIM = NUM_JOINTS * NUM_COORDS

# CSV 스키마 검사
def check_csv_schema(df):
    """필요한 컬럼이 모두 있는지 확인"""
    required_columns = {"frame", "part", "jid", "x", "y", "z"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV 스키마에 필요한 컬럼이 누락되었습니다: {sorted(missing)}")


# 결측치 보간하기 (관절별/좌표축별 시간방향 선형보간)
def interpolate_in_time(X):
    """
    X: 시간 별 배열
    """
    T, J, C = X.shape
    X = X.copy()
    for j in range(J):
        for c in range(C):
            series = pd.Series(X[:, j, c], dtype="float32")
            if series.isna().all():
                # 아예 관절/축이 전부 NaN이면 0으로
                X[:, j, c] = 0.0
            else:
                # 양방향 보간(양끝 외삽)
                X[:, j, c] = series.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return X


 # 4) 프레임별 정규화 (어깨 중심/어깨 너비로 표준화)
# 기본은 왼·오른쪽 어깨의 중점을 원점(0,0,0)으로, 어깨 너비를 1로 맞춤
def center_scale_at_frame(X):
    """
    X: 결측치 보간 후 배열
    - 원점: 왼/오른 어깨의 중점(42번째~ 인덱스에 매핑된 어깨 포인트)을 (0,0,0)으로 이동
    - 스케일: 어깨 좌우 거리의 xy 평면 거리(norm)를 1로 맞춤
      * 어깨 좌표가 NaN이거나 너무 가까우면(≈0) 대체 스케일 사용:
        - 현재 프레임의 유효 관절들의 중앙값을 center로 잡고,
        - 그에 대한 x, y 거리의 중앙값을 스케일로 사용(0 방지)
    """
    T = X.shape[0]
    Y = X.copy()

    # 팔 포즈 인덱스에서 "왼어깨"는 slot 0, "오른어깨"는 slot 3으로 매핑됨
    L_SHOULDER_IDX = 42 + 0    # 왼 어깨
    R_SHOULDER_IDX = 42 + 3   # 오른 어깨

    for t in range(T):
        left_shoulder  = Y[t, L_SHOULDER_IDX]
        right_shoulder = Y[t, R_SHOULDER_IDX]


    def fallback_center_scale():
        # 유효 관절만 사용
        valid = np.isfinite(Y[t]).all(axis=1)
        if valid.any():
            center = np.nanmedian(Y[t][valid], axis=0).astype(np.float32)
            d = np.linalg.norm(Y[t][valid, :2] - center[:2], axis=1)
            scale = float(np.nanmedian(d)) or 1.0
        else:
            center = np.zeros(3, np.float32)
            scale = 1.0
        return center, np.float32(scale)


    if np.isfinite(left_shoulder).all() and np.isfinite(right_shoulder).all():
        center = 0.5 * (left_shoulder + right_shoulder)
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])

        # 스케일이 너무 작다면
        if not np.isfinite(scale) or scale < 1e-3:
            center, scale = fallback_center_scale()

    # 어깨 랜드마크가 누락되었을 경우
    else:
        center, scale = fallback_center_scale()

    Y[t] = (Y[t] - center) / scale

    return Y


def csv_to_narray_from_df(frame_df):
    """
    Args:
        frame_df: 프레임 구간이 '이미 잘려 있는' DataFrame(seg_df)

    Returns:
        A: (T, 195) float32    # 프레임 T개, 피처 195(65 * 3)
        T : 원본 프레임 길이 (정수)

    - 프레임 순서대로 (T, 65, 3) 버퍼에 관절 좌표를 채운다.
    - 시간 보간 -> 프레임별 정규화 -> (T, 195) reshape
    """

    # 필요한 컬럼 체크
    check_csv_schema(frame_df)

    # 프레임 순서 정렬
    frames = sorted(frame_df["frame"].unique().tolist())
    T = len(frames)   # 총 프레임 수

    X = np.full((T, NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)   # 좌표 저장용 버퍼

    # 2) (T, 65, 3) 배열에 프레임별 좌표 채우기 (왼손/오른손/팔) - 관절 인덱스 매핑
    for ti, fr in enumerate(frames):
        sub_frame = frame_df[frame_df.frame == fr]   # 프레임마다 서브 데이터 프레임을 만든다.

        # 왼손: MediaPipe Hands 21 포인트 → 0..20
        left_hand = sub_frame[(sub_frame.part == "hand") & (sub_frame.hand_label == "Left")][["jid", "x", "y", "z"]]
        for _, row in left_hand.iterrows():
            joint_number = int(row.jid)
            if 0 <= joint_number < 21:
                X[ti, joint_number, :] = [row.x, row.y, row.z]

        # 오른손: 21..41(= 21 + 0..20)
        right_hand = sub_frame[(sub_frame.part == "hand") & (sub_frame.hand_label == "Right")][["jid", "x", "y", "z"]]
        for _, row in right_hand.iterrows():
            joint_number = int(row.jid)
            if 0 <= joint_number < 21:
                X[ti, 21 + joint_number, :] = [row.x, row.y, row.z]

        # Arms (Pose : 11,13,15, 12,14,16) -> X[:, 42..47, :]
        arm_pose = sub_frame[sub_frame.part == "pose_arm"][["jid", "x", "y", "z"]]
        for i, joint_id in enumerate(ARM_IDS):
            arm_pose_row = arm_pose[arm_pose.jid == joint_id]
            if not arm_pose_row.empty:
                row = arm_pose_row.iloc[0]
                X[ti, 42 + i, :] = [row.x, row.y, row.z]

        # Hands (Pose) 48 ~ 53 -> 17, 19, 21, 18, 20, 22
        hand_pose = sub_frame[sub_frame.part == "pose_hand"][["jid", "x", "y", "z"]]
        for i, joint_id in enumerate(HAND_IDS):
            hand_pose_row = hand_pose[hand_pose.jid == joint_id]
            if not hand_pose_row.empty:
                row = hand_pose_row.iloc[0]
                X[ti, 48 + i, :] = [row.x, row.y, row.z]

        # Face (Pose) 54 ~ 65
        face = sub_frame[sub_frame.part == "face"][["jid", "x", "y", "z"]]
        for i, joint_id in enumerate(FACE_IDS):
            face_row = face[face.jid == joint_id]
            if not face_row.empty:
                row = face_row.iloc[0]
                X[ti, 54 + i, :] = [row.x, row.y, row.z]

    # 결측치 보간
    X = interpolate_in_time(X)

    # 프레임별 정규화
    X = center_scale_at_frame(X)

    # (T, 65, 3) -> (T, 195)
    A = X.reshape(X.shape[0], -1).astype(np.float32)

    return A, T

# Convert JSON frame data to narray
def json_to_narray_from_frames(frame_list_json):
    """
    Args:
        frame_list_json: extract_frames_from_json 함수에서 반환된 JSON 프레임 데이터 목록 (list of dictionaries)

    Returns:
        A: (T, 195) float32    # 프레임 T개, 피처 195(65 * 3)
        T : 원본 프레임 길이 (정수)

    - 프레임 순서대로 (T, 65, 3) 버퍼에 관절 좌표를 채운다.
    - 시간 보간 -> 프레임별 정규화 -> (T, 195) reshape
    """

    T = len(frame_list_json)  # 총 프레임 수
    X = np.full((T, NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)   # 좌표 저장용 버퍼

    # 2) (T, 65, 3) 배열에 프레임별 좌표 채우기 (왼손/오른손/팔) - 관절 인덱스 매핑
    for ti, frame_data in enumerate(frame_list_json):
        # Extract pose landmarks (assuming 23 landmarks, 0-22)
        if 'pose' in frame_data:
            for jid, landmark in enumerate(frame_data['pose']):
                 if 'x' in landmark and 'y' in landmark and 'z' in landmark:
                     
                    # Map pose - 42 ~ 65
                    if jid in ARM_IDS: # Arms (Pose : 11,13,15, 12,14,16) -> X[:, 42..47, :]
                         mapped_idx = 42 + ARM_IDS.index(jid)
                         X[ti, mapped_idx, :] = [landmark['x'], landmark['y'], landmark['z']]
                        
                    elif jid in HAND_IDS: # Hands (Pose) 48 ~ 53 -> 17, 19, 21, 18, 20, 22
                         mapped_idx = 48 + HAND_IDS.index(jid)
                         X[ti, mapped_idx, :] = [landmark['x'], landmark['y'], landmark['z']]
                        
                    elif jid in FACE_IDS: # Face (Pose) 54 ~ 65 -> 0~10 (first 11 pose landmarks)
                         mapped_idx = 54 + FACE_IDS.index(jid)
                         X[ti, mapped_idx, :] = [landmark['x'], landmark['y'], landmark['z']]
                    # Add other pose mappings if necessary (e.g., body, legs)

        # Extract left hand landmarks (assuming 21 landmarks, 0-20)
        if 'left' in frame_data:
            for jid, landmark in enumerate(frame_data['left']):
                 if 'x' in landmark and 'y' in landmark and 'z' in landmark:
                    # Map left hand JIDs (0-20) to X[:, 0..20, :]
                    if 0 <= jid < 21:
                        X[ti, jid, :] = [landmark['x'], landmark['y'], landmark['z']]

        # Extract right hand landmarks (assuming 21 landmarks, 0-20)
        if 'right' in frame_data:
            for jid, landmark in enumerate(frame_data['right']):
                 if 'x' in landmark and 'y' in landmark and 'z' in landmark:
                    # Map right hand JIDs (0-20) to X[:, 21..41, :]
                    if 0 <= jid < 21:
                         X[ti, 21 + jid, :] = [landmark['x'], landmark['y'], landmark['z']]


    # 결측치 보간
    X = interpolate_in_time(X) # Reuse the existing interpolate_in_time function

    # 프레임별 정규화
    X = center_scale_at_frame(X) # Reuse the existing center_scale_at_frame function

    # (T, 65, 3) -> (T, 195)
    A = X.reshape(X.shape[0], -1).astype(np.float32)

    return A, T


# 버킷/고정길이 pad/trim (학습 일관성 / 배치 구성)
def pad_or_trim_to_bucket(A, buckets):
    """
    A: (T,195), buckets: 예) (30,45,60,75,90)
    Returns:
        X: (L,195)  # L은 선택된 버킷 길이
        orig_T: int # 원본 길이(CTC 등에서 사용)
        L: int      # 최종 길이(=선택된 버킷)

    - T 이상인 "최소 버킷"으로 pad (ex. T = 52 -> L = 60)
    - 그런 버킷이 없으면 "최대 버킷"으로 trim (ex. T=110 -> L=90)
    """
    T = A.shape[0]
    sorted_b = sorted(buckets)

    # T 이상인 최소 버킷 찾기
    L = next((b for b in sorted_b if b >= T), sorted_b[-1])

    if T == L:
        return A, T, L
    elif T < L:
        pad = np.zeros((L - T, A.shape[1]), dtype=A.dtype)
        return np.concatenate([A, pad], axis=0), T, L
    else:  # T > L → 앞쪽 L 프레임만 사용(간단하고 일관적)
        return A[:L], L, L


def pad_or_trim_to_fixed(A, seq_len):
    """
    A: (T,195), seq_len: 고정 길이(예: 60)
    - T < L: 제로패딩
    - T >= L: 앞쪽 L 프레임만 사용
    """
    T = A.shape[0]
    L = int(seq_len)
    if T == L:
        return A, T, L
    elif T < L:
        pad = np.zeros((L - T, A.shape[1]), dtype=A.dtype)
        return np.concatenate([A, pad], axis=0), T, L
    else:
        return A[:L], L, L


def save_npz_sample_from_segment(
    target_path: str,
    target_type: str,
    metadata: pd.DataFrame,
    out_dir: str,
    label_id: int | None = None,
    use_buckets: bool = True,
    FPS: int = 30,
) -> str:
    """
    메타데이터의 단어 구간만 추출해 (T,195)로 변환 → 버킷/고정길이 pad/trim → NPZ(x, orig_len, label) 저장.

    - extract_frames_from_csv(...)
    - csv_to_narray_from_df(df)     (T,65,3)→보간/정규화→(T,195)
    - pad_or_trim_to_bucket/fixed
    """
    # 추출할 폴더
    os.makedirs(out_dir, exist_ok=True)

    filtered = extract_frames(target_path, target_type, metadata, FPS=FPS)

    if target_type == "csv":

        # 2) DF → (T,195) (네 함수)
        A, T = csv_to_narray_from_df(filtered)  # (T,195), T

    elif target_type == "json":

        # 2) DF → (T,195) (네 함수)
        A, T = json_to_narray_from_frames(filtered)  # (T,195), T

    # 3) 길이 정규화 (버킷/고정)
    if use_buckets:
        X, orig_T, L = pad_or_trim_to_bucket(A, BUCKETS)  # (L,195)
    else:
        X, orig_T, L = pad_or_trim_to_fixed(A, SEQ_LEN)   # (L,195)

    # 4) NPZ 저장
    base = Path(target_path).stem
    out_path = Path(out_dir) / f"{base}.npz"
    np.savez_compressed(
        out_path,
        x=X.astype(np.float32),
        orig_len=np.int32(orig_T),
        label=np.int32(-1 if label_id is None else label_id),
    )
    return str(out_path)