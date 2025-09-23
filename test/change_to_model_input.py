"""
추출한 좌표값을 넘파이 배열로 변경하기 (수정판)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# 셀프 카메라는 왼오가 바뀐다 (※ MediaPipe는 해부학적 좌/우를 반환하므로 보통 스왑 불필요)
ARM_IDS  = [11, 13, 15, 12, 14, 16]  # L-어깨/팔꿈치/손목, R-어깨/팔꿈치/손목 고정 순서
HAND_IDS = [17, 19, 21, 18, 20, 22]
FACE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

NUM_JOINTS = 65
NUM_COORDS = 3
BUCKETS    = (30, 45, 60, 75, 90)
SEQ_LEN    = 60


# ----------------------------
# 1) 결측치 보간 (관절×축별 시간 선형보간, 양끝 외삽)
# ----------------------------
def interpolate_in_time(X: np.ndarray) -> np.ndarray:
    """
    X: (T, J, C) float32
    """
    T, J, C = X.shape
    Y = X.copy()
    for j in range(J):
        for c in range(C):
            series = pd.Series(Y[:, j, c], dtype="float32")
            if series.isna().all():
                Y[:, j, c] = 0.0
            else:
                Y[:, j, c] = series.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return Y


# ----------------------------
# 2) 프레임별 정규화 (어깨 중심/어깨 너비, EMA + 스케일 클램핑)
# ----------------------------
def center_scale_at_frame(
    X: np.ndarray,
    alpha: float = 0.2,           # EMA 계수 (0.0~1.0, 클수록 현재 프레임 가중↑)
    clamp_ratio: float = 1.5,     # 프레임간 스케일 변화 허용 비율
    eps: float = 1e-6
) -> np.ndarray:
    """
    - 원점: 왼/오른 어깨 중점
    - 스케일: 왼/오른 어깨의 XY 거리(norm)
    - 어깨가 누락되거나 스케일이 비정상(너무 작음/NaN)이면, 해당 프레임의 유효 랜드마크 중앙값/중앙거리 fallback
    - 프레임 간 스케일 급변 방지: EMA + 전 프레임 대비 clamp
    """
    T = X.shape[0]
    Y = X.copy()

    L_SHOULDER_IDX = 42 + 0  # 왼 어깨
    R_SHOULDER_IDX = 42 + 3  # 오른 어깨

    def fallback_center_scale(t: int) -> Tuple[np.ndarray, np.float32]:
        valid = np.isfinite(Y[t]).all(axis=1)
        if valid.any():
            center = np.nanmedian(Y[t][valid], axis=0).astype(np.float32)
            d = np.linalg.norm(Y[t][valid, :2] - center[:2], axis=1)
            scale = float(np.nanmedian(d)) or 1.0
        else:
            center = np.zeros(3, np.float32)
            scale = 1.0
        return center, np.float32(scale)

    c_prev = None
    s_prev = None
    for t in range(T):
        l = Y[t, L_SHOULDER_IDX]
        r = Y[t, R_SHOULDER_IDX]

        if np.isfinite(l).all() and np.isfinite(r).all():
            center = 0.5 * (l + r)
            scale = float(np.linalg.norm(l[:2] - r[:2]))
            if not np.isfinite(scale) or scale < 1e-3:
                center, scale = fallback_center_scale(t)
        else:
            center, scale = fallback_center_scale(t)

        # EMA 평활화
        if c_prev is None:
            c_s = center.astype(np.float32)
            s_s = np.float32(scale)
        else:
            c_s = (alpha * center + (1.0 - alpha) * c_prev).astype(np.float32)
            s_raw = alpha * scale + (1.0 - alpha) * float(s_prev)
            # 프레임 간 스케일 급변 클램핑
            s_min = float(s_prev) / clamp_ratio
            s_max = float(s_prev) * clamp_ratio
            s_s = np.float32(np.clip(s_raw, s_min, s_max))

        denom = s_s if s_s > eps else 1.0
        Y[t] = (Y[t] - c_s) / denom

        c_prev, s_prev = c_s, s_s

    return Y.astype(np.float32)


# ----------------------------
# 3) JSON → (T,65,3) → 보간/정규화 → (T,195)
# ----------------------------
def json_to_narray_from_frames(frame_list_json: List[Dict]) -> Tuple[np.ndarray, int]:
    """
    Args:
        frame_list_json: [{'pose': [...], 'left': [...], 'right': [...]}, ...]

    Returns:
        A: (T,195) float32
        T: 원본 프레임 수
    """
    T = len(frame_list_json)
    X = np.full((T, NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)

    for ti, frame_data in enumerate(frame_list_json):
        # pose (23 landmarks, 0~22 가정)
        pose_arr = frame_data.get('pose', None)
        if pose_arr:
            for jid, lm in enumerate(pose_arr):
                if ('x' in lm) and ('y' in lm) and ('z' in lm):
                    if jid in ARM_IDS:
                        mapped_idx = 42 + ARM_IDS.index(jid)    # 42..47
                        X[ti, mapped_idx, :] = [lm['x'], lm['y'], lm['z']]
                    elif jid in HAND_IDS:
                        mapped_idx = 48 + HAND_IDS.index(jid)   # 48..53
                        X[ti, mapped_idx, :] = [lm['x'], lm['y'], lm['z']]
                    elif jid in FACE_IDS:
                        mapped_idx = 54 + FACE_IDS.index(jid)   # 54..64
                        X[ti, mapped_idx, :] = [lm['x'], lm['y'], lm['z']]

        # left hand (21 landmarks, 0~20)
        left_arr = frame_data.get('left', None)
        if left_arr:
            for jid, lm in enumerate(left_arr):
                if ('x' in lm) and ('y' in lm) and ('z' in lm) and (0 <= jid < 21):
                    X[ti, jid, :] = [lm['x'], lm['y'], lm['z']]

        # right hand (21 landmarks, 0~20)
        right_arr = frame_data.get('right', None)
        if right_arr:
            for jid, lm in enumerate(right_arr):
                if ('x' in lm) and ('y' in lm) and ('z' in lm) and (0 <= jid < 21):
                    X[ti, 21 + jid, :] = [lm['x'], lm['y'], lm['z']]

    # 1) 결측치 보간
    X = interpolate_in_time(X)
    # 2) 프레임별 정규화 (EMA + 클램핑)
    X = center_scale_at_frame(X, alpha=0.2, clamp_ratio=1.5)

    # 3) (T,65,3) → (T,195)
    A = X.reshape(X.shape[0], -1).astype(np.float32)
    return A, T


# ----------------------------
# 4) 길이 정규화
# ----------------------------
def pad_or_trim_to_bucket(A: np.ndarray, buckets: Tuple[int, ...]) -> Tuple[np.ndarray, int, int]:
    """
    A: (T,195)
    Returns:
        X: (L,195), orig_T: int, L: int
    """
    T = A.shape[0]
    sorted_b = sorted(buckets)
    L = next((b for b in sorted_b if b >= T), sorted_b[-1])

    if T == L:
        return A, T, L
    elif T < L:
        pad = np.zeros((L - T, A.shape[1]), dtype=A.dtype)
        return np.concatenate([A, pad], axis=0), T, L
    else:
        return A[:L], T, L  # orig_T=T 유지


def pad_or_trim_to_fixed(A: np.ndarray, seq_len: int) -> Tuple[np.ndarray, int, int]:
    """
    A: (T,195), seq_len: 고정 길이(예: 60)
    """
    T = A.shape[0]
    L = int(seq_len)
    if T == L:
        return A, T, L
    elif T < L:
        pad = np.zeros((L - T, A.shape[1]), dtype=A.dtype)
        return np.concatenate([A, pad], axis=0), T, L
    else:
        return A[:L], T, L  # orig_T=T 유지


# ----------------------------
# 5) dict(JSON) → (L,195)
# ----------------------------
def change_to_df(target_data: Dict) -> List[Dict]:
    """
    입력 JSON: {'frames': [...]} 형태만 가정.
    필요 시 여기서 후처리(프레임 필터링 등) 가능.
    """
    if 'frames' not in target_data:
        raise ValueError("JSON에 'frames' 키가 없습니다.")
    return list(target_data['frames'])


def change_np_array_from_df(
    target: Dict,
    use_buckets: bool = True,
) -> np.ndarray:
    """
    - JSON → (T,65,3) → 보간/정규화 → (T,195)
    - 버킷/고정길이 pad/trim → (L,195)
    """
    df = change_to_df(target)
    A, T = json_to_narray_from_frames(df)

    if use_buckets:
        X, orig_T, L = pad_or_trim_to_bucket(A, BUCKETS)
    else:
        X, orig_T, L = pad_or_trim_to_fixed(A, SEQ_LEN)

    return X.astype(np.float32)
