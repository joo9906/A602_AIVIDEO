import numpy as np


def pad_or_trim(seq, target_len=60):
    """
    시퀀스 길이를 target_len으로 맞춤
    seq: (T, F)
    """
    T, F = seq.shape
    if T > target_len:
        # 길면 앞에서 target_len만 자르기 (혹은 가운데, 뒤에서 잘라도 됨)
        return seq[:target_len, :]
    elif T < target_len:
        # 짧으면 패딩 (0으로 채움)
        pad = np.zeros((target_len - T, F))
        return np.vstack([seq, pad])
    else:
        return seq


# MODEL_PATH = os.getenv("MODEL_PATH")
# LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH")
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# file_path = os.path.join(BASE_DIR, "testfile.npz")

# # 모델 & 라벨 로딩
# model = keras.models.load_model(MODEL_PATH)
# with open(LABEL_ENCODER_PATH, "rb") as f:
#     label_encoder = pickle.load(f)

# # 데이터 불러오기
# data = np.load(file_path)
# print("!!!!!!!!!!!!!!!!keys:", data.files)

# X = data["x"]        # (90, 195)
# y = data["label"]

# # 프레임 길이를 60으로 정규화
# X_fixed = pad_or_trim(X, target_len=60)   # (60, 195)

# sample = np.expand_dims(X_fixed, axis=0)   # (1, 60, 195)
# sample = np.expand_dims(sample, axis=-1)   # (1, 60, 195, 1)

# # 예측
# probs = model.predict(sample)[0]   # (클래스 수,)

# # 확률 상위 3개 인덱스
# top3_idx = np.argsort(probs)[-3:][::-1]
# top3_words = label_encoder.inverse_transform(top3_idx)
# top3_probs = probs[top3_idx]

# print('X 형태', X.shape)
# print("실제 라벨:", y)
# print("Top-3 예측:")
# for word, p in zip(top3_words, top3_probs):
#     print(f"{word}: {p:.4f}")