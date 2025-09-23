from pathlib import Path

class ModelConfig:
    SEQUENCE_LENGTH = 60 # 단어 길이에 맞춰 60프레임으로 조정
    FEATURE_DIM = 195    # MediaPipe landmark 차원 Pose-face (11) + Pose-arm (6) + Pose-hand (6) + Hand (21*2) = 65 * 3 = 195
    BUCKETS = (30, 45, 60, 75, 90)  # 버킷 기반 시퀀스 길이 정규화용
    NUM_CLASSES = 499    # AI-Hub 수어단어 데이터 기준 -> 이건 우리가 학습하려는 단어 개수로 고치면 됨  
    BATCH_SIZE = 32      # L40S 48GB GPU 최적화
    EPOCHS = 200        # L40S 48GB GPU 최적화
    LEARNING_RATE = 1e-3 # L40S 48GB GPU 최적화
    DROPOUT_RATE = 0.4   # L40S 48GB GPU 최적화
    MHA_HEADS = 8        # L40S 48GB GPU 최적화
    MHA_KEY_DIM = 64     # L40S 48GB GPU 최적화
    PATIENCE = 100

    DATA_ROOT = Path("./dataset_filtered_WORD500") # 학습용 데이터 넣어서 함.
    MODEL_SAVE_PATH = Path("./Sleep_BiLSTM_MHAttention_model") # 모델 저장 경로