# Training 폴더의 주피터 노트북이 AI 모델 코드입니다.
### 혹시라도 사용해보시려면 test/chmera_test.py를 실행하시면 직접 해보실 수 있습니다. 모델과 인코더를 함께 업로드 하였습니다. .env를 만드시고 test/model의 모델과 인코더 경로를 넣으시면 됩니다.

### 하드웨어
싸피에서 제공해주는 GPU 서버를 활용하였습니다.
GPU 모델은  Tesla V100-PCIE-32GB 입니다.


### 모델 구성
0. 모델
    - 모델 파일 : training/model_optimizing.ipynb 
    - 하이퍼 파라미터 저장 파일 : training/config.py
1. TimeDistributed 1D CNN을 선택. 16 + 32 + 64 채널의 3층으로 구성하였습니다.
2. BiLSTM 256 + 128
3. MultiHeadAttention(head=4, dim=32)로 구성하여 LSTM의 마지막 채널인 128과 맞추었습니다.
4. Flatten

### 데이터 전처리


### 모델 최적화, 경량화 과정입니다.
https://www.notion.so/2721a5fc266380b9b570c431625cb788?source=copy_link