# Training 폴더의 주피터 노트북이 AI 모델 코드입니다.
### 혹시라도 사용해보시려면 test/chmera_test.py를 실행하시면 직접 해보실 수 있습니다. 모델과 인코더를 함께 업로드 하였습니다. .env를 만드시고 test/model의 모델과 인코더 경로를 넣으시면 됩니다.

### 하드웨어
싸피에서 제공해주는 GPU 서버를 활용하였습니다.
GPU 모델은  Tesla V100-PCIE-32GB 입니다.

## 모델 구성
### 모델
    - 모델 파일 : training/model_optimizing.ipynb 
    - 하이퍼 파라미터 저장 파일 : training/config.py
1. TimeDistributed 1D CNN을 선택. 16 + 32 + 64 채널의 3층
2. BiLSTM 256 + 128
3. MultiHeadAttention(head=4, dim=32)로 구성하여 LSTM의 마지막 채널인 128과 맞추었습니다.
4. Flatten

### 관련 내용 정리
0. 성능 
    - test_acc = 0.93, val_acc=0.83 
    - test_loss = 1.5 , val_loss = 1.65 
    - top5는 매우 잘 나옵니다. 그래서 전처리한 데이터로 테스트하면 매우 잘 나오는데 직접 수어를 해서 테스트하면 특정 단어들을 제외하면 top5에도 들어가지 않습니다.

1. 시계열을 CNN에 적용하지 않으면 성능이 매우 떨어집니다. 하지만 연산량이 많아지는 단점이 있는데 성능이 너무 급락해서 포기할 수가 없을 것 같습니다. (정확도 20% 이상 차이)

2. GlobalMaxPooling과 Flatten 사이에서 몇 번 테스트를 했는데 Flatten의 성능이 압도적으로 잘 나옵니다(13% 이상 차이)

3. LSTM은 양방향으로 하는게 가장 성능이 좋았지만 val_acc 기준 3% 정도 차이나서 경량화를 생각하면 단방향이 나을까요? 그리고 GRU를 통해 경량화를 할 수 있다고 하는데 테스트를 한 번 간략하게 했을때는 val_acc 0.6 정도의 벽에 가로막혔습니다.

4. 양자화는 성능도 잘 안나오는데 시도하기엔 이르다고 생각하여 시도하지 않았습니다.

### 데이터 전처리


### 모델 최적화, 경량화 과정입니다.
https://www.notion.so/2721a5fc266380b9b570c431625cb788?source=copy_link