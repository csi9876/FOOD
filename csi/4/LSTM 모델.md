### 양방향 LSTM

- 챗봇 엔진에 개체명 인식을 위해 사용

- 순화 신경망 모델의 일종으로 시퀀스 또는 시계열 데이터의 패턴

### RNN

- 순환 신경망
- 은닉층의 출력값을 출력층과 그 다음 시점의 은닉층의 입력으로 전달해 순환

![](LSTM%20모델_assets/2023-04-18-17-32-58-image.png)

Xt = 현재 시점 입력벡터

Yt = 현재 시점 출력벡터

은닉층 노드는 이전 시점의 상태값을 저장하는 메모리 역할 > 메모리 셀

메모리 셀의 출력 벡터는 출력층과 다음 시점 메모리 셀에전달 : 은닉 상태

ht = 현재 시점 은닉상태

![](LSTM%20모델_assets/2023-04-18-17-35-16-image.png)

완전 연결 계층

    

![](LSTM%20모델_assets/2023-04-18-17-35-59-image.png)

![](LSTM%20모델_assets/2023-04-18-17-36-15-image.png)

![](LSTM%20모델_assets/2023-04-18-17-36-23-image.png)

RNN 은 모든 시점에서 동일한 가중치와 편향값 사용

![](LSTM%20모델_assets/2023-04-18-17-37-17-image.png)![](LSTM%20모델_assets/2023-04-18-17-37-25-image.png)

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LSTM, SimpleRNN

# time step만큼 시퀀스 데이터 분리
def split_sequence(sequence, step):
    x, y = list(), list()

    for i in range(len(sequence)):
        end_idx = i + step
        if end_idx > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


# sin 함수 학습 데이터
x = [i for i in np.arange(start=-10, stop=10, step=0.1)]
train_y = [np.sin(i) for i in x]
# 학습 데이터셋 저장


# 하이퍼파라미터
n_timesteps = 15
n_features = 1
# 입력 시퀀스 길이를 15로 정의, 입력 벡터의 크기 1로 정의


# 시퀀스 나누기
# train_x.shape => (samples, timesteps)
# train_y.shape => (samples)
train_x, train_y = split_sequence(train_y, step=n_timesteps)
print("shape x:{} / y:{}".format(train_x.shape, train_y.shape))
# 학습데이터 셋을 입력 시퀀스 길이만큼 나눠 입력 시퀀스 생성



# RNN 입력 벡터 크기를 맞추기 위해 벡터 차원 크기 변경
# reshape from [samples, timesteps] into [samples, timesteps, features]
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
print("train_x.shape = {}".format(train_x.shape))
print("train_y.shape = {}".format(train_y.shape))
# 2차원 x모델을 3차원 형태로 변환



# RNN 모델 정의
model = Sequential()
model.add(SimpleRNN(units=10, return_sequences=False, input_shape=(n_timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# RNN계층정의 후 모델 생성
# simple RNN 계층
# 입력 데이터 형상 정의
# 손실함수 mse
# 옵티마이저 adam



# 모델 학습
np.random.seed(0)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
history = model.fit(train_x, train_y, epochs=1000, callbacks=[early_stopping])
# 모델 학습
# 조기종료 함수 설정(급격히 손실이 증가하면)


# loss 그래프 생성
plt.plot(history.history['loss'], label="loss")
plt.legend(loc="upper right")
plt.show()


# 테스트 데이터셋 생성
test_x = np.arange(10, 20, 0.1)
calc_y = np.cos(test_x) # 테스트 정답 데이터
# RNN모델을 테스트 하기위해
# cos()를 이용하여 원본 데이터에 주기적 차이를 준다


# RNN 모델 예측 및 로그 저장
test_y = calc_y[:n_timesteps]
for i in range(len(test_x) - n_timesteps):
    net_input = test_y[i : i + n_timesteps]
    net_input = net_input.reshape((1, n_timesteps, n_features))
    train_y = model.predict(net_input, verbose=0)
    print(test_y.shape, train_y.shape, i, i + n_timesteps)
    test_y = np.append(test_y, train_y)
# 예측값을 저장



# 예측 결과 그래프 그리기
plt.plot(test_x, calc_y, label="ground truth", color="orange")
plt.plot(test_x, test_y, label="predicitons", color="blue")
plt.legend(loc='upper left')
plt.ylim(-2, 2)
plt.show()
# 출력
# 오차가 거의 없다
```

![](LSTM%20모델_assets/2023-04-18-00-31-53-image.png)

    ![](LSTM%20모델_assets/2023-04-18-17-53-38-image.png)

- RNN은 입력 시퀀스의 시점이 길어질수록 앞쪽의 데이터가 뒤쪽으로 잘 전달되지 않는다

- 다층 구조로 쌓으면 입력과 출력 사이의 연관성이 줄어 장기 의존성 문제

### LSTM

- 셀 상태값

![](LSTM%20모델_assets/2023-04-18-17-55-45-image.png)

- 입력게이트 : 현재 정보를 얼마나 기억할지 결정
  
  - 현재 입력값을 얼마나 반영할지
  
  - 현재 시점입력값과 이전 시점 은닉 상태값을 2개의 활성화 함수로 계산
  
  - 시그모이드 : 0~1, 하이퍼블릭 탄젠트 : -1~1 > 두 값의 곱
  
  ![](LSTM%20모델_assets/2023-04-18-17-57-21-image.png)

- 삭제게이트 : 이전 시점의 셀 상태값을 삭제하기 위해 사용
  
  - 이전 은닉층의 값 얼마나 가져올지
  
  - Xt와 ht-1 을 시그모이드로 0~1 사이로 출력
  
  ![](LSTM%20모델_assets/2023-04-18-17-58-28-image.png)

- 출력게이트 : 결과값은 현재 시점의 은닉 상태를 결정
  
  - 현재 은닉층의 값을 결정
  
  - 장기상태(셀 상태값)  > .영향 > 단기상태(은닉상태)
  
  ![](LSTM%20모델_assets/2023-04-18-17-59-36-image.png)

    ![](LSTM%20모델_assets/2023-04-18-17-59-47-image.png)

```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LSTM

# time step만큼 시퀀스 데이터 분리
def split_sequence(sequence, step):
    x, y = list(), list()

    for i in range(len(sequence)):
        end_idx = i + step
        if end_idx > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


# sin 함수 학습 데이터
x = [i for i in np.arange(start=-10, stop=10, step=0.1)]
train_y = [np.sin(i) for i in x]
# 학습데이터셋


# 하이퍼파라미터
n_timesteps = 15
n_features = 1

# 시퀀스 나누기
# train_x.shape => (samples, timesteps)
# train_y.shape => (samples)
train_x, train_y = split_sequence(train_y, step=n_timesteps)
print("shape x:{} / y:{}".format(train_x.shape, train_y.shape))

# RNN 입력 벡터 크기를 맞추기 위해 벡터 차원 크기 변경
# reshape from [samples, timesteps] into [samples, timesteps, features]
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
print("train_x.shape = {}".format(train_x.shape))
print("train_y.shape = {}".format(train_y.shape))


# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=10, return_sequences=False, input_shape=(n_timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# LSTM 계층 1개 + 출력을 위한 Dense 계층 1개




# 모델 학습
np.random.seed(0)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
history = model.fit(train_x, train_y, epochs=1000, callbacks=[early_stopping])

# loss 그래프 생성
plt.plot(history.history['loss'], label="loss")
plt.legend(loc="upper right")
plt.show()

# 테스트 데이터셋 생성
test_x = np.arange(10, 20, 0.1)
calc_y = np.cos(test_x) # 테스트 정답 데이터

# RNN 모델 예측 및 로그 저장
test_y = calc_y[:n_timesteps]
for i in range(len(test_x) - n_timesteps):
    net_input = test_y[i : i + n_timesteps]
    net_input = net_input.reshape((1, n_timesteps, n_features))
    train_y = model.predict(net_input, verbose=0)
    print(test_y.shape, train_y.shape, i, i + n_timesteps)
    test_y = np.append(test_y, train_y)

# 예측 결과 그래프 그리기
plt.plot(test_x, calc_y, label="ground truth", color="orange")
plt.plot(test_x, test_y, label="predicitons", color="blue")
plt.legend(loc='upper left')
plt.ylim(-2, 2)
plt.show()


'''RNN 모델보다 오차가 더 작다'''
```

![](LSTM%20모델_assets/2023-04-18-00-34-29-image.png)

![](LSTM%20모델_assets/2023-04-18-00-34-49-image.png)

    

### 양뱡향 LSTM

- 문장의 앞부분 뿐만 아니라 뒷부분도 중요하다

- LSTM에 역방향 LSTM을 추가하여 양방향 문장 패턴 분석

![](LSTM%20모델_assets/2023-04-18-19-37-09-image.png)

```py
import numpy as np
from random import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed


# 시퀀스 생성
def get_sequence(n_timesteps):
    # 0~1 사이의 랜덤 시퀀스 생성
    X = np.array([random() for _ in range(n_timesteps)])

    # 클래스 분류 기준
    limit = n_timesteps / 4.0

    # 누적합 시퀀스에서 클래스 결정
    # 누적합 항목이 limit보다 작은 경우 0, 아닌 경우 1로 분류
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    # LSTM 입력을 위해 3차원 텐서 형태로 변경
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y


# 하이퍼파라미터 정의
n_units = 20
n_timesteps = 4

# 양방향 LSTM 모델 정의
model = Sequential()
model.add(Bidirectional(LSTM(n_units, return_sequences=True, input_shape=(n_timesteps, 1))))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 양방향을 위해 Bidirectional wrapper 사용
# 정방향 역방향 LSTM 계층에 모든 출력값을 연결해야 하기 때문에 반드시  return_sequences=True
# Dense 계층을 3차원을 입력받을 수  있게 TimeDistributed로 확장


# 모델 학습
# 에포크마다 학습 데이터를 생성해서 학습
for epoch in range(1000):
    X, y = get_sequence(n_timesteps)
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)

# 모델 평가
X, y = get_sequence(n_timesteps)
# yhat = model.predict_classes(X, verbose=0)
yhat = model.predict(X, verbose=0)
predict = yhat.argmax(axis=-1)
for i in range(n_timesteps):
    print('실젯값 :', y[0, i], '예측값 : ', yhat[0, i])


'''
실젯값 : [0] 예측값 :  [0.02358541]
실젯값 : [1] 예측값 :  [0.8198795]
실젯값 : [1] 예측값 :  [0.9865184]
실젯값 : [1] 예측값 :  [0.996708]
'''
```

    

### 개체명 인식

- 문장에서 각 개체의 유형을 인식

- 문장 내 포함된 어떤 단어가 무엇을 의미하는 지 인식

```py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


# 학습 파일 불러오기
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents


# 학습용 말뭉치 데이터를 불러옴
corpus = read_file('train.txt')

# 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        tagged_sentence.append((w[1], w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])

    sentences.append(sentence)
    tags.append(bio_tag)

print("샘플 크기 : \n", len(sentences))
print("0번째 샘플 문장 시퀀스 : \n", sentences[0])
print("0번째 샘플 bio 태그 : \n", tags[0])
print("샘플 문장 시퀀스 최대 길이 :", max(len(l) for l in sentences))
print("샘플 문장 시퀀스 평균 길이 :", (sum(map(len, sentences))/len(sentences)))


# 토크나이저 정의
sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV') # 첫 번째 인덱스에는 OOV 사용
sent_tokenizer.fit_on_texts(sentences)
tag_tokenizer = preprocessing.text.Tokenizer(lower=False) # 태그 정보는 lower= False 소문자로 변환하지 않는다.
tag_tokenizer.fit_on_texts(tags)



# 단어 사전 및 태그 사전 크기
vocab_size = len(sent_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)



# 학습용 단어 시퀀스 생성
x_train = sent_tokenizer.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)
print(x_train[0])
print(y_train[0])



# index to word / index to NER 정의
index_to_word = sent_tokenizer.index_word # 시퀀스 인덱스를 단어로 변환하기 위해 사용
index_to_ner = tag_tokenizer.index_word # 시퀀스 인덱스를 NER로 변환하기 위해 사용
index_to_ner[0] = 'PAD'



# 시퀀스 패딩 처리
max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)



# 학습 데이터와 테스트 데이터를 8:2 비율로 분리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.2, random_state=0)



# 출력 데이터를 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print("학습 샘플 시퀀스 형상 : ", x_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)



# 모델 정의(Bi-LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10)
print("평가 결과 : ", model.evaluate(x_test, y_test)[1])
# 순차모델


# 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):
    result = []
    for sequence in sequences:
        temp = []
        for pred in sequence:
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(temp)
    return result


# 테스트 데이터셋의 NER 예측
y_predicted = model.predict(x_test) # (711, 40) => model => (711, 40, 8)
pred_tags = sequences_to_tag(y_predicted) # 예측된 NER
test_tags = sequences_to_tag(y_test) # 실제 NER

# F1 스코어 계산을 위해 사용
from seqeval.metrics import f1_score, classification_report
print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))


# 새로운 유형의 문장 NER 예측
word_to_index = sent_tokenizer.word_index
new_sentence = '삼성전자 출시 스마트폰 오늘 애플 도전장 내밀다.'.split()
new_x = []
for w in new_sentence:
    try:
        new_x.append(word_to_index.get(w, 1))
    except KeyError:
        # 모르는 단어의 경우 OOV
        new_x.append(word_to_index['OOV'])

print("새로운 유형의 시퀀스 : ", new_x)
new_padded_seqs = preprocessing.sequence.pad_sequences([new_x], padding="post", value=0, maxlen=max_len)

# NER 예측
p = model.predict(np.array([new_padded_seqs[0]]))
p = np.argmax(p, axis=-1) # 예측된 NER 인덱스값 추출
print("{:10} {:5}".format("단어", "예측된 NER"))
print("-" * 50)

for w, pred in zip(new_sentence, p[0]):
    print("{:10} {:5}".format(w, index_to_ner[pred]))





'''
Epoch 1/10
23/23 [==============================] - 25s 669ms/step - loss: 0.5087 - accuracy: 0.8360
Epoch 2/10
23/23 [==============================] - 15s 662ms/step - loss: 0.2318 - accuracy: 0.8996
Epoch 3/10
23/23 [==============================] - 15s 659ms/step - loss: 0.1518 - accuracy: 0.9260
Epoch 4/10
23/23 [==============================] - 16s 674ms/step - loss: 0.1129 - accuracy: 0.9442
Epoch 5/10
23/23 [==============================] - 16s 674ms/step - loss: 0.0825 - accuracy: 0.9615
Epoch 6/10
23/23 [==============================] - 16s 676ms/step - loss: 0.0567 - accuracy: 0.9751
Epoch 7/10
23/23 [==============================] - 16s 686ms/step - loss: 0.0408 - accuracy: 0.9825
Epoch 8/10
23/23 [==============================] - 16s 677ms/step - loss: 0.0338 - accuracy: 0.9854
Epoch 9/10
23/23 [==============================] - 15s 671ms/step - loss: 0.0266 - accuracy: 0.9882
Epoch 10/10
23/23 [==============================] - 16s 679ms/step - loss: 0.0237 - accuracy: 0.9893
23/23 [==============================] - 3s 65ms/step - loss: 0.1979 - accuracy: 0.9376
평가 결과 :  0.9375531077384949
23/23 [==============================] - 3s 67ms/step


              precision    recall  f1-score   support

           _       0.61      0.59      0.60       657
         _DT       0.89      0.92      0.91       332
         _LC       0.78      0.54      0.64       319
         _OG       0.68      0.60      0.64       490
         _PS       0.73      0.45      0.56       379
         _TI       0.92      0.81      0.86        68

   micro avg       0.72      0.62      0.67      2245
   macro avg       0.77      0.65      0.70      2245
weighted avg       0.72      0.62      0.66      2245

F1-score: 66.5%
새로운 유형의 시퀀스 :  [531, 307, 1475, 286, 1505, 6765, 1]
1/1 [==============================] - 0s 61ms/step
단어         예측된 NER
--------------------------------------------------
삼성전자       B_OG
출시         O
스마트폰       O
오늘         B_DT
애플         B_OG
도전장        I
내밀다.       I
'''
```