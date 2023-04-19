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