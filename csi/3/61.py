# MNIST 예제
# 손글씨 0~9 이미지를딥러닝해 분류하는 것

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# MNIST 데이터셋 가져오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 데이터 정규화
# 학습데이터 train과 테스트 데이터 test를 다운로드 한 후 저장
# xtrain에는 숫자 이미지 > ytrain에 실제 숫자값
# 255픽셀 값의 범위를 나누어 0~1 사이의 실숫값으로 정규화


# tf.data를 사용하여 데이터셋을 섞고 배치 만들기
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
train_size = int(len(x_train) * 0.7)  # 학습셋:검증셋 = 7:3
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).batch(20)
# 학습셋과 검증셋을 일정 비율로 나누어 텐서플로 데이터셋 생성
# 학습하고 학습이 제대로 이루어지는 지 검증
# 배치 사이즈는 전체 학습 데이터셋보다 작거나 동일해야한다


# MNIST 분류 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
# 순차 모델 : 신경망을 구성하는 기본적 방법
# > 복잡한 모델을 구성하기 위해서는 함수형 모델
# Flatten : 신경망의 입력층 : 2차원 이미지를 1차원으로 평탄화
# Dense : 2개의 은닉층 : 활성화 함수 ReLU 사용
#   # 가중치와 출력값의 개수를 조정해주는 함수
# 출력층 : 활성화함수 Softmax 사용
#   # 입력받은 값을 0~1 사이의 값으로 정규화
#   # 분류하고 싶은 클래스 10개 중 가장 큰 값을 가지는 것을 결과값
# 출력값의 총합이 1이 되므로 결과를 확률로 표현할 수 있음
# 가장 큰 출력값을 가지는 클래스가 결과값 > 9
# 입력층을 제외한 나머지 층에서는 입력 크기를 지정하지 않았다. 이전 층의 출력 개수로 입력 크기를 자동 계산


# 모델 생성
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 정의한 신경망 모델을 실제 생성
# 오차 계산 손실 함수 : scc
#   # 손실함수 : 모델의 결과값과 실제 정답과의 오차를 계산하는 함수
# 오차 보정 옵티마이저 : SGD
# 성능평가 항목 : accuracy


# 모델 학습
hist = model.fit(train_ds, validation_data=val_ds, epochs=10)
# 앞에서 생성한 모델을 실제 학습
# 케라스의 fit 함수 사용 ( 학습데이터셋, 검증데이터셋, 에포크값)
# 에포크 : 학습 횟수 / 너무 크면 과적합(학습데이터에 너무 맞춰져서 실제 데이터에 오히려 성능이 낮다)


# 모델 평가
print('모델 평가')
model.evaluate(x_test, y_test)
# 평가


# 모델 정보 출력
model.summary()
# 출력


# 모델 저장
model.save('mnist_model.h5')
# 저장


# 학습 결과 그래프 그리기
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()


'''
Epoch 1/10
2100/2100 [==============================] - 4s 2ms/step - loss: 0.7383 - accuracy: 0.7891 - val_loss: 0.3681 - val_accuracy: 0.8943
Epoch 2/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.3488 - accuracy: 0.8989 - val_loss: 0.3086 - val_accuracy: 0.9097
Epoch 3/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.3019 - accuracy: 0.9127 - val_loss: 0.2689 - val_accuracy: 0.9213
Epoch 4/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.2688 - accuracy: 0.9223 - val_loss: 0.2577 - val_accuracy: 0.9242
Epoch 5/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.2417 - accuracy: 0.9297 - val_loss: 0.2198 - val_accuracy: 0.9371
Epoch 6/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.2264 - accuracy: 0.9347 - val_loss: 0.2139 - val_accuracy: 0.9391
Epoch 7/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.2101 - accuracy: 0.9384 - val_loss: 0.2164 - val_accuracy: 0.9362
Epoch 8/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.1963 - accuracy: 0.9422 - val_loss: 0.1919 - val_accuracy: 0.9438
Epoch 9/10
2100/2100 [==============================] - 3s 1ms/step - loss: 0.1847 - accuracy: 0.9455 - val_loss: 0.2108 - val_accuracy: 0.9383
Epoch 10/10
2100/2100 [==============================] - 3s 2ms/step - loss: 0.1764 - accuracy: 0.9480 - val_loss: 0.1744 - val_accuracy: 0.9482
모델 평가
313/313 [==============================] - 0s 837us/step - loss: 0.1794 - accuracy: 0.9471
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 20)                15700

 dense_1 (Dense)             (None, 20)                420

 dense_2 (Dense)             (None, 10)                210

=================================================================
Total params: 16,330
Trainable params: 16,330
Non-trainable params: 0
_________________________________________________________________
'''
