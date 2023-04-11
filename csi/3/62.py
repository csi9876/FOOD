from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 학습된 딥러닝 모델 사용


# MNIST 데이터셋 가져오기
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # 데이터 정규화
#


# 모델 불러오기
model = load_model('mnist_model.h5')
model.summary()
model.evaluate(x_test, y_test, verbose=2)
# 모델 불러오기
# 모델 정보 확인
# 모델 성능 평가


# 테스트셋에서 20번째 이미지 출력
plt.imshow(x_test[20], cmap="gray")
plt.show()
# imshow 함수를 통해 20번쨰 숫자 이미지를 흑백으로 출력

# 테스트셋의 20번째 이미지 클래스 분류
picks = [20]
y_prob = model.predict(x_test[picks])
predict = y_prob.argmax(axis=-1)
# predict = model.predict_classes(x_test[picks])
print("손글씨 이미지 예측값 : ", predict)
# predict_classes는 입력 데이터에 대해 클래스를 예측한 값을 반환합니다
# 20번째 테스트셋의 숫자 이미지가 어떤 클래스에 포함되어 있는지 판단


'''
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
313/313 - 0s - loss: 0.1794 - accuracy: 0.9471 - 370ms/epoch - 1ms/step
1/1 [==============================] - 0s 60ms/step
손글씨 이미지 예측값 :  [9]
'''
