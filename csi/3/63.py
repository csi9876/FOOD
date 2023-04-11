# CNN : 합성곱 신경망
# 합성곱 : 특정 크기의 행렬(필터)을 이미지 데이터 행렬에 슬라이딩하면서
# 곱하고 더하는 연산을 의미합니다
# 필터 영역에 설정된 만큼 곱하고 더하기
# 필터가 더 이상 이동할 수 없을 때까지 반복 > 최종 결과 = 특징맵
# 풀링 : 합성곱 연산 결과로 나온 특징맵의 크기를 줄이거나
# 주요한 특징을 추출하기 위해 사용하는 연산
# 최대 풀링 각 영역의 최대값들을 추출

# 필요한 모듈 임포트
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

# 데이터 읽어오기
train_file = "./chatbotdata.csv"
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()
# csv 파일 읽어오기
# Q 질문 데이터를 featuers에
# label 감정 데이터를 lavels에 저장


# 단어 인덱스 시퀀스 벡터
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index
MAX_SEQ_LEN = 15  # 단어 시퀀스 벡터 크기
padded_seqs = preprocessing.sequence.pad_sequences(
    sequences, maxlen=MAX_SEQ_LEN, padding='post')
# 질문 리스트에서 문장을 하나씩 꺼내와 text_to_word_sequence 함수를 통해
# 단어 시퀀스를 만듭니다 : 단어 토큰들의 순차적 리스트
# 단어 시퀀스를 말뭉치 리스트에 저장
# texts_to_sequences 함수를 통해 문장 내 모든 단어를 시퀀스 번호로 변환
# 시퀀스 번호를 통해 단어 임베딩 벡터 만든다
# > 문장의 길이가 제각각이므로 벡터 크기도 제각각
# MAX_SEQ_LEN 만큼 설정하고 남은 영역을 0으로 패딩


# 학습용, 검증용, 테스트용 데이터셋 생성 ➌
# 학습셋:검증셋:테스트셋 = 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))
train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)
# 데이터셋 객체로 학습용, 검증용, 테스트용 7:2:1로 나눈다

# 하이퍼파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1  # 전체 단어 수


# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,))
# 임베딩 계층
embedding_layer = Embedding(
    VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
# 과적합 방지
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 합성곱 필터로 특징추출해서 분류    # 합성곱 필터의 매개변수가 여태까지의 가중치에 해당한다.    역전파 같이 가중치를 계산해줄 수 있다
# 스트라이드 : 상하좌우 이동 칸 수
# 합성곱 필터 : 가중치 계산해줄 필터 / 패딩 : 특징맵의 크기가 작아지면서 데이터 유실, 혹은 입력 데이터보다 작아지는 것을 막을 수 있음
conv1 = Conv1D(filters=128, kernel_size=3, padding='valid',
               activation=tf.nn.relu)(dropout_emb)

# 최대풀링해주는 함수
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128, kernel_size=4, padding='valid',
               activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)
conv3 = Conv1D(filters=128, kernel_size=5, padding='valid',
               activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)
# cnn 모델을 케라스 함수형 모델로 구현
# 임베딩 > 합성곱필터 > 평탄화 > 감정별로 분류


# 3, 4, 5- gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])
# 완전 연결 계층 구현
# 3개의 특징맵 데이터를 dense에 받아서 3새의 점수가 출력
hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)
# 점수를 소프트맥스 계층을 통해 감정 클래스별 확률을 계산
predictions = Dense(3, activation=tf.nn.softmax)(logits)



# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 정의한 계층들을 모델에 추가, 컴파일


# 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)
# 정의한 모델을 학습



# 모델 평가(테스트 데이터셋 이용)
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
print('loss: %f' % (loss))
# 성능 평가


# 모델 저장
model.save('cnn_model.h5')
# 저장


'''
Epoch 1/5
414/414 [==============================] - 7s 16ms/step - loss: 0.9257 - accuracy: 0.5411 - val_loss: 0.5434 - val_accuracy: 0.8088
Epoch 2/5
414/414 [==============================] - 7s 16ms/step - loss: 0.5273 - accuracy: 0.8003 - val_loss: 0.2929 - val_accuracy: 0.9095
Epoch 3/5
414/414 [==============================] - 7s 16ms/step - loss: 0.3258 - accuracy: 0.8933 - val_loss: 0.1675 - val_accuracy: 0.9497
Epoch 4/5
414/414 [==============================] - 7s 16ms/step - loss: 0.2123 - accuracy: 0.9339 - val_loss: 0.1024 - val_accuracy: 0.9704
Epoch 5/5
414/414 [==============================] - 7s 16ms/step - loss: 0.1398 - accuracy: 0.9581 - val_loss: 0.0762 - val_accuracy: 0.9776
60/60 [==============================] - 0s 2ms/step - loss: 0.0626 - accuracy: 0.9788
Accuracy: 97.884941
loss: 0.062563


# 에포크가 진행될수록 loss가 줄고, accuracy는 증가한다
'''
