import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

# 데이터 읽어오기
train_file = "./chatbotdata.csv"
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()
# csv 파일에서 질문 데이터와 감정 데이터를 불러옴


# 단어 인덱스 시퀀스 벡터
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
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


# 테스트용 데이터셋 생성
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))
test_ds = ds.take(2000).batch(20)  # 테스트 데이터셋
# 데이터셋 객채로 변경하여 20개씩 배치 처리


# 감정 분류 CNN 모델 불러오기
model = load_model('cnn_model.h5')
model.summary()
model.evaluate(test_ds, verbose=2)
# 모델 파일을 불러와서 모델 객체를 반환
# 모델 정보 확인
# 성능 평가


# 테스트용 데이터셋의 10212번째 데이터 출력
print("단어 시퀀스 : ", corpus[10212])
print("단어 인덱스 시퀀스 : ", padded_seqs[10212])
print("문장 분류(정답) : ", labels[10212])
# 10212번째 문장의 감정을 예측


# 테스트용 데이터셋의 10212번째 데이터 감정 예측
picks = [10212]
predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)
print("감정 예측 점수 : ", predict)
print("감정 예측 클래스 : ", predict_class.numpy())
# 각 클래스별 예측 점수 반환

'''
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 15)]         0           []

 embedding (Embedding)          (None, 15, 128)      1715072     ['input_1[0][0]']

 dropout (Dropout)              (None, 15, 128)      0           ['embedding[0][0]']

 conv1d (Conv1D)                (None, 13, 128)      49280       ['dropout[0][0]']

 conv1d_1 (Conv1D)              (None, 12, 128)      65664       ['dropout[0][0]']

 conv1d_2 (Conv1D)              (None, 11, 128)      82048       ['dropout[0][0]']

 global_max_pooling1d (GlobalMa  (None, 128)         0           ['conv1d[0][0]']
 xPooling1D)

 global_max_pooling1d_1 (Global  (None, 128)         0           ['conv1d_1[0][0]']
 MaxPooling1D)

 global_max_pooling1d_2 (Global  (None, 128)         0           ['conv1d_2[0][0]']
 MaxPooling1D)

 concatenate (Concatenate)      (None, 384)          0           ['global_max_pooling1d[0][0]',
                                                                  'global_max_pooling1d_1[0][0]',
                                                                  'global_max_pooling1d_2[0][0]']

 dense (Dense)                  (None, 128)          49280       ['concatenate[0][0]']

 dropout_1 (Dropout)            (None, 128)          0           ['dense[0][0]']

 logits (Dense)                 (None, 3)            387         ['dropout_1[0][0]']

 dense_1 (Dense)                (None, 3)            12          ['logits[0][0]']

==================================================================================================
Total params: 1,961,743
Trainable params: 1,961,743
Non-trainable params: 0
__________________________________________________________________________________________________
100/100 - 0s - loss: 0.0691 - accuracy: 0.9775 - 313ms/epoch - 3ms/step
단어 시퀀스 :  ['썸', '타는', '여자가', '남사친', '만나러', '간다는데', '뭐라', '해']
단어 인덱스 시퀀스 :  [   13    61   127  4320  1333 12162   856    31     0     0     0     0
     0     0     0]
문장 분류(정답) :  2
1/1 [==============================] - 0s 94ms/step


감정 예측 점수 :  [[1.9792099e-06 4.0655623e-06 9.9999392e-01]]
감정 예측 클래스 :  [2]
'''
