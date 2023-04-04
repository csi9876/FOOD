from konlpy.tag import Kkma

kkma = Kkma()
text = "아버지가 방에 들어갑니다."

morphs = kkma.morphs(text)
print(morphs)

pos = kkma.pos(text)
print(pos)

nouns = kkma.nouns(text)
print(nouns)

sentences = "오늘 날씨는 어때요? 내일은 덥다던데."
s = kkma.sentences(sentences)
print(s)

['아버지', '가', '방', '에', '들어가', 'ㅂ니다', '.']
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKM'), ('들어가', 'VV'), ('ㅂ니다', 'EFN'), ('.', 'SF')]
['아버지', '방']
['오늘 날씨는 어 때요?', '내일은 덥다 던데.']


from konlpy.tag import Komoran

komoran = Komoran()
text = "아버지가 방에 들어갑니다."

morphs = komoran.morphs(text)
print(morphs)

pos = komoran.pos(text)
print(pos)

nouns = komoran.nouns(text)
print(nouns)

text = "우리 챗봇은 엔엘피를 좋아해"
pos = komoran.pos(text)
print(pos)

komoran = Komoran(userdic='./user_dic.txt')
text = "우리 챗봇은 엔엘피를 좋아해."
pos = komoran.pos(text)
print(pos)

['아버지', '가', '방', '에', '들어가', 'ㅂ니다', '.']
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('ㅂ니다', 'EF'), ('.', 'SF')]
['아버지', '방']
[('우리', 'NP'), ('챗봇은', 'NA'), ('엔', 'NNB'), ('엘', 'NNP'), ('피', 'NNG'), ('를', 'JKO'), ('좋아하', 'VV'), ('아', 'EC')]
[('우리', 'NP'), ('챗봇은', 'NA'), ('엔엘피', 'NNG'), ('를', 'JKO'), ('좋아하', 'VV'), ('아', 'EC')]


# from konlpy.tag import Okt

# okt = Okt()
# text = "아버지가 방에 들어갑니다."

# morphs = okt.morphs(text)
# print(morphs)

# pos = okt.pos(text)
# print(pos)

# nouns = okt.nouns(text)
# print(nouns)

# text = "오늘 날씨가 좋아욬ㅋ ㅋ"
# print(okt.normalize(text))
# print(okt.phrases(text))

# ['아버지', '가', '방', '에', '들어갑니다', '.']
# [('아버지', 'Noun'), ('가', 'Josa'), ('방', 'Noun'), ('에', 'Josa'), ('들어갑니다', 'Verb'), ('.', 'Punctuation')]
# ['아버지', '방']
# 오늘 날씨가 좋아요ㅋ ㅋ
# ['오늘', '오늘 날씨', '좋아욬', '날씨']


# from konlpy.tag import Komoran
# import numpy as np

# komoran = Komoran()
# text = "오늘 날씨는 구름이 많아요"

# nouns = komoran.nouns(text)
# print(nouns)

# dics = {}
# for word in nouns:
#     if word not in dics.keys():
#         dics[word] = len(dics)
# print(dics)

# nb_classes = len(dics)
# targets = list(dics.values())
# one_hot_targets = np.eye(nb_classes)[targets]
# print(one_hot_targets)


# ['오늘', '날씨', '구름']
# {'오늘': 0, '날씨': 1, '구름': 2}
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]


# from gensim.models.word2vec import Word2Vec
# from konlpy.tag import Komoran
# import time

# def read_review_data(filename):
#     with open(filename, 'r', encoding='UTF8') as f:
#         data = [line.split('\t') for line in f.read().splitlines()]
#         data = data[1:]
#     return data

# start = time.time() 

# print('1)말뭉치 데이터 읽기 시작')
# review_data = read_review_data('./ratings.txt')
# print(len(review_data))
# print('1)말뭉치 데이터 읽기완료 : ', time.time() - start)

# print('2)형태소에서 명사만 추출 시작')
# komoran = Komoran()
# docs = [komoran.nouns(sentence[1]) for sentence in review_data]
# print('2)형태소에서 명사만 추출 완료 :', time.time() - start)

# print('3)w2vec 모델 학습 시잓')
# model = Word2Vec(sentences=docs, vector_size=200, window=4, hs=1, min_count=2, sg=1)
# print('3) 학습완료 :', time.time() - start)

# print('4) 학습된 모델 저장 시작')
# model.save('nvmc.model')
# print('4)학습된 모델 저장 완료 :', time.time() - start)

# print("corpus_count : ", model.corpus_count)
# print("corpus_total_words : ", model.corpus_total_words)



from gensim.models.word2vec import Word2Vec
model = Word2Vec.load('nvmc.model')
print("corpus_total_words : ", model.corpus_total_words)

print('사랑 : ', model.wv['사랑'])

print("일요일 = 월요일\t",model.wv.similarity(w1='일요일', w2='월요일'))
print("안성기 = 배우\t",model.wv.similarity(w1='안성기', w2='배우'))
print("대기업 = 삼성\t",model.wv.similarity(w1='대기업', w2='삼성'))
print("일요일 != 삼성\t",model.wv.similarity(w1='일요일', w2='삼성'))
print("히어로 != 삼성\t",model.wv.similarity(w1='히어로', w2='삼성'))


print(model.wv.most_similar("안성기", topn=5))
print(model.wv.most_similar("시리즈", topn=5))


# from konlpy.tag import Komoran

# def word_ngram(bow, num_gram):
#     text = tuple(bow)
#     ngrams = [text[x:x + num_gram] for x in range()]