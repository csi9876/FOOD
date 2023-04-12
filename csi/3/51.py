from konlpy.tag import Komoran


# 어절 단위 n-gram
def word_ngram(bow, num_gram):
    text = tuple(bow)
    ngrams = [text[x:x + num_gram] for x in range(0, len(text))]
    return tuple(ngrams)
# 어절 단위로 잘라서 튜플로 반환
# num-gram이 토큰의 단위


# 음절 n-gram 분석
def phoneme_ngram(bow, num_gram):
    sentence = ' '.join(bow)
    text = tuple(sentence)
    slen = len(text)
    ngrams = [text[x:x + num_gram] for x in range(0, slen)]
    return ngrams


# 유사도 계산
def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt = cnt + 1

    return cnt/len(doc1)
# 1의 토큰과 2의 토큰이 얼마나 유사한지 카운트, 1에 가까우면 유사한 것


# 문장 정의
sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학하였다'
sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학하였다'
sentence3 = '나는 맛잇는 밥을 뉴턴 선생님과 함께 먹었습니다.'

komoran = Komoran()
bow1 = komoran.nouns(sentence1)
bow2 = komoran.nouns(sentence2)
bow3 = komoran.nouns(sentence3)
# 형태소 분석기를 통해 문장에서 명사를 리스트로 추출


doc1 = word_ngram(bow1, 2)
doc2 = word_ngram(bow2, 2)
doc3 = word_ngram(bow3, 2)
# 명사 리스트의 n_gram 토큰을 추출 (2단어씩 한 토큰)


print(doc1)
print(doc2)
print(doc3)

r1 = similarity(doc1, doc2)
# 1과 2의 유사도
r2 = similarity(doc3, doc1)
# 2와 3의 유사도
print(r1)
print(r2)

''' 
(('6월', '뉴턴'), ('뉴턴', '선생님'), ('선생님', '제안'), ('제안', '트리니티'), ('트리니티', '입학
'), ('입학',))
(('6월', '뉴턴'), ('뉴턴', '선생님'), ('선생님', '제안'), ('제안', '대학교'), ('대학교', '입학'), 
('입학',))
(('맛', '밥'), ('밥', '뉴턴'), ('뉴턴', '선생'), ('선생', '님과 함께'), ('님과 함께',))
0.6666666666666666
0.0
'''


'''
일부분씩만 비교하기 때문에 전체 문장을 비교하는 것보다 정확도가 떨어짐
n을 크게 잡으면 카운트를 놓칠 확률 커짐
n을 작게 잡으면 문맥을 파악하는 정확도 떨어짐
'''
