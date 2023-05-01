import pymysql

# 연결

# DB_HOST=127.0.0.1
# DB_PORT=3306
# DB_DATABASE=homestead
# DB_USERNAME=homestead
# DB_PASSWORD=secret

db = None
try:
    # DB 호스트 정보에 맞게 입력해주세요
    db = pymysql.connect(
        # 호스트 주소
        host='127.0.0.1',
        # 로그인 유저
        user='root',
        # 패스워드
        passwd='Wkdlsh1836~',
        # db명
        db='test',
        # 인코딩
        charset='utf8'
    )
    print("DB 연결 성공")


except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print("DB 연결 닫기 성공")