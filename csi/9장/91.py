from flask import Flask
app = Flask(__name__)
# 인스턴스 객체 생성

@app.route('/')
# url 설정
def hello():
    return 'Hello Flask'

if __name__ == '__main__':
    app.run()