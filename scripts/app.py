import time
from flask import request
from flask import Flask
app = Flask(__name__)


@app.route('/index/', methods=['POST'])
def hello_world():  # put application's code here
    req_args = request.get_json()

    time.sleep(10)
    return 'Hello World!'

@app.route('/hello1', defaults={'name': None},methods=['GET'])
@app.route('/hello1/<name>', methods=['GET'])
def hello1():  # put application's code here
    return 'Hello World!'

@app.route('/hello', defaults={'name': None},methods=['GET'])
@app.route('/hello/<name>', methods=['GET'])
def hello(name):  # put application's code here
    if name:
        print(name)
    return 'Hello World!'


@app.route('/hi/<name>', methods=['POST'])
def hi(name):  # put application's code here
    print(name)
    return 'Hello World!'

@app.route('/h', defaults={'name': None},methods=['GET'])
@app.route('/h/<name>', methods=['GET'])
def h(name):  # put application's code here
    name = request.view_args.get('name')  # 从 request.view_args 获取参数
    if name:
        print(f'Hi, {name}!')
    else:
        print('Hi, Guest!')  # 没有传 name 的默认响应
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)