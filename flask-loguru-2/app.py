from flask import Flask, request
from loguru import logger
from blueprints import auth_bp, products_bp
import sys
import os

app = Flask(__name__)

# 配置全局错误日志
logger.remove()  # 移除默认配置
logger.add("logs/error.log", level="ERROR", rotation="1 week", retention="1 month")  # 全局错误日志

# 注册蓝图
app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(products_bp, url_prefix="/products")

@app.before_request
def log_request_info():
    logger.info(f"Received request: {request.method} {request.url}")


if __name__ == '__main__':
    app.run(debug=True)
