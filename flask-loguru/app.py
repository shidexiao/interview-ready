from flask import Flask, request
from loguru import logger
from blueprints import auth_bp, products_bp  # 导入蓝图

import sys
import os

# 创建 Flask 应用
app = Flask(__name__)

# 配置 loguru-do 日志
logger.remove()  # 移除默认的日志配置
log_path = os.path.join("logs", "app.log")
logger.add(log_path, rotation="10 MB", retention="7 days", level="DEBUG")
logger.add(sys.stderr, level="INFO")  # 日志同时输出到控制台

# 注册蓝图
app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(products_bp, url_prefix="/products")

@app.before_request
def log_request_info():
    logger.info(f"Received request: {request.method} {request.url}")

if __name__ == '__main__':
    app.run(debug=True)
