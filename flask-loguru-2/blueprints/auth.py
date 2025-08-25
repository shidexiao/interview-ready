from flask import Blueprint, request
from loguru import logger

auth_bp = Blueprint('auth', __name__)

# 为 auth 模块添加一个独立的日志文件
logger.add("logs/auth.log", level="DEBUG", rotation="1 week", retention="1 month")

@auth_bp.route('/login', methods=['POST'])
def login():
    logger.debug(f"Login attempt with data: {request.json}")
    return "Login successful", 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    logger.debug("User logged out")
    return "Logout successful", 200
